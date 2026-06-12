//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is sealed onto the [`ProfileIdentity`] at its construction (one fold over
//! `(config, max_settle, events)`) and is the lifetime-stable identity of the Profile. `ProfileMap`
//! keeps `(resource, config_hash) → ProfileId` and updates `Resource.profiles` in lockstep —
//! `attach`/`detach` are the only mutators of either index.

use crate::ids::{ProbeCorrelation, ProfileId, ResourceId, TimerId};
use crate::op::ProofAuthority;
use crate::probe::ProbeSlot;
use crate::resource::ResourceKind;
use crate::scan_config::{ProfileIdentity, ScanConfig};
use crate::snapshot::tree::TreeSnapshot;
use crate::sub::ClassSet;
use crate::tree::Tree;
use compact_str::CompactString;
use slotmap::{SecondaryMap, SlotMap};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One fire cycle, split by the fire-transition boundary.
///
/// A burst lives `Idle → Active(ActiveBurst) → Idle`. The fire transition (`Verifying → Awaiting`)
/// is a typed state-machine move from [`PreFireBurst`] to [`PostFireBurst`]: the two sides have
/// disjoint valid mutators, valid timers, valid probe responses, and accumulator semantics.
/// Encoding the split at the type level means a field that has no post-fire consumer (e.g.
/// `forced`, `last_event_time`) cannot leak across the boundary by construction.
///
/// **Pre-fire** (`Batching | Verifying | Draining`): event-driven debounce window, in-flight verify
/// or self-stable / descendants-pending idle. Carries the event-provenance accumulator (`dirty`)
/// and the settle-deadline source of truth (`last_event_time`). Quiescence is folded by
/// [`quiescence_verdict`] at the dispatch — the engine's two observation channels (walker authority
/// C1, event-quiet witness C2, the `forced` bit) determine the verdict at the floor, so the burst
/// carries no per-sample fold state.
///
/// **Post-fire** (`Awaiting | Rebasing | Settling`): the actuator gate, then the *structural
/// mirror* of the pre-fire loop — `Settling ⇄ Rebasing` is `Batching ⇄ Verifying`, bounded by its
/// own [`CeilingState`] lifecycle (pre-fire's `(burst_deadline, forced)` analogue, collapsed into a
/// sum), over the *post-command* tree. The same fold floor ([`quiescence_verdict`]) computes the
/// post-fire verdict from the rebase response's `(authority, forced)` pair — `forced` projected
/// from [`CeilingState::Reached`] at the `profile_probe_gate` read — and no prior sample carries
/// across the fire. The pre-fire fields that encode a fire decision do not cross the boundary — the
/// typed [`PreFireBurst::into_post_fire`] move drops them, and the `BurstDeadline` timer becomes
/// structurally irrelevant ([`PostFireBurst::timer_token`] folds it to `None` for post-fire phases,
/// so the engine's stale-drain lazily collects the heap entry). Its one fresh accumulator is the
/// post-fire `dirty`, which `absorb_event_into_fire_tail` feeds; it is not a proof-obligation
/// source (the `WholeSubtree` walk observes everything regardless), only the fire-tail residual
/// restart seed, reset at every `Rebasing` re-entry so a `Stable` terminal restarts only on the
/// genuine final-window race.
#[derive(Debug)]
pub enum ActiveBurst {
    PreFire(PreFireBurst),
    PostFire(PostFireBurst),
}

/// Event provenance accumulated across a burst's pre-fire life (and, for the post-fire fire-tail,
/// the residual restart seed).
///
/// Key = the live engine slot the event named. Value = that slot's path, `Arc::clone`d at ingest
/// from the already-resolved live `&Resource` (the `watch_demand > 0` gate proved the slot live).
/// Where an event landed is a *historical fact* — immutable from the instant of ingest and immune
/// to the slot later being reaped (delete-recreate at the same path). A reaped key never
/// invalidates its captured path.
///
/// The Standard pre-fire proof obligation derives from the **values**, never the keys:
/// [`Self::chains`] is the dirty root→leaf chains the walker must freshly observe, and
/// [`Self::lca_path`] is their component-wise lowest common ancestor — the tightest directory the
/// probe can root at without excluding a chain. Sourcing both from the captured paths is what makes
/// an empty `Chains` over a fully reaped-id set unconstructable: liveness never filters the
/// projection.
///
/// The map is keyed by the slot, not reduced to a bare path set, for two reasons: per-slot
/// **last-writer-wins** dedup (a slot firing N events contributes one entry, not N — see
/// [`Self::note`]), and retaining the live-slot id as the cheap basis for any future caller needing
/// *current* liveness rather than history (today none on the Standard pre-fire path — the projection
/// reads only the values). No public setter — [`Self::note`] is the sole accumulator edge.
#[derive(Debug, Default)]
pub struct DirtyProvenance(BTreeMap<ResourceId, Arc<Path>>);

impl DirtyProvenance {
    /// An empty accumulator. `const` for the `burst.rs` constructors and the typed post-fire move.
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Record that an `FsEvent` named `id` at `path`. The sole accumulator edge. `path` is an
    /// `Arc::clone` of the live `&Resource`'s materialised path captured at the ingest site — total
    /// by construction (the `watch_demand > 0` gate proved the slot live), so no fallible
    /// `path_of`, no `Option`. Last-writer -wins per id; ids are stable, so a repeat event for one
    /// slot re-stores the identical path.
    pub fn note(&mut self, id: ResourceId, path: Arc<Path>) {
        self.0.insert(id, path);
    }

    /// No event recorded yet. The Seed first-fire witness (`seed_owes_first_fire`) and the
    /// fire-tail residual restart gate read this; a Standard pre-fire burst is non-empty by
    /// construction (its constructor notes the trigger).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Drop every recorded event. Crate-private — the asymmetric clear is the post-fire side's
    /// privilege and is owned by the typed edge-method [`PostFireBurst::reset_residual`]; the `pub`
    /// mutator surface on [`Self`] ([`Self::note`], [`Self::is_empty`], [`Self::chains`],
    /// [`Self::lca_path`]) is symmetric across pre-fire and post-fire and stays shared.
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// The dirty root→leaf chains for [`crate::ProofObligation::Chains`]: every captured path,
    /// `BTreeSet`-ordered for deterministic replay. Every captured path is at-or-under the burst's
    /// probe target by construction (the target is the live id at [`Self::lca_path`], or the anchor
    /// fallback — both ancestors-or-equal of every value), so no "intersect with the target subtree"
    /// filter is needed: it would be a tautology. Never empty for a Standard pre-fire burst.
    #[must_use]
    pub fn chains(&self) -> BTreeSet<Arc<Path>> {
        self.0.values().map(Arc::clone).collect()
    }

    /// The component-wise lowest common ancestor of every captured path — the tightest directory
    /// the walker can root at without excluding a chain. `None` iff empty.
    ///
    /// Component-wise (not byte-prefix) is load-bearing: `/a` must not match `/ab`. Sound because v1
    /// forbids symlinks / cross-filesystem, so a shared component prefix is genuine Tree ancestry. A
    /// lone captured path (the dominant single-file-edit case) returns itself with no allocation; the
    /// engine resolves the result to a live id and promotes a File leaf to its parent Dir.
    #[must_use]
    pub fn lca_path(&self) -> Option<Arc<Path>> {
        let mut values = self.0.values();
        let first = values.next()?;
        let mut lca: &Path = first;
        for p in values {
            lca = common_prefix(lca, p);
        }
        if lca == first.as_ref() {
            Some(Arc::clone(first))
        } else {
            Some(Arc::from(lca))
        }
    }
}

/// Longest shared **component** prefix of two paths, borrowed from `a`. Walks `Path::components` in
/// lockstep, then strips `a`'s trailing components past the divergence via `Path::parent` (each
/// step a sub-slice of `a`, so the result keeps `a`'s lifetime). Component -wise, so `/a` is never
/// a prefix of `/ab`. Both inputs are absolute (materialised from the root chain) and share at
/// least the root, so the result is never empty.
fn common_prefix<'a>(a: &'a Path, b: &Path) -> &'a Path {
    let shared = a
        .components()
        .zip(b.components())
        .take_while(|(x, y)| x == y)
        .count();
    let total = a.components().count();
    let mut prefix = a;
    for _ in shared..total {
        prefix = prefix.parent().unwrap_or(prefix);
    }
    prefix
}

/// Sealed N=2 hash-channel sample carrier — the source of the verdict floor's
/// [`QuiescenceWitness::HashChannel`] `prior` field. Owned by both [`PreFireBurst`] and
/// [`PostFireBurst`] (a separate sample sequence each, across the fire boundary): `None` until the
/// first Authoritative sample advances it, `Some(hash)` after.
///
/// The newtype *is* the single-writer discipline. The inner `Option` is private and [`Self::advance`]
/// is the sole mutation; both that and [`Self::fresh`] are `pub(crate)`, so the engine — which holds
/// a blanket `&mut` to the burst for cat-a phase swaps — has no syntactic name for the field and
/// cannot reset it. The `transition_to_settling` carrier-clobber footgun is a compile error, not a
/// silent quiescence regression. Mirrors [`DirtyProvenance`]'s `pub(crate)`-clear seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CertifiedSample(Option<u128>);

impl CertifiedSample {
    /// A carrier that has observed no sample yet — the birth value at every burst construction. No
    /// construction path may seed a prior sample (the no-bypass discipline): the post-command /
    /// post-rebase tree a fresh burst samples is a *different* tree, so carrying a hash across
    /// would be a category error.
    #[must_use]
    pub(crate) const fn fresh() -> Self {
        Self(None)
    }

    /// Record `hash` as the current Authoritative sample and return the prior (`None` on the first
    /// sample). The returned prior is the [`QuiescenceWitness::HashChannel`] `prior` input. The
    /// `Authoritative`-only contract sits at the caller (the verdict choke in
    /// `certify_probe_response`, reached via [`Profile::advance_certified_sample`]); the carrier
    /// itself is a total `&mut` mutation with no phase gate, since its lifetime is the burst's.
    #[must_use]
    pub(crate) const fn advance(&mut self, hash: u128) -> Option<u128> {
        let prior = self.0;
        self.0 = Some(hash);
        prior
    }
}

/// Sealed monotone fold latch — the frozen "this burst folds instead of fires" decision, owned by
/// [`PreFireBurst`] only. Born from the operator's birth consult, set (never cleared) by the
/// reverse-race retro-latch [`PreFireBurst::latch_fold`], and read at verdict time via
/// [`ProfileState::burst_fold_latched`].
///
/// Like [`CertifiedSample`], the newtype *is* the seal: the inner `bool` is private and the
/// constructor / mutator / reader are all `pub(crate)`, so the engine's blanket cat-a `&mut` cannot
/// flip the latch outside the retro-latch edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FoldLatch(bool);

impl FoldLatch {
    /// The birth value — the operator's `absorb` window state consulted at the burst's birth instant
    /// ([`Profile::absorb_window_live`]). `true` iff a window was already live, so the burst is born
    /// folding; the reverse race (operator arms *after* birth) is handled by [`Self::latch`].
    #[must_use]
    pub(crate) const fn born(consulted_live: bool) -> Self {
        Self(consulted_live)
    }

    /// Set the latch — monotone (set-only) and idempotent under re-arm. The sole in-life mutation,
    /// driven by the [`ActiveBurst::latch_fold`] cascade for the reverse race.
    pub(crate) const fn latch(&mut self) {
        self.0 = true;
    }

    /// Whether this burst folds its terminal verdict instead of firing.
    #[must_use]
    pub(crate) const fn is_latched(self) -> bool {
        self.0
    }
}

/// Pre-fire lifecycle — every phase before the fire transition.
///
/// Fields are split across three roles:
/// - **Burst-scoped invariants** (`intent`, `forced`, `burst_deadline`): survive every pre-fire
///   phase transition.
/// - **Pre-fire event state** (`dirty`, `last_event_time`): populated by `event_drives_batching` on
///   every `FsEvent`, for both intents (both burst constructors are Batching-first). `dirty`'s
///   captured paths are the obligation + scope basis re-projected at each `transition_to_verifying`
///   for a Standard burst, and live-but-inert for a Seed (anchor target + `WholeSubtree`);
///   `last_event_time` is the settle deadline's source of truth for both.
/// - **Phase-resident facts** ([`PreFirePhase::Verifying`]'s `target`): the probe's resource target
///   lives on the variant's payload, so it exists exactly when a probe does. No placeholder field
///   carries across `Batching` / `Draining` where no probe is in flight.
///
/// `dirty` is preserved across the burst's pre-fire lifetime because the obligation + scope are
/// re-projected from it at every reconfirm (`Draining → Verifying`) — the *projection* mutates, the
/// captured -path *basis* doesn't.
///
/// Quiescence is folded at the dispatch by [`quiescence_verdict`] — a pure projection of
/// `ProofAuthority` (the walker certificate) and the burst's `forced` flag (the bounded
/// ceiling-bypass). Any per-burst inputs the fold consumes home on this struct, so they survive the
/// in-place phase swaps of the pre-fire lifetime; struct-local fields can't leak across the typed
/// move to [`PostFireBurst`] at the fire boundary.
///
/// `last_event_time` is the source of truth for the settle deadline: the settle timer is scheduled
/// once on Batching entry and reschedules on expiry only when `last_event_time` has advanced since.
/// Event arrivals while already in Batching update this field but do **not** re-insert a fresh heap
/// entry — heap inserts are bounded to one per `last_event_time + settle` boundary, regardless of
/// event density. Seeded `Some(burst-start)` by *both* burst constructors (both are
/// Batching-first); the `Option` survives only because `on_settle_expired` reads it defensively and
/// folds a `None` straight to the `Verifying` transition.
#[derive(Debug)]
pub struct PreFireBurst {
    pub burst_deadline: TimerId,
    pub phase: PreFirePhase,
    pub intent: BurstIntent,
    pub forced: bool,
    /// Event provenance — every `FsEvent` that drove (or is driving) this burst, captured `(slot,
    /// path)` at ingest. Constructed with the trigger by *both* `start_standard_burst` (always —
    /// its trigger is mandatory) and `start_seed_burst` (iff the Seed has a triggering `FsEvent`;
    /// empty otherwise), then `event_drives_batching` notes each later FsEvent during the pre-fire
    /// phases (`Batching | Verifying | Draining`), for *both* intents.
    ///
    /// **Two intent-specific consumers.**
    /// - *Standard* projects the captured **paths** to the `ProofObligation::Chains` and their
    ///   component-LCA (resolved to a live id by `pre_fire_target`) to the probe target — both
    ///   immune to slot reaping because they read history, not current liveness.
    /// - *Seed* targets the anchor and carries `ProofObligation::WholeSubtree` unconditionally, so
    ///   this is **not** its probe-target / obligation source; instead its *non-emptiness is the
    ///   first-fire witness*. A fresh, never-fired Seed fires its `SubtreeRoot` Subs iff it
    ///   observed activity (`!dirty.is_empty()`, the engine's `seed_owes_first_fire` gate); empty ⇔
    ///   no activity ⇔ restart-safe silent pin (a daemon restart over a static tree must not
    ///   re-fire — Specter persists no baseline, so every restart is a fresh Seed). A recovery Seed
    ///   (`any_fired`) ignores this and uses the drift oracle instead.
    pub dirty: DirtyProvenance,
    /// Wall-clock instant of the most recent `FsEvent` that drove this burst. The **source of
    /// truth** for the Batching settle deadline: the live settle timer's heap entry pins to a fixed
    /// deadline (`burst-start + settle`, or `prior_last_event + settle` after a reschedule), but
    /// the deadline the burst is *waiting for* is `last_event_time + settle`. The on-expiry
    /// reschedule check reconciles the two — if `now − last_event_time < settle` the expiry handler
    /// reschedules a fresh entry at `last_event_time + settle` and stays in Batching; otherwise it
    /// transitions to Verifying.
    ///
    /// **Three construction states.** `Option<Instant>` is genuinely 2D — `None` is a first-class
    /// burst-start shape on the cold path, not a defensive fallback only `on_settle_expired` reads:
    ///
    /// - `Some(now)` from `start_standard_burst` — the burst-start `FsEvent` is the first event and
    ///   seeds the field.
    /// - `Some(now)` from `start_seed_burst(Some(trigger))` — a triggering `FsEvent` drove the
    ///   re-Seed (the `drive_burst` Idle+!baseline path). `seed_owes_first_fire` reads the
    ///   non-empty `dirty` as the activity witness; the burst-start instant seeds the settle
    ///   deadline exactly as Standard.
    /// - `None` from `start_seed_burst(None)` — cold attach. No driving event drove this burst; no
    ///   `Batching` phase exists at construction (the cold path is Verifying-at-construction), so
    ///   there is no settle deadline to source. An `FsEvent` arriving during the cold walk routes
    ///   through `event_drives_batching`, which Cancels the verify slot, writes `last_event_time =
    ///   Some(now)`, schedules a fresh settle_timer, and re-enters `Batching` — the field becomes
    ///   meaningful exactly when a deadline exists to source.
    ///
    /// **Updaters.**
    /// - `event_drives_batching` on every event (`Some(now)`).
    /// - `retry_drives_batching` **pins to `Some(now)`** — the verify just responded, and pinning
    ///   the timestamp removes the `Instant` monotonicity dependency from the reschedule
    ///   correctness argument.
    /// - **Preserved** across `transition_to_verifying` (the reconfirm path) and
    ///   `transition_to_draining` — phase swaps without semantic resets.
    pub last_event_time: Option<Instant>,
    /// Consecutive *motion-observing* [`QuiescenceVerdict::Retry`] re-entries (hash-channel
    /// disagreement, `observed_motion: true`) with no intervening driving event. Born `0`;
    /// `retry_drives_batching` increments only when the Retry observed motion —
    /// a transient walker refusal or probe failure observed *nothing* and holds the streak —
    /// and `event_drives_batching` zeroes on every driving `FsEvent` (a delivered in-mask event
    /// explains the observed motion and breaks the streak). A surviving streak therefore witnesses
    /// event-*silent* windows that each *hashed differently*: the disagreement-denominated count
    /// the hint floor was calibrated against. Read once, at the forced-ceiling terminal with the
    /// disagreement bit set: a streak at-or-above the engine's hint floor upgrades the generic
    /// despite-change diagnostic to [`crate::Diagnostic::ChangeOutsideEventMask`] (the proof
    /// object is moving via change classes outside the Profile's `events` mask). Dropped by
    /// omission at [`Self::into_post_fire`]: the rebase loop counts its own windows over the
    /// post-command tree ([`PostFireBurst::retry_streak`]).
    pub retry_streak: u32,
    /// Pre-fire N=2 sample carrier — see [`CertifiedSample`] for the sealed single-writer contract.
    /// Engaged (read at the verdict floor) only when the burst owes quiescence proof (Standard,
    /// triggered Seed, post-recovery Seed) AND [`Profile::events_witness_quiescence`] is `false`;
    /// otherwise born fresh and never advanced — the fold consumes
    /// [`QuiescenceWitness::EventsReliable`] instead. Dropped by omission at
    /// [`Self::into_post_fire`]: the post-fire side opens its own
    /// [`PostFireBurst::last_certified_hash`] sequence over the post-command tree.
    pub(crate) last_certified_hash: CertifiedSample,
    /// Frozen "fold instead of fire" decision — see [`FoldLatch`] for the sealed monotone-latch
    /// contract. When set, a would-be-firing verdict is overridden to a silent baseline advance
    /// ([`crate::Diagnostic::QuiescenceAbsorbed`]).
    ///
    /// **Orthogonal to [`Self::intent`].** Intent feeds the proof-obligation axis
    /// ([`Profile::event_chains_prove_quiescence`] — events-complete `Standard ⇒ Chains`, all else
    /// ⇒ `WholeSubtree`); a fold-latched burst still runs its probe semantics in full and changes
    /// only the *terminal consequence*. **Dropped by omission** at
    /// [`Self::into_post_fire`] — a fold replaces the fire, so a latched burst must never cross the
    /// boundary; the move debug-asserts `!latched` as the structural dual of the verdict-time
    /// override.
    pub(crate) fold_latched: FoldLatch,
}

/// Pre-fire phase discriminator.
///
/// `Batching` carries its own correlation token (`settle_timer: TimerId`) because timer correlation
/// is per-Burst and has no peer slot to live on. `Verifying` carries a [`ProbeSlot`] and the
/// probe's resource target: the pre-fire stability probe's liveness, identity, *and* scope all live
/// on the phase, so a verify in flight without a correlation or without a target is unconstructable
/// and I5 ("at most one outstanding probe") is a representability property of the single slot.
/// `Draining` carries no correlation token of its own: its exit is driven by a fresh query over the
/// live tree ([`ProfileState::in_active_standard_burst`]), swept at every `finish_burst_to_idle` —
/// no per-phase token, no cached counter.
#[derive(Debug)]
pub enum PreFirePhase {
    /// Activity-gap detection. `settle_timer` is the armed debounce timer; an `FsEvent` reschedules
    /// it (`event_drives_batching`), timer expiry advances to `Verifying`
    /// (`transition_to_verifying`).
    Batching { settle_timer: TimerId },
    /// Pre-fire stability probe.
    ///
    /// `slot` is armed with the correlation the response must echo while the probe is in flight; it
    /// is empty only for the transient post-Cancel window before the burst re-arms `Batching`
    /// (`event_drives_batching`). Consuming the response disarms the slot exactly once — the
    /// structural consume-once guarantee.
    ///
    /// `target` is the resource id the probe was scoped to, computed at construction by
    /// `pre_fire_target` and immutable for the variant's lifetime. For events-complete Standard
    /// bursts ([`Profile::event_chains_prove_quiescence`]): the live id at the component-LCA of
    /// `dirty`'s captured paths (File leaf promoted to its parent Dir; anchor on any resolution
    /// miss). For Seed bursts (triggered or cold-walk) and events-incomplete Standard bursts: the
    /// Profile's anchor. The Verifying response reads this for the post-fire snapshot-commit
    /// target.
    ///
    /// Constructing the variant *requires* both fields, so a verify phase without a correlation or
    /// without a target cannot exist:
    ///
    /// ```compile_fail
    /// use specter_core::{PreFirePhase, ProbeSlot};
    /// // Missing `target` — the struct literal is incomplete.
    /// let _ = PreFirePhase::Verifying { slot: ProbeSlot::empty() };
    /// ```
    ///
    /// ```
    /// use specter_core::{PreFirePhase, ProbeSlot, ResourceId};
    /// let _ = PreFirePhase::Verifying {
    ///     slot: ProbeSlot::empty(),
    ///     target: ResourceId::default(),
    /// };
    /// ```
    Verifying { slot: ProbeSlot, target: ResourceId },
    /// Self-stable; descendants pending. The stable snapshot lives on `Profile.current` —
    /// `fire_or_seal` commits `current` to the stable response immediately before classification,
    /// so the tree-reconcile / Watch side keeps a faithful baseline. The reconfirm probe (Draining
    /// → Verifying, fired by the `finish_burst_to_idle` sweep once no covered descendant is still
    /// in an Active Standard burst) folds its verdict through [`quiescence_verdict`] over the fresh
    /// `(authority, forced)` pair — never against `Profile.current` — so the verdict does not
    /// depend on the splice-mutated snapshot. Holding a duplicate `TreeSnapshot` on the variant
    /// would only invite drift between the two references.
    Draining,
}

/// Post-fire lifecycle — the structural mirror of [`PreFireBurst`].
///
/// Post-fire runs its own quiescence loop over the *post-command* tree, so it mirrors the pre-fire
/// shape: a loop bound ([`CeilingState`], the post-fire analogue of pre-fire's `burst_deadline` +
/// `forced` pair) and a `last_event_time` (mirror of the pre-fire field of the same name), captured
/// by `absorb_event_into_fire_tail` on every absorbed FsEvent. The pre-fire fields that encode a
/// *fire decision* do not cross the boundary, dropped by leaving them out of
/// [`PreFireBurst::into_post_fire`]:
/// - `forced`: the pre-fire `forced` bit decided the pre-burst fire over the pre-command tree; the
///   post-fire ceiling latch ([`CeilingState::Reached`]) is a disjoint decision over the
///   post-command tree. The two decisions don't carry across, and the post-fire side opens a fresh
///   [`CeilingState::NotStarted`].
/// - No `burst_deadline`: the pre-fire ceiling; the post-fire one is carried by
///   [`CeilingState::Armed`]. The stale pre-fire timer lazy-drops via
///   [`PostFireBurst::timer_token`]'s `Settle | BurstDeadline` arm.
/// - No probe target on the post-fire side: Rebasing always targets the Profile's anchor (the
///   variant carries the `ProbeSlot` only).
///
/// The pre-fire `dirty` (the captured-path basis) also does not cross; the post-fire
/// `final_window_residual` is a *distinct, freshly-empty* provenance accumulator, not the pre-fire
/// one carried over. `last_event_time` likewise opens fresh (`None`) — the absorb tail reckons from
/// its own first absorbed event.
///
/// `intent: BurstIntent` survives post-fire so `dispatch_rebase_{vanished,failed}` can tag the
/// `ProbeVanished` / `ProbeFailed` diagnostic with it (Seed-driven drift rebases and
/// Standard-driven post-fire rebases both reach PostFire, and the diagnostic distinguishes them).
/// It is also the field [`ProfileState::in_active_standard_burst`] reads — the reconfirm query
/// treats a post-fire Standard burst as still covering its ancestors for the burst's full lifetime.
/// The fire-tail residual restart is **not** gated on it: the reconfirm is a fresh query, not a
/// per-origin refcount, so a Seed origin restarts just as a Standard one does.
///
/// **Single construction seam.** Every `PostFireBurst` is born fresh — `ceiling:
/// CeilingState::NotStarted`, `last_certified_hash: None` — through [`Self::new`];
/// [`PreFireBurst::into_post_fire`] (the typed fire move) is its only production caller. The
/// post-command tree is a *different tree* than the one the pre-fire burst observed, so neither the
/// pre-fire N=2 sample carrier (`PreFireBurst::last_certified_hash`) nor any other pre-fire fold
/// input carries across the fire: the rebase loop opens its own independent sample sequence over
/// the post-command tree.
#[derive(Debug)]
pub struct PostFireBurst {
    pub intent: BurstIntent,
    pub phase: PostFirePhase,
    /// The final-window restart seed — events absorbed during the post-fire tail (`Awaiting |
    /// Rebasing | Settling`), captured `(slot, path)` by `absorb_event_into_fire_tail` in
    /// `drive_burst`'s post-fire arm. Single-purpose: when the rebase loop terminates
    /// `Authoritative` on a `ReturnToIdle` burst with a non-empty residual, restart a fresh
    /// debounced Standard burst seeded from it (`into_pre_fire_residual` moves the whole
    /// provenance, so the restarted burst's first verify has its captured paths intact). A zombie
    /// (`Reap`) burst, an empty residual, or a ceiling terminal (no restart) drops it at
    /// `finish_burst_to_idle`. The restarted burst's settle window reckons from the rebase-response
    /// instant, not the absorbed events', a bounded ≤ one-`settle` extra re-fire latency.
    ///
    /// **Per-entry reset.** Cleared at *every* `Rebasing` entry (`transition_to_rebasing`, both the
    /// first `Awaiting → Rebasing` walk and each `Settling → Rebasing` re-arm), so when the loop
    /// terminates the residual holds only events from the **final** probe round-trip — the genuine
    /// final-window race (a change observed by the sensor's certifying walk's instant but after the
    /// engine could fold it). Without this per-entry reset, any tree-touching command would leave a
    /// non-empty residual and spuriously restart; with it the restart fires only for the real race.
    ///
    /// **Not a proof-obligation source.** The rebase probe walks `WholeSubtree`, so earlier-round
    /// absorbs are folded into the rebase verdict directly by the walk — never read off this
    /// accumulator.
    pub final_window_residual: DirtyProvenance,
    /// Wall-clock instant of the most recent `FsEvent` absorbed into this post-fire burst by
    /// `absorb_event_into_fire_tail` (or the `Rebasing → Settling` transition instant via
    /// `transition_to_settling` — the HashChannel spacing window's deadline source of truth). The
    /// post-fire mirror of [`PreFireBurst::last_event_time`]; born `None` (the absorb tail reckons
    /// from its own first absorbed event, not from the fire instant).
    ///
    /// **Writers** (cat-a, both `engine/burst.rs`):
    /// - `absorb_event_into_fire_tail` — on every absorbed event, exactly mirroring
    ///   `event_drives_batching`'s pre-fire write.
    /// - `transition_to_settling` — at the sole Settling entry (`Rebasing → Settling` undischarged
    ///   loop-back), pinning `Some(now)` so the spacing window's quiet-check is anchored on the
    ///   transition instant rather than a stale absorb instant.
    ///
    /// **Reader.** `handle_post_fire_settle_expired` consumes the timestamp to decide reschedule vs
    /// transition, mirroring `on_settle_expired`'s pre-fire fork.
    pub last_event_time: Option<Instant>,
    /// Consecutive *motion-observing* rebase-loop [`QuiescenceVerdict::Retry`] re-entries
    /// (hash-channel disagreement, `observed_motion: true`) with no intervening absorbed event —
    /// the post-fire mirror of [`PreFireBurst::retry_streak`], counting this loop's own windows
    /// over the post-command tree. Born `0`; `transition_to_settling` increments only when the
    /// Retry observed motion (a transient refusal or probe failure holds the streak),
    /// `absorb_event_into_fire_tail` zeroes on every absorbed `FsEvent`. Read once, at the
    /// `RebaseCeiling` forced terminal with the disagreement bit set, for the same
    /// [`crate::Diagnostic::ChangeOutsideEventMask`] upgrade. Dropped by omission at
    /// [`Self::into_pre_fire_residual`].
    pub retry_streak: u32,
    /// The rebase-loop ceiling lifecycle — the post-fire mirror of [`PreFireBurst::forced`] + the
    /// pre-fire `burst_deadline` pair, collapsed into a single sum type. See [`CeilingState`] for
    /// the three valid states and the algorithmic-edge writers.
    ///
    /// Folded into [`quiescence_verdict`] at the dispatch as `forced = matches!(self.ceiling,
    /// CeilingState::Reached)`. The fold is the only response-path consumer; the
    /// [`Self::timer_token`] projection for [`TimerKind::RebaseCeiling`] reads the
    /// [`CeilingState::Armed`] payload.
    pub ceiling: CeilingState,
    /// Post-fire N=2 sample carrier — see [`CertifiedSample`] — for the rebase loop's `WholeSubtree`
    /// samples. A rebase commits a new baseline and can never commit a mid-write state, so the
    /// post-fire loop always owes quiescence proof: [`Profile::events_witness_quiescence`] `== false`
    /// is the *sole* engagement gate (for events-reliable Profiles the field is born fresh and never
    /// advanced; the fold consumes [`QuiescenceWitness::EventsReliable`] instead). Dropped by
    /// omission at [`Self::into_pre_fire_residual`] — the restarted pre-fire burst opens its own
    /// sequence over the post-rebase tree.
    pub(crate) last_certified_hash: CertifiedSample,
}

/// Post-fire phase discriminator — the structural mirror of [`PreFirePhase`].
///
/// `Awaiting` has no pre-fire peer (the actuator gate); `Settling ⇄ Rebasing` is the post-fire
/// `Batching ⇄ Verifying` loop, bounded by `rebase_ceiling`.
///
/// `Awaiting { outstanding, gate_deadline }`: effects emitted, counter decrements on each
/// `EffectComplete` for this Profile's `DedupKey`s. Reaching zero advances to `Settling` (or, when
/// the burst carries [`BurstFinish::Reap`], finishes the burst directly). `gate_deadline` is the
/// recovery timer for an actuator that never reports completion — its expiry forces the burst into
/// `Rebasing` (skipping `Settling`: the bounded wait has already elapsed) or, on a zombie burst,
/// directly into [`crate::ProfileState::Idle`] via reap.
///
/// `Rebasing` carries a [`ProbeSlot`]: the post-fire baseline-capture probe's liveness *and*
/// identity live on the phase, so a rebase in flight without its correlation is unconstructable.
/// The rebase response folds through [`quiescence_verdict`]; a [`QuiescenceVerdict::Stable`]
/// verdict rebases `baseline := current` and finishes (or restarts on a non-empty residual), a
/// [`QuiescenceVerdict::Retry`] verdict loops back through `Settling`.
///
/// `Settling { settle_timer }`: settle-sized spacing wait between rebase samples — the `Rebasing ⇄
/// Settling` retry loop, entered only on a [`QuiescenceVerdict::Retry`]. The post-fire mirror of
/// [`PreFirePhase::Batching`] in its retry-spacing role (the natural rebase entry is probe-first,
/// so `Settling` debounces only the retry loop, not the command's own event tail). No
/// [`ProbeSlot`]: no probe is in flight during the spacing window (the slot lives on `Rebasing`),
/// only the settle timer. `absorb_event_into_fire_tail` updates [`PostFireBurst::last_event_time`]
/// on every absorbed `FsEvent`; `handle_post_fire_settle_expired` reads the same field on expiry
/// and either reschedules (events arrived since the timer was scheduled) or transitions to
/// `Rebasing`. `settle_timer` is the phase's correlation token, exactly as `Batching`'s is.
#[derive(Debug)]
pub enum PostFirePhase {
    Awaiting {
        outstanding: u32,
        gate_deadline: TimerId,
    },
    /// Post-fire baseline-capture probe at the anchor. The [`ProbeSlot`] holds the correlation the
    /// rebase response must echo while it is in flight; the single disarm at response dispatch is
    /// the consume-once guarantee. The variant requires the slot, so a rebase phase without a
    /// correlation is unrepresentable:
    ///
    /// ```compile_fail
    /// use specter_core::PostFirePhase;
    /// let _: PostFirePhase = PostFirePhase::Rebasing;
    /// ```
    Rebasing(ProbeSlot),
    /// Settle-sized spacing wait between rebase samples (the `Rebasing ⇄ Settling` retry loop,
    /// entered only on a [`QuiescenceVerdict::Retry`]) — the post-fire mirror of
    /// [`PreFirePhase::Batching`] in its retry-spacing role. `settle_timer` is the live settle
    /// deadline; absorbed `FsEvent`s update [`PostFireBurst::last_event_time`], and on expiry
    /// `handle_post_fire_settle_expired` reschedules if events arrived since, otherwise drives
    /// `Settling → Rebasing` for the next sample. No [`ProbeSlot`]: a stray `EffectComplete` /
    /// probe response here is a late, untracked arrival (folded to the same routing as `Rebasing`).
    Settling { settle_timer: TimerId },
}

/// The rebase-loop ceiling lifecycle — three valid states.
///
/// The `(Armed + Reached)` pair the prior two-field shape (`forced: bool` + `rebase_ceiling:
/// Option<TimerId>`) flagged as algorithmically unreachable is now unrepresentable.
///
/// The post-fire analogue of pre-fire's `(burst_deadline: TimerId, forced: bool)` pair: a single
/// in-life timer reference held while the ceiling is armed, then a terminal latch the next probe
/// response folds through [`quiescence_verdict`] over `(authority, forced = true)`.
///
/// **Two writers, one edge each.** Both cat-a in `engine/burst.rs`:
/// - `Engine::arm_rebase_loop_ceiling` — [`Self::NotStarted`] → [`Self::Armed`] at the natural
///   `Awaiting → Rebasing` entry. Single caller: `on_effect_complete::LastReached + ReturnToIdle`.
/// - `Engine::force_pending_post_fire` — [`Self::Armed`] → [`Self::Reached`] (natural ceiling
///   expiry; the prior timer reference is dropped at the same write) or [`Self::NotStarted`] →
///   [`Self::Reached`] (gate-deadline-recovery latches the ceiling without an in-heap timer entry).
///
/// The single-source falsifiability grep is one line: `rg 'self\.ceiling = ' --type rust crates/` —
/// expect exactly the two writers plus the burst-fresh default in `PostFireBurst::new`.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CeilingState {
    /// Pre-`Settling` entry — no ceiling timer exists yet. The burst-fresh default at
    /// [`PostFireBurst::new`].
    NotStarted,
    /// Ceiling timer live in the heap. Reachable only via `Engine::arm_rebase_loop_ceiling`'s sole
    /// edge, from [`Self::NotStarted`]. The payload is the [`TimerId`] the `Engine`'s `pop_expired`
    /// resolves the heap entry against; the post-fire [`PostFireBurst::timer_token`] projection for
    /// [`TimerKind::RebaseCeiling`] reads it.
    Armed(TimerId),
    /// Ceiling fired (`handle_rebase_ceiling`'s `force_pending_post_fire` call, from [`Self::Armed`])
    /// OR gate-deadline-recovery latched the ceiling without arming a timer (`handle_gate_deadline`'s
    /// non-zombie arm, from [`Self::NotStarted`]). Both routes land here; the next probe response
    /// folds through [`quiescence_verdict`] over `(authority, forced = true)`.
    Reached,
}

/// Verdict of one `EffectComplete` against the post-fire counter.
///
/// Three variants, not a `bool`, because the route is resolved from the same call: "decremented,
/// still in flight" vs "last completion" vs "not even Awaiting" must each be representable.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AwaitVerdict {
    /// Decremented, still `> 0` — more in flight; stay Awaiting.
    Decremented,
    /// Hit zero (pre-decrement `≤ 1`) — last completion; caller routes on.
    LastReached,
    /// Not `Active(PostFire(Awaiting))` — a late/untracked completion.
    NotAwaiting,
}

/// The quiescence-proof channel that applies to one certified probe response.
///
/// Parallel to [`ProofAuthority`] (which proves *accuracy* of the snapshot): the witness encodes
/// *which* of the two safety channels discharges quiescence at the verdict floor.
///
/// - [`Self::EventsReliable`] — settle-window silence alone discharges quiescence; the hash channel
///   is bypassed structurally (no carrier read, no comparison). This variant deliberately *doubles*
///   over two epistemically distinct situations that fold identically, so no dispatch consumer
///   distinguishes them — collapsing them is the over-discrimination axiom in force, not an
///   oversight:
///   - the Profile's `events_union` covers in-place writes (see
///     [`Profile::events_witness_quiescence`]), so the stream is a sufficient witness and silence
///     *proves* quiescence; or
///   - the burst's consequence owes no quiescence proof at all (cold-Seed `SilentPin`) — nothing to
///     prove, so any witness trivially suffices.
/// - [`Self::HashChannel`] — events-incomplete fire-bearing burst. Quiescence requires two
///   consecutive Authoritative samples to agree on `leaf_hash` / `dir_hash`. `prior` is the
///   burst-resident carrier read (`None` on first sample); `response` is the current observation's
///   hash. Equality `Some(prior) == response` is the stability witness; disagreement (including
///   `prior = None`) routes the verdict to [`QuiescenceVerdict::Retry`].
///
/// Constructed at the verdict choke (`certify_probe_response`); consumed by [`quiescence_verdict`].
/// The two paths are explicit at the call site — `Option<HashChannel>`-like alternatives would
/// elide the meaning behind anonymous wrapping.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QuiescenceWitness {
    /// Settle-window silence is sufficient — no hash channel engaged. Doubles as "no proof owed"
    /// (cold-Seed `SilentPin`) and "stream is a sufficient witness"; both meanings fold identically.
    EventsReliable,
    /// Hash-equality channel: `Stable` iff `prior == Some(response)`.
    HashChannel { prior: Option<u128>, response: u128 },
}

impl QuiescenceWitness {
    /// True iff the channel was active AND observed concrete sample-to-sample motion: `prior =
    /// Some(p)` with `p != response`. `EventsReliable` and the first-sample `prior = None` both
    /// answer `false` — there is no observed disagreement, only the absence of confirmation.
    ///
    /// The one witness test behind both motion-shaped verdict bits, so they cannot drift apart:
    /// [`StableReason::Forced`]'s `hash_channel_disagreed` (the forced-ceiling diagnostic
    /// selector) and [`QuiescenceVerdict::Retry`]'s `observed_motion` (the streak-counting fork).
    const fn observed_disagreement(self) -> bool {
        matches!(
            self,
            Self::HashChannel { prior: Some(p), response } if p != response,
        )
    }
}

/// The fire-path arm of [`QuiescenceVerdict::Stable`] — natural fire vs. bounded-ceiling fallback.
///
/// Pulled out as a sub-enum (not two `bool`s on `Stable`) so the impossible state `(forced=false,
/// hash_channel_disagreed=true)` is unrepresentable at the type level — that combination produces
/// [`QuiescenceVerdict::Retry`] at the fold, never a `Stable`.
///
/// - [`Self::Natural`] — settle-window silence held (the [`QuiescenceWitness::EventsReliable`]
///   path) OR the hash channel agreed on its current sample (`prior == Some(response)`). No ceiling
///   diagnostic owed.
/// - [`Self::Forced`] — `BurstDeadline` / `RebaseCeiling` fallback fired. Fire / rebase anyway
///   against the freshest observation. The dispatch maps `hash_channel_disagreed` to a diagnostic
///   asymmetrically: post-fire always emits a forced-ceiling diagnostic — the generic
///   [`crate::Diagnostic::RebaseCeilingForced`] carrying the bit as `observed_change` (loud on both
///   — no `Effect` records the forced fallback downstream); pre-fire diagnoses only on `true` and
///   stays silent on `false` because `forced` already propagates onto `Effect.forced`. On either
///   side, a `true` bit at the tail of a persistent event-silent retry streak (the burst's
///   `retry_streak` at-or-above the engine's hint floor) upgrades the generic diagnostic to
///   [`crate::Diagnostic::ChangeOutsideEventMask`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StableReason {
    /// Settle witness held — natural fire/pin/rebase path. No ceiling diagnostic owed.
    Natural,
    /// Bounded `max_settle` / `RebaseCeiling` fallback. Fire / rebase against the freshest
    /// observation; `hash_channel_disagreed` selects the diagnostic at the dispatch.
    Forced { hash_channel_disagreed: bool },
}

/// Verdict of one certified probe response — the pure projection of the engine's three-axis
/// dispatch decision `(ProofAuthority × forced × QuiescenceWitness)` onto the verdict floor.
///
/// - [`Self::Stable`] — walker certified AND quiescence proven. The inner [`StableReason`]
///   distinguishes natural fire from the bounded-ceiling fallback (and, on the latter, carries the
///   diagnostic-selection bit).
/// - [`Self::Retry`] — non-firing, non-terminal: either the walker certified but the hash channel
///   observed `prior != Some(response)` at this sample (events-incomplete fire-bearing burst), or
///   the walker refused on some chain (transient non-observation — `EACCES`, a chmod-000 chain) and
///   the bounded ceiling has not yet fired. Both origins route the same way at both dispatch sites
///   (pre-fire re-Batch via `Engine::retry_drives_batching`, post-fire re-Settle via
///   `Engine::transition_to_settling`); neither commits. `observed_motion` is the one bit on which
///   they diverge: only a concrete channel disagreement counts toward the burst's `retry_streak`
///   (the mask-blindspot witness) — a refusal observed nothing, so it holds the streak instead.
///   The transient `first_unread` is consumed only on the [`Self::Abandon`] terminal, and the
///   channel-disagreement provenance persists through the burst's `last_certified_hash` carrier for
///   the eventual forced-ceiling read. The bounded `BurstDeadline` / `RebaseCeiling` eventually
///   surfaces a [`StableReason::Forced`] (channel-disagreement path) or [`Self::Abandon`]
///   (walker-refused path) terminal.
/// - [`Self::Abandon`] — bounded terminal: the ceiling already fired and the walker still refused
///   on some chain (`first_unread`). No commit; the dispatch diagnoses `*CeilingUnreadable` and
///   finishes the burst.
///
/// Constructed solely by [`quiescence_verdict`]; the dispatch site consumes the variants and never
/// re-constructs. Not `Copy` — `Abandon` carries an `Arc<Path>`.
///
/// **Over-discrimination axiom.** Every variant must have a dispatch consumer that distinguishes it
/// from every other variant. A field with no consumer is over-discrimination — collapse to the
/// next-coarser variant. So `Retry` subsumes the unstable case rather than a separate `Unstable`
/// variant, and transient `Undischarged` carries no `first_unread` (the dispatch never reads it).
/// `Retry.observed_motion` earns its keep under the same axiom: the dispatch sites count vs hold
/// the streak on it. Auditable in one grep: every variant tag must appear in a dispatch arm whose
/// body diverges from at least one sibling.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum QuiescenceVerdict {
    /// Walker certified + quiescence proven. Fire / pin / rebase against the freshest observation;
    /// the inner [`StableReason`] captures the proof path (natural vs. bounded-ceiling fallback)
    /// and the diagnostic-selection bit on the latter.
    Stable(StableReason),
    /// Non-firing, non-terminal — loop back through the settle window for another sample. Subsumes
    /// two structurally-distinct origins (hash-channel disagreement; transient walker refusal) that
    /// share the same routing at both dispatch sites. `observed_motion` is `true` only on a
    /// concrete channel disagreement (`prior = Some(p)`, `p != response`) — the streak-counting
    /// fork the dispatch sites read (see the type-level docs); the fold is the only place the
    /// origin is knowable, exactly as for [`StableReason::Forced`]'s `hash_channel_disagreed`.
    Retry { observed_motion: bool },
    /// Bounded terminal: the ceiling already fired and the walker still refused on `first_unread`.
    /// The dispatch surfaces the unread path via `*CeilingUnreadable` and finishes the burst
    /// without committing — an unread region must never become the dedup / Seed baseline.
    Abandon { first_unread: Arc<Path> },
}

/// Fold the verdict-floor inputs into a [`QuiescenceVerdict`]. Total, pure, side-effect-free —
/// three axes (`authority × forced × witness`) projected to three variants.
///
/// - [`ProofAuthority::Undischarged`] + `forced` ⇒ [`QuiescenceVerdict::Abandon`] carrying
///   `first_unread` verbatim. The witness is irrelevant on this arm: an unread chain blocks the
///   fire regardless of any hash-channel observation, and the carrier was not advanced anyway (the
///   cat-(b) edge is Authoritative-only).
/// - [`ProofAuthority::Undischarged`] + `!forced` ⇒ [`QuiescenceVerdict::Retry`] with
///   `observed_motion: false` — a refusal observed nothing, so it holds the streak rather than
///   counting toward the mask-blindspot witness. `first_unread` is dropped at the fold (one
///   `Arc::drop` instead of clone-then-drop downstream): the transient arm at both dispatch sites
///   has no consumer for it today, and the carrier was not advanced.
/// - [`ProofAuthority::Authoritative`] + `forced` ⇒ [`QuiescenceVerdict::Stable`] with
///   [`StableReason::Forced`]. `hash_channel_disagreed` is `true` iff the channel was active AND
///   `prior` is `Some(p)` with `p != response` (the strong "tree was visibly still moving" signal).
///   `EventsReliable` and first-sample [`QuiescenceWitness::HashChannel`] (`prior = None`) both
///   fold to `false` — there is no observed disagreement, only the absence of confirmation.
/// - [`ProofAuthority::Authoritative`] + `!forced` ⇒ [`QuiescenceVerdict::Stable`]
///   ([`StableReason::Natural`]) iff the witness proves quiescence
///   ([`QuiescenceWitness::EventsReliable`] OR `HashChannel { prior: Some(p), response }` with `p ==
///   response`). Otherwise (`HashChannel` with `prior = None` OR `prior != Some(response)`) ⇒
///   [`QuiescenceVerdict::Retry`], whose `observed_motion` runs the same witness test as the
///   Forced arm's `hash_channel_disagreed` — `true` on a concrete `Some(p) != response`, `false`
///   on the first-sample `None` (absence of confirmation, not observed motion). The
///   channel-disagreement provenance persists through the burst's `last_certified_hash` carrier;
///   the eventual forced-ceiling read reconstructs the strong-signal
///   `*CeilingForcedDespiteChange` if disagreement persists, so no operator-visible signal is lost.
#[must_use]
pub fn quiescence_verdict(
    authority: ProofAuthority,
    forced: bool,
    witness: QuiescenceWitness,
) -> QuiescenceVerdict {
    match authority {
        ProofAuthority::Undischarged { first_unread } if forced => {
            QuiescenceVerdict::Abandon { first_unread }
        }
        ProofAuthority::Undischarged { .. } => {
            // `first_unread` dropped at the fold — the transient arm at both dispatch sites has no
            // consumer for it, and the carrier was not advanced (the cat-(b) edge is
            // Authoritative-only). One `Arc::drop` instead of clone-then-drop downstream. The
            // refusal observed nothing, so the streak holds.
            QuiescenceVerdict::Retry {
                observed_motion: false,
            }
        }
        ProofAuthority::Authoritative if forced => {
            // Ceiling bypass: fire / rebase against freshest observation. The diagnostic-selection
            // bit is the shared witness test — see `QuiescenceWitness::observed_disagreement`.
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: witness.observed_disagreement(),
            })
        }
        ProofAuthority::Authoritative => match witness {
            QuiescenceWitness::EventsReliable => QuiescenceVerdict::Stable(StableReason::Natural),
            QuiescenceWitness::HashChannel {
                prior: Some(p),
                response,
            } if p == response => QuiescenceVerdict::Stable(StableReason::Natural),
            // The unforced twin of the Forced arm's bit, computed by the same shared test: a
            // concrete `Some(prior) != response` is observed motion; the first-sample `None` is
            // absence of confirmation (the agree case was consumed by the arm above).
            w @ QuiescenceWitness::HashChannel { .. } => QuiescenceVerdict::Retry {
                observed_motion: w.observed_disagreement(),
            },
        },
    }
}

impl PreFireBurst {
    /// The `TimerId` armed on this burst for `kind`, or `None` if the pre-fire shape doesn't carry
    /// a slot for `kind`.
    ///
    /// Pre-fire owns:
    /// - [`TimerKind::Settle`] — lives on [`PreFirePhase::Batching`] only; the field is absent in
    ///   `Verifying`/`Draining` and the arm returns `None`.
    /// - [`TimerKind::BurstDeadline`] — non-Optional on [`PreFireBurst`]; always
    ///   `Some(self.burst_deadline)`.
    /// - [`TimerKind::AwaitGateDeadline`] / [`TimerKind::PostFireSettle`] /
    ///   [`TimerKind::RebaseCeiling`] — type-impossible here (these fields live on
    ///   [`PostFireBurst`] only); the arms return `None` to encode the structural absence.
    ///
    /// Consumed via the [`ActiveBurst`] / [`ProfileState`] delegation chain by the engine's
    /// stale-timer filter; each layer only enumerates the kinds its data shape can actually carry,
    /// so the type-impossible pairs fold to `None` at the leaf without a wildcard fallthrough.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match kind {
            TimerKind::Settle => match &self.phase {
                PreFirePhase::Batching { settle_timer } => Some(*settle_timer),
                PreFirePhase::Verifying { .. } | PreFirePhase::Draining => None,
            },
            TimerKind::BurstDeadline => Some(self.burst_deadline),
            TimerKind::AwaitGateDeadline | TimerKind::PostFireSettle | TimerKind::RebaseCeiling => {
                None
            }
        }
    }

    /// Construct a pre-fire burst — the single construction seam.
    ///
    /// Born fresh, always: `forced` is `false` (the force-fire flag flips in-life only on
    /// `BurstDeadline` expiry, via the engine's cat-a `force_pending`), `last_certified_hash` opens
    /// `CertifiedSample::fresh` (sole in-life writer: the cat-(b)
    /// [`Self::advance_certified_sample`]), and `retry_streak` opens `0` (cat-a writers only:
    /// `retry_drives_batching` increments, `event_drives_batching` zeroes). Those invariant-bearing
    /// fields take no parameter precisely because *no* construction path may seed them — the
    /// no-bypass discipline applied to construction, mirroring [`PostFireBurst::new`].
    ///
    /// `fold_latched` *is* a parameter — the operator's birth consult
    /// ([`Profile::absorb_window_live`] at the burst's birth instant), a computed construction value
    /// like `intent`; its only in-life writer is the reverse-race retro-latch [`Self::latch_fold`].
    ///
    /// Production callers: the engine's `start_seed_burst` / `start_standard_burst`, and
    /// [`PostFireBurst::into_pre_fire_residual`] (the typed residual restart, which threads its own
    /// birth consult).
    #[must_use]
    pub const fn new(
        burst_deadline: TimerId,
        phase: PreFirePhase,
        intent: BurstIntent,
        dirty: DirtyProvenance,
        last_event_time: Option<Instant>,
        fold_latched: bool,
    ) -> Self {
        Self {
            burst_deadline,
            phase,
            intent,
            forced: false,
            dirty,
            last_event_time,
            retry_streak: 0,
            last_certified_hash: CertifiedSample::fresh(),
            fold_latched: FoldLatch::born(fold_latched),
        }
    }

    /// Advance the pre-fire N=2 sample carrier — the sole in-life mutator of
    /// `Self::last_certified_hash`. Records `hash` as the current Authoritative sample and returns
    /// the prior value (`None` on first sample). The returned prior threads through the cat-(b)
    /// cascade to the verdict choke as the [`QuiescenceWitness::HashChannel`] `prior` input.
    ///
    /// **Authoritative-only contract.** Callers gate on [`crate::ProofAuthority`] before reaching
    /// this writer — an Undischarged observation must not advance the carrier (its hash would not
    /// reflect a faithful read of every obligation chain). The gate sits at the caller (the verdict
    /// choke in `certify_probe_response`), not on this writer; the writer is a total function on
    /// the burst, mirroring [`PostFireBurst::note_effect_completion`]'s no-public-setter-floor
    /// discipline.
    ///
    /// **No phase gate.** The carrier's lifetime is the burst's lifetime (preserved across every
    /// pre-fire phase swap), so the writer takes `&mut PreFireBurst` and writes unconditionally.
    /// `Verifying` is the only structurally reachable phase at the verdict choke (a
    /// response-bearing transition); the cat-(b) edge does not re-enforce that, by design.
    #[must_use]
    pub const fn advance_certified_sample(&mut self, hash: u128) -> Option<u128> {
        self.last_certified_hash.advance(hash)
    }

    /// Set the fold latch — the retro-latch leaf of the [`ActiveBurst::latch_fold`] cascade.
    /// Set-only (monotone) and idempotent: an operator arming a window over an already-running
    /// pre-fire burst flips it once and a re-arm is a no-op. The sole in-life writer of
    /// `Self::fold_latched`; construction sets the field directly from the birth consult. Total
    /// `&mut self`, no phase gate — the carrier's lifetime is the burst's lifetime, mirroring
    /// [`Self::advance_certified_sample`].
    pub const fn latch_fold(&mut self) {
        self.fold_latched.latch();
    }

    /// Typed move from pre-fire to post-fire — the fire transition.
    ///
    /// Drops, by leaving them out of the [`PostFireBurst::new`] construction this delegates to:
    /// - `burst_deadline` — lazy-dropped by [`PostFireBurst::timer_token`]'s `None` arm once it
    ///   expires post-fire; the post-fire loop has its own ceiling.
    /// - `forced` — the pre-fire `forced` bit decided the pre-burst fire. The post-fire side opens
    ///   its own `forced: false`; the rebase-loop ceiling latch ([`PostFireBurst::ceiling`]) is a
    ///   disjoint decision over the post-command tree.
    /// - Pre-fire probe target — homed on [`PreFirePhase::Verifying`]'s payload, so it dies with
    ///   the pre-fire phase. Rebasing always targets the anchor.
    /// - `dirty` — pre-fire-only event state. Post-fire opens a *fresh, empty*
    ///   `final_window_residual` (the fire-tail residual), not the pre-fire captured-path provenance.
    /// - `last_event_time` — the pre-fire settle-deadline source. Post-fire opens its own
    ///   `last_event_time = None`; the absorb tail reckons from its own first absorbed event, not
    ///   the fire instant.
    /// - `last_certified_hash` — the pre-fire N=2 sample carrier. Post-fire opens its own
    ///   `PostFireBurst::last_certified_hash` `= None` for an independent rebase-loop sample
    ///   sequence over the post-command tree (a different tree than the one the pre-fire carrier
    ///   sampled, so cross-carrying a hash would be a category error).
    /// - `retry_streak` — counts pre-fire event-silent windows over the pre-command tree; the
    ///   rebase loop opens its own `0` for the same cross-tree reason as the sample carrier.
    /// - `fold_latched` — pre-fire-only. A fold *replaces* the fire, so a latched burst must never
    ///   reach this move; the entry `debug_assert` is the structural dual of the verdict-time
    ///   `AbsorbFold` override, fail-stopping a classify-routing bug. Post-fire has no latch.
    ///
    /// `intent` is preserved (read by `dispatch_rebase_*` for the diagnostic).
    ///
    /// `outstanding: NonZeroU32` carries the "a fire emitted ≥1 Effect" invariant as a type: a
    /// post-fire burst is born `Awaiting` at least one completion. The stored
    /// `Awaiting.outstanding` is `u32` (it decrements to zero via `note_effect_completion`); only
    /// the birth param is non-zero. The zero case never reaches this move — `fire_and_settle`
    /// routes it to `finish_burst_to_idle` instead.
    ///
    /// Sole production caller: `transition_to_awaiting` in `burst.rs`.
    #[must_use]
    pub fn into_post_fire(self, outstanding: NonZeroU32, gate_deadline: TimerId) -> PostFireBurst {
        debug_assert!(
            !self.fold_latched.is_latched(),
            "into_post_fire: fold-latched burst must not fire — a latched \
             verdict folds to AbsorbFold (silent baseline advance), never \
             crosses the fire boundary",
        );
        PostFireBurst::new(
            self.intent,
            PostFirePhase::Awaiting {
                outstanding: outstanding.get(),
                gate_deadline,
            },
            DirtyProvenance::new(),
        )
    }
}

impl PostFireBurst {
    /// The `TimerId` armed on this burst for `kind`, or `None` if the post-fire shape doesn't carry
    /// a slot for `kind`.
    ///
    /// Post-fire owns:
    /// - [`TimerKind::AwaitGateDeadline`] — lives on [`PostFirePhase::Awaiting`]'s `gate_deadline`
    ///   field; `None` once the burst leaves `Awaiting` (the field doesn't exist on `Rebasing` /
    ///   `Settling`).
    /// - [`TimerKind::PostFireSettle`] — lives on [`PostFirePhase::Settling`]'s `settle_timer`
    ///   field; `None` in `Awaiting` / `Rebasing` (no settle window in flight).
    /// - [`TimerKind::RebaseCeiling`] — lives on the [`Self::ceiling`] field as the
    ///   [`CeilingState::Armed`] payload. The other two states ([`CeilingState::NotStarted`] /
    ///   [`CeilingState::Reached`]) hold no live timer and fold to `None` — covering both the
    ///   pre-arm state (no `Settling` entry yet) and the post-fire latched state (timer consumed by
    ///   `pop_expired`, the terminal bit now structurally [`CeilingState::Reached`]). The
    ///   just-expired ceiling id lazy-drops either way — `timer_token` is `&self`, it does not
    ///   observe the consume.
    /// - [`TimerKind::Settle`] / [`TimerKind::BurstDeadline`] — type-impossible here (the fields
    ///   were dropped at [`PreFireBurst::into_post_fire`]); the arm returns `None` to encode the
    ///   structural absence.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match kind {
            TimerKind::AwaitGateDeadline => match &self.phase {
                PostFirePhase::Awaiting { gate_deadline, .. } => Some(*gate_deadline),
                PostFirePhase::Rebasing(_) | PostFirePhase::Settling { .. } => None,
            },
            TimerKind::PostFireSettle => match &self.phase {
                PostFirePhase::Settling { settle_timer } => Some(*settle_timer),
                PostFirePhase::Awaiting { .. } | PostFirePhase::Rebasing(_) => None,
            },
            TimerKind::RebaseCeiling => match self.ceiling {
                CeilingState::Armed(t) => Some(t),
                CeilingState::NotStarted | CeilingState::Reached => None,
            },
            TimerKind::Settle | TimerKind::BurstDeadline => None,
        }
    }

    /// Construct a post-fire burst — the single construction seam.
    ///
    /// Born fresh, always: `ceiling` is [`CeilingState::NotStarted`] (no ceiling timer armed yet,
    /// no terminal latched), `last_event_time` is `None` (the absorb tail reckons from its own
    /// first absorbed event, not from the fire instant), `last_certified_hash` opens
    /// `CertifiedSample::fresh` — no pre-fire sample carries across the fire — and `retry_streak`
    /// opens `0` (the rebase loop counts its own event-silent windows). Those invariant-bearing
    /// fields take no parameter precisely because *no* construction path may seed them — the only
    /// mutations are the cat-a engine helpers (`arm_rebase_loop_ceiling`,
    /// `force_pending_post_fire`, `transition_to_settling`, `absorb_event_into_fire_tail` — each
    /// documented at its production writer) plus the cat-(b) carrier writer
    /// ([`Profile::advance_certified_sample`]), the no-bypass discipline applied to construction.
    ///
    /// Sole production caller: [`PreFireBurst::into_post_fire`] (the typed fire move).
    #[must_use]
    pub const fn new(
        intent: BurstIntent,
        phase: PostFirePhase,
        final_window_residual: DirtyProvenance,
    ) -> Self {
        Self {
            intent,
            phase,
            final_window_residual,
            last_event_time: None,
            retry_streak: 0,
            ceiling: CeilingState::NotStarted,
            last_certified_hash: CertifiedSample::fresh(),
        }
    }

    /// Reset the fire-tail residual — the typed edge-method on the owner for the sole asymmetric
    /// clear of `DirtyProvenance::clear`. Cross-crate callers reach the operation only through this
    /// method; the underlying `clear` is `pub(crate)`, and [`PreFireBurst`] exposes no analogue, so
    /// the "drop a fire-bearing burst's captured paths" footgun is structurally unrepresentable on
    /// the pre-fire side.
    ///
    /// **Sole caller.** `Engine::transition_to_rebasing` at every `Rebasing` entry (the per-entry
    /// residual reset documented at the caller — under `WholeSubtree` the residual is only the
    /// final-window restart seed, so clearing per entry keeps a `Stable` terminal from spuriously
    /// restarting on every tree-touching command).
    pub fn reset_residual(&mut self) {
        self.final_window_residual.clear();
    }

    /// Apply one `EffectComplete`, returning the zero-edge verdict. The sole in-life mutator of
    /// [`PostFirePhase::Awaiting`]'s `outstanding`: floor and decrement live here on the owner — a
    /// total fn with no public setter that returns the edge, so the invariant cannot be enforced at
    /// a distance. `Rebasing` / `Settling` ⇒ [`AwaitVerdict::NotAwaiting`] (the counter drained at
    /// the `Awaiting → Rebasing` edge; a completion arriving in the rebase loop is a late,
    /// untracked arrival). Underflow (more completions than emitted Effects) trips a
    /// `debug_assert!`, saturates in release.
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        match &mut self.phase {
            PostFirePhase::Awaiting { outstanding, .. } => {
                let prev = *outstanding;
                debug_assert!(
                    prev > 0,
                    "note_effect_completion: outstanding underflow — more \
                     EffectCompletes than emitted Effects",
                );
                *outstanding = prev.saturating_sub(1);
                if prev <= 1 {
                    AwaitVerdict::LastReached
                } else {
                    AwaitVerdict::Decremented
                }
            }
            PostFirePhase::Rebasing(_) | PostFirePhase::Settling { .. } => {
                AwaitVerdict::NotAwaiting
            }
        }
    }

    /// Advance the post-fire N=2 sample carrier — the structural mirror of
    /// [`PreFireBurst::advance_certified_sample`] for the rebase loop's sample sequence. Same
    /// `Authoritative`-only caller contract; same no-phase-gate writer shape; same
    /// no-public-setter-floor discipline shared with [`Self::note_effect_completion`]. Sole in-life
    /// mutator of `Self::last_certified_hash`.
    #[must_use]
    pub const fn advance_certified_sample(&mut self, hash: u128) -> Option<u128> {
        self.last_certified_hash.advance(hash)
    }

    /// Typed move from post-fire back to a fresh pre-fire `Batching` burst — the symmetric inverse
    /// of [`PreFireBurst::into_post_fire`].
    ///
    /// Consumes the post-fire burst at the rebase-ok boundary and re-arms a Standard debounce
    /// burst, moving the `final_window_residual` provenance over whole: the events
    /// `absorb_event_into_fire_tail` captured while the rebase probe was already in flight. Without
    /// this the residual has no consumer — it drops when the post-fire burst is torn down, so a
    /// descendant change that landed during the rebase round-trip is seen only by the next
    /// unrelated event. The move keeps the captured paths intact, so the restarted Standard burst's
    /// first verify obligates over them.
    ///
    /// **In-place move, never finish-then-start.** The typed `PostFire → PreFire` move preserves
    /// the watched anchor: it neither installs nor releases a contribution, so the restarted burst
    /// keeps the original burst's kernel-watch state without a finish/start round-trip. The single
    /// balancing `Unwatch` (if any) still runs at the restarted burst's eventual reap.
    ///
    /// **Origin-agnostic.** `intent` is *set* (not inherited) to `Standard` because a restarted
    /// debounce burst *is* Standard by definition. This is precisely where a Seed-origin fire-tail
    /// residual (Seed drift → fire → rebase, with events absorbed while the rebase probe was in
    /// flight) rejoins the Standard debounce lifecycle rather than being dropped — the closed
    /// Seed-residual event-loss. The reconfirm machinery is a fresh query over the live tree, not a
    /// refcount, so there is no per-origin balance to preserve and no origin gate.
    ///
    /// `last_event_time` reckons from `now` — the rebase-response instant, not the absorbed events'
    /// (those timestamps are discarded at absorb). The restarted burst's settle window therefore
    /// carries a bounded ≤ one-`settle` extra re-fire latency in exchange for never losing the
    /// residual. The restart lands in `Batching`, so no probe is in flight; the next
    /// `transition_to_verifying` constructs a [`PreFirePhase::Verifying`] with a freshly computed
    /// target, exactly as in a fresh `start_standard_burst`. The post-fire `forced` ceiling latch,
    /// `rebase_ceiling` timer lifecycle, `last_certified_hash` N=2 sample carrier, and
    /// `retry_streak` window counter are dropped by omission — all four are post-fire-only and tied
    /// to the now-discarded post-fire sample sequence; the restarted pre-fire burst opens its own
    /// fresh `burst_deadline`, fresh `last_certified_hash: None`, and fresh `retry_streak: 0`,
    /// exactly as a fresh `start_standard_burst`.
    ///
    /// `fold_latched` is **threaded, not dropped** — it is a fresh birth consult (a construction
    /// param like `burst_deadline` / `settle_timer` / `now`), because the restart *is* the next
    /// pre-fire burst's birth. This is what lets an operator arm a window during post-fire and have
    /// it apply to the residual restart that carries the final-window-race events: the caller
    /// passes the live window's birth consult for the restart instant.
    ///
    /// Sole production caller: `restart_burst_from_fire_tail_residual` in `burst.rs`.
    #[must_use]
    pub fn into_pre_fire_residual(
        self,
        burst_deadline: TimerId,
        settle_timer: TimerId,
        now: Instant,
        fold_latched: bool,
    ) -> PreFireBurst {
        debug_assert!(
            !self.final_window_residual.is_empty(),
            "into_pre_fire_residual: empty residual — the restart has no \
             seed; the caller must gate on a non-empty fire-tail residual",
        );
        let residual = self.final_window_residual;
        PreFireBurst::new(
            burst_deadline,
            PreFirePhase::Batching { settle_timer },
            BurstIntent::Standard,
            residual,
            Some(now),
            fold_latched,
        )
    }
}

impl ActiveBurst {
    /// Delegate to the lifecycle-side projection. [`Self::PreFire`] and [`Self::PostFire`] carry
    /// disjoint timer fields by construction; this dispatcher routes to whichever side the burst is
    /// currently on without re-enumerating the type-impossible cross-pairs.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match self {
            Self::PreFire(pre) => pre.timer_token(kind),
            Self::PostFire(post) => post.timer_token(kind),
        }
    }

    /// The burst's [`BurstIntent`]. `intent` is a field on **both** [`PreFireBurst`] and
    /// [`PostFireBurst`] (it survives the fire transition); this is the lifecycle-side projection
    /// that reads it without re-enumerating the cross-pairs — same wildcard-free PreFire/PostFire
    /// shape as [`Self::timer_token`]. Sole consumer: [`ProfileState::in_active_standard_burst`].
    #[must_use]
    pub const fn intent(&self) -> BurstIntent {
        match self {
            Self::PreFire(pre) => pre.intent,
            Self::PostFire(post) => post.intent,
        }
    }

    /// Delegate to the post-fire counter; [`Self::PreFire`] carries no fire, folding to
    /// [`AwaitVerdict::NotAwaiting`] — same shape-fold as [`Self::timer_token`], no wildcard.
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        match self {
            Self::PostFire(post) => post.note_effect_completion(),
            Self::PreFire(_) => AwaitVerdict::NotAwaiting,
        }
    }

    /// Delegate to whichever burst variant is live — both pre-fire and post-fire own a
    /// `last_certified_hash` carrier (a separate sample sequence each, across the fire boundary).
    /// Wildcard- free dispatch, same layered shape as [`Self::timer_token`] and
    /// [`Self::note_effect_completion`], distinguished only by the fact that **both** variants
    /// advance (no `NotAwaiting`-style fold on this delegate: the carrier exists on both sides).
    #[must_use]
    pub const fn advance_certified_sample(&mut self, hash: u128) -> Option<u128> {
        match self {
            Self::PreFire(pre) => pre.advance_certified_sample(hash),
            Self::PostFire(post) => post.advance_certified_sample(hash),
        }
    }

    /// Drive the fold latch onto the live pre-fire burst — the lifecycle layer of the retro-latch
    /// cascade. **Asymmetric**, the mirror image of [`Self::note_effect_completion`]: the latch
    /// lives on [`PreFireBurst`] only, so `PreFire ⇒ set` and `PostFire ⇒ no-op` (a post-fire burst
    /// has already crossed the fire boundary — there is no pre-fire consequence left to fold).
    /// Wildcard-free, so a future [`Self`] variant is a compile error, not a silent miss.
    pub const fn latch_fold(&mut self) {
        match self {
            Self::PreFire(pre) => pre.latch_fold(),
            Self::PostFire(_) => {}
        }
    }
}

/// Burst-finish directive — *what does the Profile do at burst-end?*
///
/// Carried as the second payload of [`ProfileState::Active`]. Default [`Self::ReturnToIdle`]: the
/// burst completes, the Profile returns to [`ProfileState::Idle`], and the next `FsEvent` may start
/// a fresh burst. [`Self::Reap`] flips the directive: the active burst still runs to completion (so
/// the burst-end Draining-sweep reconfirm runs before the Profile leaves the map), but
/// `finish_burst_to_idle` then routes through `reap_profile` instead of returning the Profile to
/// Idle.
///
/// **Why a payload, not a parallel field on Profile.** The directive is *only* meaningful inside an
/// Active burst. Encoding it as a `bool` alongside [`ProfileState`] (the prior
/// `Profile.reap_pending`) made `(Idle, true)` representable but structurally illegal — discipline
/// enforced by convention rather than by the type system. Folding the directive into the `Active`
/// variant's payload type-bans the illegal combination by construction.
///
/// **Writers.**
/// - [`ProfileState::mark_active_for_reap`] flips [`Self::ReturnToIdle`] → [`Self::Reap`]. Sole
///   caller: `detach_sub_inner` (last Sub detached mid-burst).
/// - [`ProfileState::clear_active_reap`] flips [`Self::Reap`] → [`Self::ReturnToIdle`]. Sole
///   caller: `attach_sub_inner`'s zombie-revival arm — a fresh Sub joining a zombie Profile
///   resurrects it under the new Sub set.
///
/// **Readers.** `emit_effects` (suppress emission), `on_effect_complete` (route last completion to
/// reap vs rebase), `handle_gate_deadline` (route zombie burst directly to finish), and
/// `finish_burst_to_idle` (post-drain reap dispatch).
///
/// The directive is preserved across the fire transition (`PreFireBurst::into_post_fire`'s caller
/// threads it through `ProfileMap::map_state`'s transform closure) and across phase transitions
/// within pre-fire (`transition_to_verifying`, `_draining`, etc.) — these helpers mutate the
/// burst's inner state without touching the `Active` variant's outer shape.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum BurstFinish {
    /// Default. Burst-end transitions the Profile to [`ProfileState::Idle`].
    #[default]
    ReturnToIdle,
    /// Burst-end reaps the Profile via `reap_profile`. Set by
    /// [`ProfileState::mark_active_for_reap`]; cleared by [`ProfileState::clear_active_reap`].
    Reap,
}

/// Where should a Profile land when its last Sub detaches?
///
/// Computed by [`ProfileState::detach_lifecycle`] at the moment the last Sub is removed. The two
/// arms encode the only paths that preserve the burst-end drain ordering:
///
/// - [`Self::ReapNow`]: the Profile is [`ProfileState::Idle`] or [`ProfileState::Pending`] — there
///   is no burst to drain. `reap_profile` runs immediately, releasing the descent prefix (Pending)
///   or the anchor contribution (Idle / materialized).
/// - [`Self::DeferToBurstEnd`]: the Profile is [`ProfileState::Active`] — a burst is in flight
///   whose burst-end Draining-sweep reconfirm must run before reap. The caller flips
///   [`BurstFinish::Reap`] (via [`ProfileState::mark_active_for_reap`]) so `finish_burst_to_idle`
///   routes through `reap_profile` once the burst converges.
///
/// Lives in core (not in the engine) because the classification is a projection over
/// [`ProfileState`] — the state knows what its last-Sub-detached outcome must be.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DetachLifecycle {
    /// Profile has no burst — reap synchronously.
    ReapNow,
    /// Profile has an Active burst — mark for reap, drain runs first.
    DeferToBurstEnd,
}

/// Trigger that drove a Profile's reap, threaded into [`crate::Diagnostic::ProfileReaped`]. Two
/// paths converge on the same `reap_profile` machinery:
///
/// - [`Self::Immediate`]: `detach_sub_inner` on an Idle/Pending Profile whose last Sub just
///   detached. No burst to wait on, so reap runs inline.
/// - [`Self::DeferredFromBurst`]: `finish_burst_to_idle` honouring the [`BurstFinish::Reap`]
///   directive at burst-end. The Profile spent time as a zombie burst before reaching reap.
///
/// Operators distinguish the two for incident triage: a flood of `DeferredFromBurst` reaps signals
/// churn on Active Profiles, whereas `Immediate` is the steady-state detach path.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ReapTrigger {
    /// Reap runs inline at refcount→0 — no burst to drain.
    Immediate,
    /// Reap runs at burst-end via [`BurstFinish::Reap`] honour.
    DeferredFromBurst,
}

/// Profile state machine.
///
/// Four lifecycle states, mutually exclusive by construction:
/// - `Idle`: healthy rest — anchor claim held (possibly pre-first-pin), no probe in flight, no
///   burst, no descent. Reads/writes baseline and current as-is.
/// - `Parked`: anchorless rest — the anchor was lost in a way the engine could not immediately heal
///   (a path-fatal probe failure, a kernel watch rejection, a descent-prefix purge, an observed loss
///   with no recovery parent). No claim, no snapshots, no probe slot. Exits *only* through a recovery
///   descent (a parent / anchor-slot event, a sensor overflow, an operator re-attach) — never
///   directly into a Seed; the descent terminus is the single re-bootstrap path. A unit variant: the
///   recovery channel is `Profile.watch_root_parent` and the survival witness rides
///   `AnchorClassification::Unclassified { witness }`, so a payload here would only duplicate facts
///   that must outlive the variant anyway. The variant's job is to force match arms, not carry data.
/// - `Pending(DescentState)`: anchor doesn't yet exist on disk; the engine is probing the deepest
///   existing prefix and advancing one path component per response. The anchor's `Profile.resource`
///   slot is `DescentScaffold`-roled and carries no `watch_demand` from this Profile (the prefix
///   carries the `+1`). See `DescentState` invariants.
/// - `Active(ActiveBurst, BurstFinish)`: anchor is materialized; a stability burst is in flight.
///   The second payload is the post-burst directive — see [`BurstFinish`] for the four-site reader
///   / two-site writer surface that drives it. Default ([`BurstFinish::ReturnToIdle`]) returns the
///   Profile to Idle at burst-end; [`BurstFinish::Reap`] dispatches `reap_profile` instead.
///   Carrying the reap directive on the `Active` payload (rather than a Profile-level boolean)
///   structurally bans the illegal "reap-pending while Idle" combination.
///
/// I5 (at most one outstanding probe per Profile) is a **representability** property: the in-flight
/// probe's liveness *and* identity live on the state itself, in the single [`ProbeSlot`] of
/// whichever carrier the Profile currently is — the `Pending` descent slot, the
/// `Active(PreFire(Verifying))` slot, or the `Active(PostFire(Rebasing))` slot. One Profile is
/// exactly one of these carriers, so it holds exactly one slot and two simultaneous probes are
/// unconstructable. The response handler routes by state, gates on [`Self::probe_correlation`], and
/// consumes by disarming that slot once via [`Self::take_probe`] — the structural consume-once
/// guarantee, with no separate side-table to drift against.
#[derive(Debug, Default)]
pub enum ProfileState {
    #[default]
    Idle,
    /// Anchorless rest — awaiting recovery or operator action. Entered only through the engine's
    /// `park_profile` coordinator (the loss wrappers and the descent-prefix purge); exits only
    /// through a recovery descent (`Parked → Pending`) or detach-reap. Holds no probe slot
    /// structurally (unit variant), so I5 is untouched.
    Parked,
    /// Pending-path descent in flight. The anchor (`Profile.resource`) is `DescentScaffold`-roled
    /// and carries no `watch_demand` from this Profile; `DescentState.current_prefix` does. When
    /// the anchor materializes (descent's last component arrives) the engine transitions Pending →
    /// Idle (releasing the prefix's contribution and bumping the anchor's), then immediately Idle →
    /// `Active(PreFire(Seed), …)` via `start_seed_burst`.
    Pending(DescentState),
    /// Stability burst in flight, with a post-burst directive. See [`BurstFinish`] for the
    /// directive's semantics; the default ([`BurstFinish::ReturnToIdle`]) is set at burst-launch
    /// and the engine flips to [`BurstFinish::Reap`] via [`Self::mark_active_for_reap`] when the
    /// Profile loses its last Sub mid-burst.
    Active(ActiveBurst, BurstFinish),
}

impl ProfileState {
    /// Variant-tag projection used by diagnostics that need to name "what state was the Profile
    /// actually in" without copying the payload. The five discriminants line up with the routing
    /// classes burst helpers care about: `Idle` (pre-burst), `Parked` (anchorless rest), `Pending`
    /// (descent in flight), `ActivePreFire` (settling / verifying / draining), `ActivePostFire`
    /// (awaiting / rebasing). The fire transition (`PreFire → PostFire`) is the only edge that
    /// crosses the pre-vs-post-fire discriminator, which is exactly the same boundary the
    /// [`ActiveBurst`] type split enforces.
    ///
    /// [`BurstFinish`] is intentionally collapsed at this projection — zombie and live bursts share
    /// routing class because every burst-helper that consults the discriminant routes identically
    /// for both. Readers that need to distinguish call [`Self::burst_finish`]; readers that need
    /// the *phase* (operator display vs routing) call [`Self::label`].
    #[must_use]
    pub const fn discriminant(&self) -> ProfileStateDiscriminant {
        match self {
            Self::Idle => ProfileStateDiscriminant::Idle,
            Self::Parked => ProfileStateDiscriminant::Parked,
            Self::Pending(_) => ProfileStateDiscriminant::Pending,
            Self::Active(ActiveBurst::PreFire(_), _) => ProfileStateDiscriminant::ActivePreFire,
            Self::Active(ActiveBurst::PostFire(_), _) => ProfileStateDiscriminant::ActivePostFire,
        }
    }

    /// Operator-display projection — one [`StateLabel`] per visible phase. Distinct from
    /// [`Self::discriminant`]: the discriminant names the *routing classes* the burst helpers
    /// branch on (collapsing `Batching | Verifying | Draining` to `ActivePreFire` and `Awaiting |
    /// Rebasing | Settling` to `ActivePostFire`), whereas this projection names the nine *phases*
    /// an operator reading `specter status` / `specter list` would expect to see — every leaf of
    /// the [`PreFirePhase`] / [`PostFirePhase`] split, plus `Idle`, `Parked`, and `Pending`.
    ///
    /// [`BurstFinish`] is collapsed (a zombie burst displays the same label as a live one — the
    /// directive is operationally irrelevant to the phase name).
    #[must_use]
    pub const fn label(&self) -> StateLabel {
        match self {
            Self::Idle => StateLabel::Idle,
            Self::Parked => StateLabel::Parked,
            Self::Pending(_) => StateLabel::Pending,
            Self::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
                PreFirePhase::Batching { .. } => StateLabel::Batching,
                PreFirePhase::Verifying { .. } => StateLabel::Verifying,
                PreFirePhase::Draining => StateLabel::Draining,
            },
            Self::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
                PostFirePhase::Awaiting { .. } => StateLabel::Awaiting,
                PostFirePhase::Rebasing(_) => StateLabel::Rebasing,
                PostFirePhase::Settling { .. } => StateLabel::Settling,
            },
        }
    }

    /// Read the burst-finish directive. `Some(_)` only when the Profile is in an Active burst;
    /// `None` for Idle, Parked, and Pending (where the directive is structurally meaningless).
    ///
    /// Read by `emit_effects` (suppress emission on zombie), `on_effect_complete` (route last
    /// completion), `handle_gate_deadline` (zombie-skip), and indirectly by every test that
    /// inspects the reap directive.
    #[must_use]
    pub const fn burst_finish(&self) -> Option<BurstFinish> {
        match self {
            Self::Active(_, finish) => Some(*finish),
            Self::Idle | Self::Parked | Self::Pending(_) => None,
        }
    }

    /// Classify the reap path when a Profile's last Sub detaches. Called by `detach_sub_inner` once
    /// no Subs remain on the Profile — the result chooses between immediate `reap_profile` and
    /// deferred-to-burst-end via [`Self::mark_active_for_reap`].
    ///
    /// Lives on [`ProfileState`] because the choice is a pure projection over the variant — the
    /// engine has no other input that influences the decision.
    #[must_use]
    pub const fn detach_lifecycle(&self) -> DetachLifecycle {
        match self {
            // A parked Profile reaps synchronously alongside Idle / Pending: detaching the last Sub
            // ends the park (and its nonsteady pin) — there is no burst to drain.
            Self::Idle | Self::Parked | Self::Pending(_) => DetachLifecycle::ReapNow,
            Self::Active(_, _) => DetachLifecycle::DeferToBurstEnd,
        }
    }

    /// Flip an Active burst's [`BurstFinish`] from [`BurstFinish::ReturnToIdle`] to
    /// [`BurstFinish::Reap`]. Returns `true` iff the state was [`Self::Active`] and the directive
    /// was set (already-`Reap` returns `true` and is a silent no-op — idempotent under re-entry).
    ///
    /// **Preconditions, by intent.** Callers have already established that the state is Active (via
    /// [`Self::detach_lifecycle`] or a `matches!` guard). The `bool` return surfaces "did the flip
    /// land" so callers can `debug_assert!` against a future routing breach.
    ///
    /// **Sole writer.** `detach_sub_inner` (refcount→0 on Active). No other site has a legitimate
    /// need to mark a burst for reap.
    #[must_use]
    pub const fn mark_active_for_reap(&mut self) -> bool {
        if let Self::Active(_, finish) = self {
            *finish = BurstFinish::Reap;
            true
        } else {
            false
        }
    }

    /// Flip an Active burst's [`BurstFinish`] back from [`BurstFinish::Reap`] to
    /// [`BurstFinish::ReturnToIdle`]. Returns `true` iff the state was [`Self::Active`] *and* the
    /// prior directive was `Reap` — i.e., a zombie burst was revived. `false` on `(Active,
    /// ReturnToIdle)` (normal join — nothing to clear), Idle, and Pending.
    ///
    /// **Why the precondition narrows to current-Reap.** The clear path's *only* legitimate trigger
    /// is zombie revival in `attach_sub_inner`. Returning `false` on a non-Reap Active keeps the
    /// bool return informative: the caller branches on it to emit the
    /// [`crate::Diagnostic::ReapPendingCancelled`] diagnostic and run the post-revival cleanup
    /// (`recompute_profile_settle`).
    ///
    /// **Sole writer.** `attach_sub_inner`'s zombie-revival arm.
    #[must_use]
    pub const fn clear_active_reap(&mut self) -> bool {
        if let Self::Active(_, finish @ BurstFinish::Reap) = self {
            *finish = BurstFinish::ReturnToIdle;
            true
        } else {
            false
        }
    }

    /// The live `TimerId` for the requested `kind` slot, or `None` if the state owns no timer of
    /// that kind right now.
    ///
    /// Only [`Self::Active`] Profiles schedule timers — [`Self::Idle`], [`Self::Parked`], and
    /// [`Self::Pending`] (descent) own none and return `None` for every kind. The `Active` arm
    /// delegates to [`ActiveBurst::timer_token`], which in turn routes to whichever burst-side type
    /// ([`PreFireBurst`] or [`PostFireBurst`]) actually carries the field. Each layer only
    /// enumerates the kinds its data shape can carry, so type-impossible pairs fold to `None` at
    /// the leaf without an explicit wildcard arm.
    ///
    /// Consumed by the engine's `pop_expired` and `on_timer_expired` gates to distinguish a live
    /// timer from a stale heap orphan (cancelled because the Profile's burst was reset between
    /// `schedule` and pop).
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match self {
            Self::Active(burst, _) => burst.timer_token(kind),
            Self::Idle | Self::Parked | Self::Pending(_) => None,
        }
    }

    /// Delegate to the active burst's post-fire counter; `Idle` / `Parked` / `Pending` own none and
    /// fold to [`AwaitVerdict::NotAwaiting`]. Same layered, wildcard-free delegation as
    /// [`Self::timer_token`].
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        match self {
            Self::Active(burst, _) => burst.note_effect_completion(),
            Self::Idle | Self::Parked | Self::Pending(_) => AwaitVerdict::NotAwaiting,
        }
    }

    /// Delegate to the active burst's `last_certified_hash` carrier; `Idle` / `Parked` / `Pending`
    /// own none and fold to `None` (the "no carrier exists to advance" shape). Same layered,
    /// wildcard-free delegation as [`Self::note_effect_completion`].
    #[must_use]
    pub const fn advance_certified_sample(&mut self, hash: u128) -> Option<u128> {
        match self {
            Self::Active(burst, _) => burst.advance_certified_sample(hash),
            Self::Idle | Self::Parked | Self::Pending(_) => None,
        }
    }

    /// Drive the fold latch onto an in-flight pre-fire burst — the state layer of the retro-latch
    /// cascade, the entry [`Profile::arm_absorb`] calls. `Active ⇒` delegate to
    /// [`ActiveBurst::latch_fold`] (itself a PreFire-set / PostFire- no-op); `Idle | Parked |
    /// Pending ⇒` no-op — there is no burst whose terminal consequence the window could override,
    /// so an arm in those states only sets the window for the *next* burst's birth consult.
    /// Wildcard-free, same layered shape as [`Self::advance_certified_sample`].
    pub const fn latch_fold(&mut self) {
        match self {
            Self::Active(burst, _) => burst.latch_fold(),
            Self::Idle | Self::Parked | Self::Pending(_) => {}
        }
    }

    /// True iff the state is `Active(PreFire(Draining))`. The reconfirm cascade (the `Draining →
    /// Verifying` re-probe) keys off this predicate: at every `finish_burst_to_idle` the engine
    /// sweeps the Draining Profiles and reconfirms each whose covered-descendant query has gone
    /// false. `Idle`, `Parked`, and `Pending` are structurally not-Draining; the post-fire arm and
    /// the other pre-fire phases (Batching, Verifying) also return `false`.
    #[must_use]
    pub const fn is_draining(&self) -> bool {
        match self {
            Self::Active(ActiveBurst::PreFire(pre), _) => {
                matches!(pre.phase, PreFirePhase::Draining)
            }
            Self::Idle
            | Self::Parked
            | Self::Pending(_)
            | Self::Active(ActiveBurst::PostFire(_), _) => false,
        }
    }

    /// True iff the state is an Active **Standard** burst, in *any* phase — pre-fire (`Batching |
    /// Verifying | Draining`) or post-fire (`Awaiting | Rebasing | Settling`). Wildcard-free,
    /// mirroring [`Self::is_draining`].
    ///
    /// This is the per-Profile half of the Standard-descendant coverage query. A Standard descendant
    /// covers its ancestor for the burst's *entire* lifetime — pre-fire through post-fire, across a
    /// fire-tail residual restart — so spanning both pre- and post-fire here evaluates that lifetime
    /// fresh: the descendant counts as covering its ancestor from burst start until
    /// `finish_burst_to_idle`, whatever phase it is in. A restarted residual burst is `intent:
    /// Standard` by construction ([`PostFireBurst::into_pre_fire_residual`]), so it stays counted
    /// with no special accounting. Seed bursts return `false` — they never contribute coverage.
    ///
    /// Read through [`crate::ProfileState::in_active_standard_burst`] → `.state()` exactly as
    /// [`Self::is_draining`] is (no `Profile` delegate — the accessor convention is
    /// `.state().<pred>()`).
    #[must_use]
    pub const fn in_active_standard_burst(&self) -> bool {
        match self {
            Self::Active(burst, _) => matches!(burst.intent(), BurstIntent::Standard),
            Self::Idle | Self::Parked | Self::Pending(_) => false,
        }
    }

    /// True iff the live pre-fire burst carries the fold latch — the read side of the cascade, the
    /// engine's verdict-time override consult (`PreFireBurst::fold_latched`). Read via `.state()`,
    /// exactly as [`Self::is_draining`] / [`Self::in_active_standard_burst`] (no `Profile` delegate
    /// — the accessor convention is `.state().<pred>()`).
    ///
    /// `Active(PreFire) ⇒` the burst's latch; every other state ⇒ `false`. The latch lives on
    /// [`PreFireBurst`] only, so post-fire folds to `false` structurally — and the sole consult
    /// site (`classify_consequence`) only ever resolves this on `Active(PreFire(Verifying))`, a
    /// probe response having just arrived. Wildcard-free: a future variant is a compile error, not
    /// a silent `false`.
    #[must_use]
    pub const fn burst_fold_latched(&self) -> bool {
        match self {
            Self::Active(ActiveBurst::PreFire(pre), _) => pre.fold_latched.is_latched(),
            Self::Idle
            | Self::Parked
            | Self::Pending(_)
            | Self::Active(ActiveBurst::PostFire(_), _) => false,
        }
    }

    /// Borrow the descent payload if the state is currently [`Self::Pending`]. `None` for
    /// [`Self::Idle`], [`Self::Parked`], and [`Self::Active`] — the descent payload only lives in
    /// the `Pending` variant.
    #[must_use]
    pub const fn descent_state(&self) -> Option<&DescentState> {
        match self {
            Self::Pending(d) => Some(d),
            Self::Idle | Self::Parked | Self::Active(_, _) => None,
        }
    }

    /// Mutable counterpart to [`Self::descent_state`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        match self {
            Self::Pending(d) => Some(d),
            Self::Idle | Self::Parked | Self::Active(_, _) => None,
        }
    }

    /// The correlation of this state's in-flight probe, or `None` if the carrier holds none. A
    /// total projection over the state space: the three probe-bearing carriers — a `Pending`
    /// descent, an `Active(PreFire(Verifying))`, an `Active(PostFire(Rebasing))` — answer from
    /// their armed slot; every other state (including a disarmed slot) yields `None`.
    #[must_use]
    pub const fn probe_correlation(&self) -> Option<ProbeCorrelation> {
        match self {
            Self::Active(ActiveBurst::PreFire(burst), _) => match &burst.phase {
                PreFirePhase::Verifying { slot, .. } => slot.correlation(),
                PreFirePhase::Batching { .. } | PreFirePhase::Draining => None,
            },
            Self::Active(ActiveBurst::PostFire(burst), _) => match &burst.phase {
                PostFirePhase::Rebasing(slot) => slot.correlation(),
                PostFirePhase::Awaiting { .. } | PostFirePhase::Settling { .. } => None,
            },
            Self::Pending(d) => d.probe_correlation(),
            Self::Idle | Self::Parked => None,
        }
    }

    /// Disarm whichever probe-bearing carrier this state holds and return the prior correlation —
    /// the single state-level consume. Total: the three probe-bearing carriers (`Pending` descent,
    /// `Active(PreFire(Verifying))`, `Active(PostFire(Rebasing))`) disarm their slot; every other
    /// state (including an already-disarmed slot) is a `None` no-op. The disarm leaves the
    /// carrier's variant intact — only the slot empties — so a route computed before this call
    /// stays valid after it.
    #[must_use]
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        match self {
            Self::Active(ActiveBurst::PreFire(burst), _) => match &mut burst.phase {
                PreFirePhase::Verifying { slot, .. } => slot.disarm(),
                PreFirePhase::Batching { .. } | PreFirePhase::Draining => None,
            },
            Self::Active(ActiveBurst::PostFire(burst), _) => match &mut burst.phase {
                PostFirePhase::Rebasing(slot) => slot.disarm(),
                PostFirePhase::Awaiting { .. } | PostFirePhase::Settling { .. } => None,
            },
            Self::Pending(d) => d.disarm_probe(),
            Self::Idle | Self::Parked => None,
        }
    }
}

/// Variant tag for [`ProfileState`], carried on diagnostics that report state-machine routing
/// breaches without copying the payload.
///
/// The five variants match the routing classes the engine's burst helpers branch on. They are
/// coarser than the full state enum (`Active(PreFire(Batching{settle_timer}))` collapses to
/// `ActivePreFire`) — for diagnostic triage; see [`StateLabel`] for operator display. Stable
/// against future phase additions.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ProfileStateDiscriminant {
    /// [`ProfileState::Idle`].
    Idle,
    /// [`ProfileState::Parked`].
    Parked,
    /// [`ProfileState::Pending`].
    Pending,
    /// [`ProfileState::Active`] with [`ActiveBurst::PreFire`].
    ActivePreFire,
    /// [`ProfileState::Active`] with [`ActiveBurst::PostFire`].
    ActivePostFire,
}

/// Operator-display label for a [`ProfileState`] — the nine visible phases an operator reading
/// `specter status` / `specter list` would expect to see.
///
/// Distinct from [`ProfileStateDiscriminant`]: the discriminant names the *routing classes* the
/// engine's burst helpers branch on, whereas this enum names the *phases* the state can occupy. Two
/// enums, two consumers — diagnostics keep their stable `ActivePreFire` / `ActivePostFire` tag,
/// operator surfaces print the phase (`Batching` / `Verifying` / `Draining` / `Awaiting` /
/// `Rebasing` / `Settling`).
///
/// Constructed via [`ProfileState::label`]; the projection is exhaustive over the type space, so a
/// future phase addition is a compile error rather than a silently-collapsing display.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StateLabel {
    /// [`ProfileState::Idle`] — no burst in flight.
    Idle,
    /// [`ProfileState::Parked`] — anchorless; awaiting recovery (a parent / anchor-slot event, an
    /// overflow, a re-attach) or operator action. The honest operator row for the state that used
    /// to display as a healthy `Idle`.
    Parked,
    /// [`ProfileState::Pending`] — anchor path descent in flight.
    Pending,
    /// [`PreFirePhase::Batching`] — activity-gap settle wait.
    Batching,
    /// [`PreFirePhase::Verifying`] — pre-fire stability probe in flight.
    Verifying,
    /// [`PreFirePhase::Draining`] — self-stable, descendants still active.
    Draining,
    /// [`PostFirePhase::Awaiting`] — Effects emitted, gate counter open.
    Awaiting,
    /// [`PostFirePhase::Rebasing`] — post-fire baseline-capture probe.
    Rebasing,
    /// [`PostFirePhase::Settling`] — re-sample spacing wait (the `Rebasing ⇄ Settling` retry loop).
    Settling,
}

/// State for a Profile undergoing pending-path descent.
///
/// Lives inline on `ProfileState::Pending` for the duration of descent.
///
/// Invariants:
/// - `current_prefix` carries a `+1` `watch_demand` contribution from this Profile (added at
///   descent registration / advancement; dropped at descent end or rewind).
/// - [`DescentRemaining`] is non-empty by type construction — the anchor itself is the last
///   component, and descent transitions Pending → Idle on materialization rather than emptying the
///   path.
///
/// I5 ("at most one outstanding probe per descent") is a representability property: the descent
/// probe's liveness *and* identity live in `probe`, a single [`ProbeSlot`] on this payload. An
/// armed slot is a probe in flight; an empty slot is descent awaiting the next structural event
/// with nothing out. One descent holds exactly one slot, so two simultaneous descent probes are
/// unconstructable.
#[derive(Debug)]
pub struct DescentState {
    /// Deepest existing ancestor currently Watched. The Profile contributes `+1` to this Resource's
    /// `watch_demand`. Module-private: read via [`Self::current_prefix`], moved via
    /// [`Self::advance_to`].
    current_prefix: ResourceId,
    /// Path components from `current_prefix` (exclusive) down to the anchor (inclusive). Non-empty
    /// by type construction; single-component segments (no `/`). Module-private: reached via
    /// [`Self::remaining_components`] / [`Self::remaining_components_mut`].
    remaining_components: DescentRemaining,
    /// The descent probe slot — a linear [`ProbeSlot`]. Armed while a probe is in flight at
    /// `current_prefix` (carrying the correlation the response must echo); empty while descent
    /// awaits the next structural event. Module-private: the linear protocol is the only access
    /// path — [`Self::arm_probe`] (mint), [`Self::probe_correlation`] (read),
    /// [`Self::disarm_probe`] (consume). It cannot be cloned, so it is consumed where it lives.
    probe: ProbeSlot,
    /// Witnessed-appearance latch: the anchor's absence at this path was observed first-hand and the
    /// path later completed — an absence→presence pair, both halves probe observations (or, at
    /// construction, the anchor-loss signal that entered the descent: the loss *is* the absence half,
    /// so the latch is born set). Never reset — the latch persists through rewinds and forward
    /// advances, which mutate the sibling fields in place. The terminus consumes it: a witnessed
    /// descent's anchor materialization opens a *triggered* Seed (the witness lands in `dirty`, so
    /// `seed_owes_first_fire` sees the activity), an unwitnessed one stays cold and pins silently.
    ///
    /// Kernel events never write this latch. A directory event at a descent prefix carries no
    /// segment name on either backend, so event adjacency cannot distinguish the awaited segment
    /// from sibling churn entirely outside the Sub's scope — latching on it would false-first-fire
    /// a never-fired Sub whose anchor sat unchanged on disk the whole time. Events drive re-probes
    /// (mechanism); only the probes' own observations move the witness (verdict).
    ///
    /// An explicit field, not inferred: `watch_root_parent.is_some()` happens to distinguish
    /// re-entered from attach-time descents today, but that is incidental — the latch states the
    /// semantics.
    witnessed: bool,
    /// Standing absence observation — the pending half of the absence→presence appearance witness.
    /// Set by [`Self::note_observed_absent`] when a probe observes the anchor's path incomplete
    /// (the awaited segment observed absent under the prefix, or the prefix itself vanished);
    /// consumed by [`Self::note_observed_present`] when a later probe finds the awaited segment —
    /// the pair latches `witnessed`. A descent that only ever finds its segments present (an attach
    /// over an existing tree) never sets this bit, so its terminus stays cold no matter how much
    /// sibling churn reached the prefix watches.
    observed_absent: bool,
    /// Epoch debt: a signal (a prefix `StructureChanged`, or a sensor overflow) arrived while this
    /// descent's probe was in flight. "In flight" at the engine means only that the response is
    /// unprocessed — the walk behind it may have completed *before* the signal, so it cannot
    /// witness the signal's change. The correlation gate fences out *superseded* probes, but a
    /// probe whose walk merely predates a dropped signal was never superseded: its response would
    /// pass the gate, park the descent, and the signal (often the awaited segment's own creation)
    /// would never re-arrive — a permanent wedge.
    ///
    /// The latch closes that hole. A signal arriving mid-flight sets it
    /// ([`Self::note_reprobe_owed`]) instead of being dropped; the descent response dispatch repays
    /// the debt ([`Self::take_reprobe_owed`]) by emitting a probe that postdates every latched
    /// signal by construction. A `bool` suffices: the descent needs to know only *whether* a signal
    /// raced the probe, not how many, so any number of mid-flight signals collapse to one repay
    /// (idempotent set, single consume). The repay preserves the in-flight observation — the park
    /// arm still records the absence half of the appearance witness, so the postdating probe can
    /// complete it with presence — which is why descent latches rather than superseding the way the
    /// pre-fire burst's `event_drives_batching` does (a stale stability sample is worthless; a
    /// point-in-time absence is half a witness).
    ///
    /// Dies with the descent: terminal materialization, contract-violation abandon, and every
    /// cancel / teardown drop [`DescentState`] wholesale, so a latched-but-never-repaid debt cannot
    /// leak past the descent it belongs to (a materialized descent's Seed `WholeSubtree` probe
    /// postdates everything; an abandoned descent must not auto-re-probe a buggy walker).
    reprobe_owed: bool,
    /// Signal-bearing retry budget. A *signal-bearing* descent probe is one emitted by
    /// `engine::descent::try_emit_descent_probe` — the single choke for the two cases where a probe
    /// is the only observer of a structural signal that will not re-arrive on its own: a prefix
    /// `StructureChanged` / overflow re-trigger, and the re-probe-owed repay of a signal that raced
    /// an earlier probe. (The descent-entry, forward-advance, and rewind probes go through
    /// `emit_owner_probe` directly and are *not* signal-bearing — they re-trigger through their own
    /// channels, so a transient blip there merely parks the descent for the next event.)
    ///
    /// `None` while the outstanding / last descent probe is not signal-bearing, or once a
    /// signal-bearing probe reached a terminal fate. `Some(n)` while a signal-bearing probe is
    /// outstanding (or has just failed), having already spent `n` of its transient-retry budget.
    ///
    /// **Why a budget.** If a signal-bearing probe fails transiently (an EMFILE-class
    /// [`ProbeFailure::Transient`](crate::ProbeFailure::Transient) the walk couldn't observe past),
    /// dropping it wedges the descent: the awaited segment's own creation was the signal, and the
    /// kernel never re-announces an entry that already exists. The engine re-latches `reprobe_owed`
    /// so the response dispatch emits a fresh postdating probe — bounded by a small budget so a
    /// *persistently*-failing prefix cannot spin a tight re-probe loop. The count rides here on
    /// [`DescentState`] because the repay that re-arms the next probe runs before that probe's fate
    /// is known; the failure being dispatched is always the *previous* probe's, so the
    /// signal-bearing fact and its spent budget must outlive a single dispatch.
    ///
    /// Cleared on **either** terminal fate — a successful observation (the signal's effect is seen)
    /// or the budget being spent (the chain gives up). Both reset to `None`, so a later prefix
    /// event re-marks a fresh `Some(0)` and earns a fresh budget; there is no stale count to starve
    /// a genuinely new signal. Dies with the descent like the other carriers.
    signal_retries: Option<u8>,
}

impl DescentState {
    /// Construct a fresh descent payload. Sole producer pattern used by
    /// `materialize_path_or_pending` (Profile pending arm) and the recovery / rewind flows in
    /// `engine::descent` that re-enter `Pending` after an anchor-terminal event.
    ///
    /// Field-private; callers route through this constructor so the invariants on `current_prefix`
    /// (Watched, refcounted), `remaining_components` (non-empty by [`DescentRemaining`]'s own
    /// constructor), and `probe` (the descent's single in-flight slot) are pinned at a single
    /// boundary. Every fresh descent entry mints a correlation and emits a probe, so an honest
    /// constructor takes the `probe` slot — typically [`ProbeSlot::armed`] with the just-minted
    /// correlation. The engine's refcount setup runs around this constructor (the contribution at
    /// `current_prefix` is installed by `add_watch` separately).
    ///
    /// `witnessed` is the appearance latch's birth value: `true` only for a descent entered from an
    /// observed anchor loss (the loss signal *is* the absence half of the witness, and the
    /// materialization that ends the descent supplies the presence half); attach-time entries and
    /// the event-scan recovery construct `false` — see the field doc and the engine call sites for
    /// the per-entry rationale.
    #[must_use]
    pub const fn new(
        current_prefix: ResourceId,
        remaining_components: DescentRemaining,
        probe: ProbeSlot,
        witnessed: bool,
    ) -> Self {
        Self {
            current_prefix,
            remaining_components,
            probe,
            witnessed,
            observed_absent: false,
            reprobe_owed: false,
            signal_retries: None,
        }
    }

    /// The deepest currently-Watched ancestor on the descent path. Carries this Profile's `+1
    /// STRUCTURE` [`crate::ContribKey::ProfileDescent`] contribution.
    #[must_use]
    pub const fn current_prefix(&self) -> ResourceId {
        self.current_prefix
    }

    /// Read-only handle to the remaining-path-component chain.
    #[must_use]
    pub const fn remaining_components(&self) -> &DescentRemaining {
        &self.remaining_components
    }

    /// Mutable handle to the remaining-path-component chain. Sole callers are the engine's descent
    /// dispatcher (`engine::descent::advance_descent` consumes the head via
    /// [`DescentRemaining::advance`]) and the rewind path (`dispatch_descent_vanished` re-injects
    /// the prefix's segment via [`DescentRemaining::prepend`]).
    pub const fn remaining_components_mut(&mut self) -> &mut DescentRemaining {
        &mut self.remaining_components
    }

    /// Whether the descent has witnessed the anchor's appearance first-hand (an absence→presence
    /// observation pair, or a loss-born entry) — the latch the terminus Seed's cold/triggered split
    /// reads. See the field doc for the full semantics.
    #[must_use]
    pub const fn witnessed(&self) -> bool {
        self.witnessed
    }

    /// Record a first-hand observation that the anchor's path is incomplete — a probe observed the
    /// awaited segment absent under the prefix, or the prefix itself vanished. An absent
    /// intermediate segment implies the anchor absent (a path cannot complete through a missing
    /// directory), so one bit serves every descent depth. The standing observation is consumed by
    /// [`Self::note_observed_present`]; the pair is the only writer of the `witnessed` latch after
    /// construction.
    pub const fn note_observed_absent(&mut self) {
        self.observed_absent = true;
    }

    /// Record that a probe found the awaited segment present. Under a standing absence observation
    /// this completes the absence→presence pair and latches `witnessed` — the descent saw the
    /// anchor's path go from broken to complete, which is an appearance no matter which event (or
    /// overflow reseed) triggered the probe. Without a standing observation it is a no-op: a first
    /// observation finding the segment present is indistinguishable from the segment having existed
    /// all along, and claiming appearance there would fire on attach-over-existing (restart-safe
    /// doctrine). One-way once latched: `witnessed` survives every in-place descent mutation
    /// (advance, rewind, re-arm) until the terminus consumes it at anchor materialization.
    pub const fn note_observed_present(&mut self) {
        if self.observed_absent {
            self.witnessed = true;
            self.observed_absent = false;
        }
    }

    /// Record that a signal raced this descent's in-flight probe — set the re-probe-owed latch. The
    /// engine calls this from `on_descent_event` when a prefix event or sensor overflow arrives
    /// while a probe is already outstanding: the in-flight walk may predate the signal, so its
    /// response cannot witness the signal's change. Idempotent — any number of mid-flight signals
    /// collapse to one owed re-probe (the consume is `bool`-valued). See the field doc for why the
    /// debt is latched rather than the signal dropped.
    pub const fn note_reprobe_owed(&mut self) {
        self.reprobe_owed = true;
    }

    /// Consume the re-probe-owed latch, returning whether a debt was outstanding (and clearing it).
    /// The engine calls this once per descent response dispatch: a `true` return means a signal
    /// raced the just-consumed probe and is owed a fresh, postdating probe. The clear is
    /// unconditional so a descent that re-armed inline (advance / rewind already postdate the
    /// signal) drops the debt without a redundant re-probe.
    #[must_use]
    pub const fn take_reprobe_owed(&mut self) -> bool {
        let owed = self.reprobe_owed;
        self.reprobe_owed = false;
        owed
    }

    /// Flag the just-emitted descent probe as signal-bearing — its transient failure must re-latch
    /// a fresh postdating probe rather than park the descent. The engine calls this from the single
    /// signal-bearing emission choke (`try_emit_descent_probe`). Idempotent on the count: a repay
    /// re-emission while the budget is mid-spend (`Some(n)`) preserves `n` — only a fresh emission
    /// from `None` starts a new budget at `Some(0)`. See the [`Self::signal_retries`] field doc.
    pub const fn mark_signal_probe(&mut self) {
        if self.signal_retries.is_none() {
            self.signal_retries = Some(0);
        }
    }

    /// Transient-retry budget already spent by the outstanding / last signal-bearing probe, or
    /// `None` if that probe is not signal-bearing. The engine compares it against its retry ceiling
    /// to decide between a bounded re-latch and giving up.
    #[must_use]
    pub const fn signal_retries(&self) -> Option<u8> {
        self.signal_retries
    }

    /// Spend one unit of the signal-bearing retry budget — the engine calls this when it re-latches
    /// `reprobe_owed` for a transiently-failed signal-bearing probe. No-op when not signal-bearing
    /// (the engine only reaches here under a `Some` budget; the guard is pure defense). Saturating
    /// — the engine's ceiling stops the chain long before `u8` overflow.
    pub const fn bump_signal_retry(&mut self) {
        if let Some(n) = self.signal_retries {
            self.signal_retries = Some(n.saturating_add(1));
        }
    }

    /// Clear the signal-bearing marker and its spent budget — the outstanding signal-bearing probe
    /// reached a terminal fate (it observed, so the signal's effect is seen; or its retries were
    /// spent and the chain gave up). The next signal-bearing emission earns a fresh `Some(0)`.
    pub const fn clear_signal_probe(&mut self) {
        self.signal_retries = None;
    }

    /// Rewrite the descent's current prefix. Used by the engine's descent dispatcher on forward
    /// advance (the prior prefix's `Ok` response routes here with the newly-Watched child) and by
    /// the rewind path (`Vanished` on the prefix routes here with the parent that just took over
    /// the descent's watch).
    ///
    /// Pairs with the engine's `add_watch` / `sub_watch` calls that maintain the `+1 STRUCTURE`
    /// contribution at the new and old prefixes respectively; the typed mutator pins that the field
    /// only moves under refcount-aware control.
    pub const fn advance_to(&mut self, new_prefix: ResourceId) {
        self.current_prefix = new_prefix;
    }

    /// Arm the descent's single probe slot with a freshly-minted correlation — the **mint** edge of
    /// the descent's linear-probe protocol. The engine calls this when re-probing in place (forward
    /// advance, rewind, event re-trigger); fresh-descent entry instead constructs the slot armed
    /// via [`Self::new`]. Asserts the slot was empty (the response handler or cancel path disarmed
    /// it first) — a double-arm would orphan the prior correlation.
    pub fn arm_probe(&mut self, correlation: ProbeCorrelation) {
        self.probe.arm(correlation);
    }

    /// Identity of the descent's in-flight probe, or `None` if idle — the **read** edge of the
    /// linear-probe protocol. [`crate::ProfileState::probe_correlation`] delegates here for its
    /// descent carrier rather than reaching the private slot.
    #[must_use]
    pub(crate) const fn probe_correlation(&self) -> Option<ProbeCorrelation> {
        self.probe.correlation()
    }

    /// Consume the descent's probe: disarm the slot and return the prior correlation (`None` if
    /// already idle) — the **consume** edge of the linear-probe protocol, dual of
    /// [`Self::arm_probe`].
    ///
    /// Crate-internal by design. The engine-facing "single consume per owner" law remains the `pub`
    /// [`crate::ProfileState::take_probe`], which delegates its descent arm here, so the consume
    /// routes through the typed protocol instead of a raw field and `probe` stays module-private.
    /// Routing-once is unaffected — the engine still sees exactly one consume entry point per owner.
    #[must_use]
    pub(crate) const fn disarm_probe(&mut self) -> Option<ProbeCorrelation> {
        self.probe.disarm()
    }
}

/// Path-component chain from a descent's `current_prefix` down to the anchor.
///
/// Non-emptiness is a type-level invariant: the sole constructor [`DescentRemaining::from_vec`]
/// rejects empty inputs, and the two mutators ([`advance`](Self::advance) and
/// [`prepend`](Self::prepend)) preserve non-emptiness by construction. `CompactString` keeps
/// typical-length names (≤24 bytes) inline, so the per-element advance / rewind avoids the heap for
/// the common path.
///
/// **API discipline.**
/// - [`head`](Self::head) is the next segment under consideration — always present by invariant.
/// - [`is_terminal`](Self::is_terminal) is `true` when only the head remains; the descent dispatcher
///   routes through anchor materialization on this edge and never calls [`advance`](Self::advance).
/// - [`advance`](Self::advance) consumes the head and is debug-asserted non-terminal at call time.
///   The terminal arm has already routed through anchor materialization in production, which ends
///   the `Pending` lifecycle; advance is structurally never reachable there.
/// - [`prepend`](Self::prepend) is the rewind path's mutator: a `Vanished` response on
///   `current_prefix` re-injects the prefix's own segment as the new head while the prefix shifts
///   up one level.
///
/// **Representation.** Components are stored *reverse* of descent order: the logical head (next to
/// consume) is the `Vec`'s last element. The only mutated end is therefore the `Vec`'s O(1) tail —
/// [`advance`](Self::advance) is a `pop`, [`prepend`](Self::prepend) a `push` — instead of the O(N)
/// front shifts a forward-order `Vec` forces (`advance` runs every forward descent step). The
/// reversal is an internal detail: every accessor keeps its logical-order signature and semantics;
/// [`iter`](Self::iter) and the [`Debug`] impl present descent order so diagnostics and tests are
/// unaffected.
#[derive(Eq, PartialEq)]
pub struct DescentRemaining {
    /// Reversed: `inner.last()` is the logical head. Never empty.
    inner: Vec<CompactString>,
}

impl DescentRemaining {
    /// Construct from a `Vec` in descent order. Returns `None` iff `v` is empty, preserving the
    /// non-empty invariant. Sole intended producer is `materialize_path_or_pending`'s Pending
    /// branch, where the `prefix_idx + 1 < components.len()` gate already guarantees non-empty; the
    /// `None` arm is defense-in-depth against future callers. The one-time reverse into storage
    /// order is O(depth) on the cold descent-registration path.
    #[must_use]
    pub fn from_vec(v: Vec<CompactString>) -> Option<Self> {
        if v.is_empty() {
            None
        } else {
            let mut inner = v;
            inner.reverse();
            Some(Self { inner })
        }
    }

    /// First (next-to-consume) segment. Always present by invariant.
    #[must_use]
    pub fn head(&self) -> &CompactString {
        // Index the tail (the logical head under the reversed representation) rather than
        // `last().unwrap()` to encode the invariant at the access site — a future maintainer can't
        // weaken `head` to a defensive `Option` without also adjusting the type's construction
        // discipline.
        &self.inner[self.inner.len() - 1]
    }

    /// Number of remaining segments. Always `>= 1` by invariant.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Always `false` — non-emptiness is a type-level invariant upheld by [`Self::from_vec`]
    /// (rejects empty inputs) and the mutators ([`Self::advance`] / [`Self::prepend`]). Implemented
    /// so the `len() / is_empty()` pair is complete by Rust convention; production callers should
    /// prefer [`Self::is_terminal`].
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// True iff only the head remains (`len() == 1`). The descent dispatcher's terminal arm
    /// consumes the head via anchor materialization on this edge and never calls [`Self::advance`].
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.inner.len() == 1
    }

    /// Consume the logical head (pop the reversed `Vec`'s tail). Preserves the non-empty invariant
    /// by debug-asserting non-terminal at entry; release builds clamp on terminal (no-op) rather
    /// than violating the invariant.
    ///
    /// Production callers (`advance_descent` in `specter-engine::descent`) guard the call with
    /// [`is_terminal`](Self::is_terminal) — `dispatch_descent_ok` routes the terminal edge through
    /// anchor materialization, which replaces the `Pending` lifecycle before this method would ever
    /// be reachable on a single-element remaining.
    pub fn advance(&mut self) {
        debug_assert!(
            self.inner.len() >= 2,
            "DescentRemaining::advance called at terminal — caller must \
             check is_terminal() and route to materialization instead",
        );
        if self.inner.len() >= 2 {
            self.inner.pop();
        }
    }

    /// Rewind by one segment: re-inject `segment` as the new logical head (push onto the reversed
    /// `Vec`'s tail). Used by `dispatch_descent_vanished`'s rewind branch, where a `Vanished`
    /// response on `current_prefix` shifts the descent up one level and the vanished prefix's own
    /// segment becomes the next-to-consume component on the way back down.
    pub fn prepend(&mut self, segment: CompactString) {
        self.inner.push(segment);
    }

    /// Iterate the components in descent (logical) order. For test assertions and diagnostics only
    /// — production code uses [`head`](Self::head) / [`len`](Self::len) /
    /// [`is_terminal`](Self::is_terminal).
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &CompactString> {
        self.inner.iter().rev()
    }
}

impl std::fmt::Debug for DescentRemaining {
    /// Descent (logical) order, hiding the reversed internal storage so diagnostics read the way
    /// the path is consumed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// `Standard` — event-driven burst; preserves baseline; fires Effect on stable. `Seed` — fresh
/// Profile or post-Effect rebase; sets baseline; no Effect.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BurstIntent {
    #[default]
    Standard,
    Seed,
}

/// Discriminator for a scheduled timer's role within a Burst's lifecycle.
///
/// `Settle` — debounce timer armed during [`PreFirePhase::Batching`]. Expiry drives Batching →
/// Verifying. `BurstDeadline` — Burst-level max-settle timer armed at Burst start. Expiry sets
/// `PreFireBurst.forced = true` and dispatches by current phase. The timer is carried on
/// [`PreFireBurst`] and is structurally invalid in post-fire phases; once the burst crosses
/// [`PreFireBurst::into_post_fire`] the timer is dropped from the type's field set, and a stale
/// fire is filtered out by the [`PostFireBurst::timer_token`] projection (the engine's stale-drain
/// consumes the projection through [`ProfileState::timer_token`]). `AwaitGateDeadline` — recovery
/// timer armed at [`PostFirePhase::Awaiting`] entry. Expiry indicates the actuator is taking longer
/// than expected (likely a hung child); the engine force-transitions to `Rebasing` to re-establish
/// a baseline against disk reality. `PostFireSettle` — the post-fire mirror of `Settle`: the
/// re-sample spacing wait armed during [`PostFirePhase::Settling`]. On expiry,
/// `Engine::handle_post_fire_settle_expired` decides whether to reschedule (events arrived since
/// the timer was scheduled) or drive `Settling → Rebasing` for the next sample — the post-fire
/// analogue of pre-fire's `on_settle_expired`. Carried on [`PostFireBurst`]; structurally `None` on
/// pre-fire (the post- fire analogue of how `Settle` is `None` on `Verifying`/`Draining`).
/// `RebaseCeiling` — the post-fire mirror of `BurstDeadline`: the rebase loop's max bound, armed
/// once at the natural `Awaiting → Settling` entry and tracked on the `rebase_ceiling` lifecycle.
/// Expiry latches the loop's terminal, applied with the verdict in hand at the next `Rebasing`
/// response (the forced-mirror of `BurstDeadline → forced`). Like `BurstDeadline`, it is filtered
/// to `None` once consumed (here: once `Reached`), so the stale entry lazy-drops.
///
/// Carried alongside [`TimerId`] on the engine's heap entry and on
/// [`crate::input::Input::TimerExpired`] so dispatch routes directly on the kind without
/// re-deriving from Profile state. The [`TimerId`] continues to act as the lazy-invalidation epoch
/// — `kind` only narrows the validation slot, it does not replace it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TimerKind {
    #[default]
    Settle,
    BurstDeadline,
    AwaitGateDeadline,
    PostFireSettle,
    RebaseCeiling,
}

/// Lifecycle of a Profile's anchor `watch_demand` contribution.
///
/// Two-state machine:
/// - [`Self::None`] — Profile holds no anchor contribution. Reachable when the Profile is `Pending`
///   (descent prefix carries the STRUCTURE watch instead), `Purged` (`Input::WatchOpRejected`
///   clamped the slot), or freshly constructed pre-attach.
/// - [`Self::Held`] — Profile contributes `+1` (at its `events` mask) to its anchor's
///   `watch_demand`. Set on the path that bumped the counter (immediate-Seed in `attach_sub_inner`
///   or descent's anchor materialization); cleared on the matching decrement (anchor terminal
///   event, reap, clamp purge).
///
/// Encoded as a sum type so the dispatch sites — `release_anchor_claim`, the recompute, every
/// `dispatch_*_vanished` — read the lifecycle directly rather than combining a flag with
/// [`ProfileState`]. The trichotomy "materialized / pending / purged" emerges from `(state,
/// anchor_claim)` rather than from a third variant: every release helper treats Purged identically
/// to None (no contribution to drop), so distinguishing them at the type level adds no dispatch
/// information.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum AnchorClaim {
    #[default]
    None,
    Held,
}

/// The settled reference a *classified* anchor compares fresh probes against, in the one window
/// each variant owns:
///
/// - [`Self::Unset`] — no settled baseline yet. A freshly-classified anchor (resource attach
///   against a known-kind slot, or descent materialisation) before its first successful graft.
///   There is nothing to drift against; the first graft installs the baseline.
/// - [`Self::Snapshot`] — active mode. The last settled snapshot; the drift verdict is
///   `current.hash() != settled.hash()`.
/// - [`Self::Witness`] — loss→recovery window. The anchor vanished and its baseline snapshot was
///   dropped, but the pre-loss anchor-rooted hash is retained so the post-recovery Seed-Ok can
///   still decide whether the tree drifted while the anchor was gone. Consumed (overwritten by
///   [`Self::Snapshot`]) at the next rebase.
///
/// `Snapshot` and `Witness` are mutually exclusive *by construction* — there is no representable
/// value carrying both a live baseline and a survival witness. The "a present baseline implies no
/// survival witness" rule is therefore a type property, not a checked convention.
#[derive(Debug)]
enum SettledState<S> {
    Unset,
    Snapshot(S),
    Witness(u128),
}

/// The per-payload operations the generic [`SettledState`] / [`AnchorClassification`] projections
/// need without re-wrapping a [`TreeSnapshot`] just to read it. Implemented once per concrete
/// anchor payload (`LeafEntry` for File anchors, `Arc<DirSnapshot>` for Dir anchors); keeps the
/// per-kind hash route and the owned re-wrap localised instead of fanned out across the accessors.
trait AnchorPayload {
    /// Anchor-rooted digest — `LeafEntry::leaf_hash` for File, `DirSnapshot::dir_hash` for Dir.
    fn payload_hash(&self) -> u128;
    /// Owned [`TreeSnapshot`] re-wrap (`Arc` bump for Dir, copy for File). The sum stores the inner
    /// payload, never a `TreeSnapshot`, so the owned-projection accessors mint the wrapper on demand.
    fn rewrap(&self) -> TreeSnapshot;
}

impl AnchorPayload for crate::snapshot::tree::LeafEntry {
    fn payload_hash(&self) -> u128 {
        self.leaf_hash()
    }
    fn rewrap(&self) -> TreeSnapshot {
        TreeSnapshot::File(self.clone())
    }
}

impl AnchorPayload for Arc<crate::snapshot::tree::DirSnapshot> {
    fn payload_hash(&self) -> u128 {
        self.dir_hash()
    }
    fn rewrap(&self) -> TreeSnapshot {
        TreeSnapshot::Dir(self.clone())
    }
}

impl<S: AnchorPayload> SettledState<S> {
    /// The settled anchor-rooted hash, or `None` when no settled reference exists yet
    /// ([`Self::Unset`]). `Snapshot` digests its payload; `Witness` returns the retained pre-loss
    /// hash directly. This is also the witness a clear captures: the value that must survive into
    /// [`AnchorClassification::Unclassified`] so a later recovery can still detect drift.
    fn to_hash(&self) -> Option<u128> {
        match self {
            Self::Unset => None,
            Self::Snapshot(s) => Some(s.payload_hash()),
            Self::Witness(h) => Some(*h),
        }
    }

    /// The owned baseline snapshot — `Some` only in active mode ([`Self::Snapshot`]). `Unset` (no
    /// baseline yet) and `Witness` (baseline dropped at loss) have no snapshot to lend.
    fn snapshot(&self) -> Option<TreeSnapshot> {
        match self {
            Self::Snapshot(s) => Some(s.rewrap()),
            Self::Unset | Self::Witness(_) => None,
        }
    }

    /// The active baseline's anchor-rooted hash — `Some` only in the `Snapshot` arm. `Unset` and
    /// `Witness` yield `None`. The hash-only sibling of [`Self::snapshot`] (no `TreeSnapshot`
    /// re-wrap) and the Snapshot-only complement of [`Self::witness_hash`] within
    /// [`Self::to_hash`]'s domain.
    fn snapshot_hash(&self) -> Option<u128> {
        match self {
            Self::Snapshot(s) => Some(s.payload_hash()),
            Self::Unset | Self::Witness(_) => None,
        }
    }

    /// The retained pre-loss hash — `Some` only across the loss→recovery window (`Witness`). An
    /// active `Snapshot` baseline and `Unset` both yield `None`: neither carries a survival witness.
    ///
    /// The Witness-only complement of [`Self::snapshot_hash`] (the Snapshot-only hash projection)
    /// within [`Self::to_hash`]'s domain — the four accessors are one lattice over the sum:
    /// `to_hash` is `Some` iff exactly one of `snapshot_hash` / `witness_hash` is, never both (the
    /// variants are disjoint), so no arm is double-counted and the witness can never be silently
    /// folded into the active-baseline projection. [`Self::snapshot`] is the owned-projection
    /// sibling of `snapshot_hash`; the algebra holds no owned witness, so the owned lattice covers
    /// Snapshot only.
    const fn witness_hash(&self) -> Option<u128> {
        match self {
            Self::Witness(h) => Some(*h),
            Self::Unset | Self::Snapshot(_) => None,
        }
    }
}

/// The anchor's on-disk classification and its settled reference, as one sum.
///
/// The discriminant *is* the anchor kind: there is no separately stored `kind` to disagree with the
/// snapshot variant. `current = Dir ⇒ kind = Dir`, `current = File ⇒ kind = File`, and
/// `unclassified ⇒ no snapshot` are structural — an ill-shaped pair cannot be constructed, so the
/// engine's typed probe-dispatch chain is the *only* place kind agreement is decided, and a clear /
/// install sequence cannot leave the pair half-written.
///
/// **`Dir.current` is dual-purpose.** Besides the drift-comparison snapshot, its entries *are* the
/// covered-descendant watch-claim membership set: [`Profile::take_current`] hands the live `Dir`
/// snapshot to the wholesale-deletion walk that releases every per-descendant contribution. A
/// parallel descendant-id collection would duplicate exactly what the snapshot already encodes (and
/// re-introduce the drift surface this sum removes); the live `Dir` snapshot is the single source
/// of that membership.
#[derive(Debug)]
enum AnchorClassification {
    /// Kind not yet known, or cleared at anchor loss. No snapshot is representable here. `witness`
    /// carries the pre-loss anchor-rooted hash across the loss window (set when the cleared anchor
    /// had a settled reference; `None` for a fresh, never-classified Profile).
    Unclassified { witness: Option<u128> },
    File {
        current: Option<crate::snapshot::tree::LeafEntry>,
        settled: SettledState<crate::snapshot::tree::LeafEntry>,
    },
    Dir {
        current: Option<Arc<crate::snapshot::tree::DirSnapshot>>,
        settled: SettledState<Arc<crate::snapshot::tree::DirSnapshot>>,
    },
}

/// Frozen config identity plus the two caches that are *total functions* of it. Private fields and
/// a sole constructor make "derived once from a frozen identity, never independently writable" a
/// structural property rather than a documented convention. The partition hash is not cached here:
/// the identity arrives already sealed over it ([`ProfileIdentity::config_hash`] is a field read),
/// so a Profile-side mirror would be a second copy of the same key.
///
/// `identity` ([`ProfileIdentity`] = `{config, max_settle, events, hash}`) is the Profile partition
/// key's config half; `exclude_strings` and `has_per_file_fds` are each a pure projection of it,
/// materialised once at [`Self::new`].
#[derive(Debug)]
struct ProfileConfig {
    identity: ProfileIdentity,
    exclude_strings: Arc<[CompactString]>,
    has_per_file_fds: bool,
}

impl ProfileConfig {
    /// Derive both caches from a frozen [`ProfileIdentity`]: `exclude_strings` projects
    /// [`ScanConfig::exclude_globs`](crate::ScanConfig::exclude_globs) in the builder-canonical
    /// order (already sorted by source, so no re-sort; the empty slice for shapes that carry no
    /// excludes); `has_per_file_fds` is true iff the event mask carries CONTENT or METADATA
    /// (covered Leaves then need their own FDs).
    fn new(identity: ProfileIdentity) -> Self {
        let has_per_file_fds = identity
            .events()
            .intersects(ClassSet::CONTENT | ClassSet::METADATA);
        let exclude_strings: Arc<[CompactString]> = identity
            .config()
            .exclude_globs()
            .iter()
            .map(|g| CompactString::from(g.source()))
            .collect();
        Self {
            identity,
            exclude_strings,
            has_per_file_fds,
        }
    }
}

/// The Profile's deferred-release obligations to the Tree refcount aggregate. The pure-step `Tree`
/// has no `Drop` reach, so each obligation is encoded as a cached id/flag here and released
/// explicitly at detach / reap / purge. Drift between this record and the Tree's contribution map
/// is a **Tree refcount leak**, so every write routes through a typed accessor that keeps the cache
/// and the Tree aggregate in lockstep.
///
/// **Scope boundary — do not widen.** This holds *only* the two homeless cached tokens whose sole
/// purpose is deferred release. It deliberately excludes the other two of the four Tree claims,
/// each of which is a side-effect of a primary concern that owns it:
/// - the **descent-prefix** claim *is* `ProfileState::Pending`'s `DescentState::current_prefix`;
///   release routes through the state machine.
/// - the **1-to-N covered-descendant** claims *are* `AnchorClassification::Dir.current`'s entries
///   (the live snapshot is the membership set; [`Profile::take_current`] hands it to the
///   wholesale-deletion walk).
///
/// Co-locating either here would duplicate that state and re-create the exact drift surface this
/// decomposition removes.
#[derive(Debug)]
struct TreeContributions {
    /// "Do I owe `sub_watch(resource, ProfileAnchor(pid))`?" — the anchor contribution flag. The
    /// reap-time trichotomy (materialized / pending / purged) emerges from `(state, anchor_claim)`,
    /// so this stays orthogonal to the classification sum.
    anchor_claim: AnchorClaim,
    /// Cached parent Resource carrying this Profile's `ContribKey::ProfileParent` STRUCTURE
    /// contribution. `None` when the anchor is itself a Tree root (root rename detection then
    /// unavailable). Also the anchor-loss recovery channel — deliberately preserved across
    /// `discard_anchor_state`; released only by reap / `WatchOpRejected` purge. A stale cache here
    /// leaks the old parent's `+1`.
    watch_root_parent: Option<ResourceId>,
}

/// How an [`AbsorbWindow`] retires.
///
/// Set by [`Profile::arm_absorb`] from the operator's `absorb` signal: a bare `absorb` (no
/// duration) is [`Self::ConsumeOnFirst`] — a one-shot cover for the single expected replication;
/// `absorb --for <dur>` is [`Self::PersistUntil`] — a time-boxed window covering a run of them.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AbsorbMode {
    /// Retire on the first *folded fire*. The window clears the moment a would-have-fired burst
    /// folds. A non-firing fold (a Cold-Seed `SilentPin`, which proves nothing) does **not**
    /// consume it — the window survives for the first burst that would genuinely have fired.
    ConsumeOnFirst,
    /// Persist until `expiry`, folding every fireable burst in between. Untouched by burst
    /// completion — rides across sequential bursts and goes inert by time alone.
    PersistUntil,
}

/// Operator `absorb` window — the per-Profile record of intent that *outlives* individual bursts (a
/// long replication transfer can outlast many settle windows).
///
/// Distinct from the per-burst fold decision (`PreFireBurst::fold_latched`), which is frozen at
/// burst birth and dies with the burst: the window is the *intent*, the latch is one burst's
/// *frozen verdict* of that intent.
///
/// **Plain data.** The lazy-expiry invariant — "a window past its `expiry` is absent" — is enforced
/// by [`Profile`] keeping its `Profile::absorb` field private and live-gating every projection
/// through [`Profile::absorb_window_if_live`] (the lone owner of the `now < expiry` rule, behind
/// both the boolean [`Profile::absorb_window_live`] consult and the `show` surface). There is
/// deliberately no `&mut` clear on the inert read: it would break the shared immutable borrow the
/// birth consult takes, and an inert window is harmless (`*_live` is `false`, `show` hides it). No
/// [`crate::TimerId`] backs it — an un-consumed window needs no wake-up to go inert.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AbsorbWindow {
    /// Wall-clock instant at or past which the window reads inert.
    pub expiry: Instant,
    /// Retirement discipline — see [`AbsorbMode`].
    pub mode: AbsorbMode,
}

/// One stability state machine per `(Resource, ProfileIdentity)`, decomposed into single-concern
/// substructures.
///
/// Only `settle` (the per-Profile mutable debounce param the engine recomputes as
/// `min(remaining_subs.settles)`) is a `pub` field; every other concern is module-private, exposing a
/// typed accessor / transition API — the cross-crate write surface is `Profile`'s `pub fn`s, never a
/// field assignment. The substructures that own a cross-field invariant: `ProfileConfig` (frozen
/// identity ⇒ derived caches), `AnchorClassification` (snapshot-shape ⊕ baseline/witness exclusion),
/// `TreeContributions` (deferred Tree releases — drift = refcount leak). The burst state machine
/// needs no such wrapper: it is the plain module-private [`ProfileState`] field `state`, read via
/// [`Self::state`] and transitioned via [`Self::transition_state`] / the typed-move accessors — its
/// variants (`Idle | Pending | Active`) and their payloads (`DescentState`, the `ActiveBurst` split)
/// are themselves the single source of every burst invariant. The `Draining → Verifying` reconfirm is
/// a *fresh query* ([`crate::ProfileState::in_active_standard_burst`] over the live tree), not a
/// cached counter. (Effect fire history is per-Sub — [`crate::Sub::has_fired`] — not a Profile
/// substructure; *fold* history is the mirror image — per-Profile, since folding is per-Profile — and
/// lives here as `Self::absorb` / its count.)
#[derive(Debug)]
pub struct Profile {
    /// The Tree slot this Profile's stability machine anchors at — the slot axis of the `(resource,
    /// config_hash)` partition key.
    ///
    /// **Write-once** at [`Profile::new`]: re-assigning this would desynchronise
    /// [`ProfileMap::by_resource`] (the secondary index by [`ResourceId`]),
    /// [`crate::Resource::profiles`] (the slot-side back-ref vector), and every reader of
    /// [`Self::resource`]. The invariant is held by encapsulation — module-private with no setter —
    /// matching the discipline on [`Self::cfg`] (`config_hash` is the other half of the partition
    /// key, frozen the same way).
    resource: ResourceId,
    /// Frozen config identity and its derived caches. Read via [`Self::config`] /
    /// [`Self::config_hash`] / [`Self::exclude_strings`] / [`Self::max_settle`] / [`Self::events`]
    /// / [`Self::has_per_file_fds`]; never independently writable (sole constructor
    /// [`ProfileConfig::new`]).
    cfg: ProfileConfig,
    /// Per-Profile mutable debounce interval. **Not identity** — `max_settle` is the identity half
    /// ([`Self::max_settle`]); `settle` the engine recomputes as `min(remaining_subs.settles)` on
    /// attach/detach. Stays `pub`: `recompute_profile_settle` writes it directly and there is no
    /// cross-field invariant to guard here (the config layer's `validate_settle` is the `settle <=
    /// max_settle` boundary; [`Self::new`] debug-asserts it).
    pub settle: Duration,
    /// The anchor's classification and settled reference as one sum (kind ⊕ live snapshot ⊕ settled
    /// baseline ⊕ survival witness). The discriminant *is* the kind; "no snapshot while
    /// unclassified" and "no baseline while a survival witness is held" hold by construction.
    /// Reads: [`Self::kind`] / [`Self::current`] / [`Self::baseline`] / [`Self::current_dir`] /
    /// [`Self::baseline_dir`] / [`Self::settled_hash`] / [`Self::current_is_some`]. Writes:
    /// [`Self::install_dir_current`] / [`Self::install_file_current`] / [`Self::rebase_baseline`] /
    /// [`Self::take_current`] / [`Self::clear_anchor_classification`] /
    /// [`Self::materialize_anchor`]. `Resource.kind` is a separate Tree-side parallel cache the
    /// engine never consults for the anchor's own kind in any post-attach path.
    anchor: AnchorClassification,
    /// Burst state machine. Module-private; the variant payloads carry every burst invariant by
    /// construction (the [`ActiveBurst`] split type-bans cross-phase field leaks), so no wrapper or
    /// sidecar counter is needed. Read via [`Self::state`], transitioned via
    /// [`Self::transition_state`] / the typed-move accessors. The `Draining → Verifying` reconfirm
    /// is a fresh query over the live tree ([`ProfileState::in_active_standard_burst`]), not cached
    /// here.
    state: ProfileState,
    /// Deferred-release obligations to the Tree refcount aggregate (`anchor_claim`,
    /// `watch_root_parent`). Drift = refcount leak. Read via [`Self::anchor_claim`] /
    /// [`Self::watch_root_parent`]; written via [`Self::install_anchor_claim_held`] /
    /// [`Self::release_anchor_claim_now`] / [`Self::set_watch_root_parent`] /
    /// [`Self::take_watch_root_parent`].
    contributions: TreeContributions,
    /// Operator `absorb` window, or `None` when no fold is armed. The per-Profile *intent* that
    /// drives each burst's per-burst fold latch: a burst consults it once at birth
    /// ([`Self::absorb_window_live`]) to freeze [`PreFireBurst::fold_latched`]. Private — the
    /// lazy-expiry invariant lives here, not on the plain-data [`AbsorbWindow`]. Written only by
    /// [`Self::arm_absorb`] (set + retro-latch the in-flight burst) and [`Self::note_absorb_fold`]
    /// (consume-on-first); read live-gated via [`Self::absorb_window_if_live`] (and its boolean
    /// [`Self::absorb_window_live`]), raw via [`Self::absorb_window`].
    absorb: Option<AbsorbWindow>,
    /// Count of folds this Profile has absorbed — the per-Profile mirror of per-Sub fire history
    /// (`Sub::fire_count`). A fold is per-Profile *by construction* (every Sub on a Profile folds
    /// together), so the count lives where its identity is and shares the window's lifetime (both
    /// reset when a config-hash change rebuilds the Profile). Bumped by [`Self::note_absorb_fold`];
    /// read via [`Self::absorb_count`] and projected per-Sub at the `show` boundary.
    absorb_count: u64,
}

impl Profile {
    /// Construct a fresh Profile: state `Idle` (no burst-finish directive yet), no
    /// baseline/current, no watch-root parent. (Effect fire history is per-Sub —
    /// [`crate::Sub::has_fired`] — not a Profile concern.)
    ///
    /// `identity` ([`ProfileIdentity`] = `{config, max_settle, events, hash}`) is the Profile
    /// partition key's config half, taken by value: it arrives already sealed over its canonical
    /// hash, and `ProfileConfig::new` derives the `exclude_strings` / `has_per_file_fds`
    /// projections from it. There is no path to a Profile with an unset or stale hash. The sole
    /// production caller (`find_or_create_profile`) already holds the `ProfileIdentity` and moves
    /// it straight in — no field unpack, no clone.
    ///
    /// `settle` is the per-Profile mutable debounce interval (recomputed by the engine as
    /// `min(remaining_subs.settles)`), distinct from the identity's `max_settle`. The `settle <=
    /// max_settle` relation is a `debug_assert!`: the config layer's `validate_settle` is the real
    /// boundary (it enforces `max_settle >= 4 × settle`); a breach here means a caller bypassed
    /// config validation.
    ///
    /// `kind` is the anchor's classified shape at construction, projected into the
    /// `AnchorClassification` sum: `None` ⇒ `Unclassified` (a `DescentScaffold` or freshly-`ensure`d
    /// slot; descent materialisation classifies it via [`Self::materialize_anchor`], the first
    /// Seed-Ok via [`Self::install_dir_current`] / [`Self::install_file_current`]); `Some(Dir)` /
    /// `Some(File)` ⇒ a classified anchor with no snapshot or baseline yet (the first probe response
    /// grafts the current snapshot). `Some(Unknown)` is defensively dead: `Resource::kind()` maps
    /// `Unknown → None`, so the sole production caller never threads `Some(Unknown)`; the arm is
    /// debug-asserted and degrades to `Unclassified` (the same shape as `None`) in release rather
    /// than panicking or constructing an illegal state.
    #[must_use]
    pub fn new(
        resource: ResourceId,
        identity: ProfileIdentity,
        settle: Duration,
        kind: Option<ResourceKind>,
    ) -> Self {
        debug_assert!(
            settle <= identity.max_settle(),
            "Profile::new: settle ({settle:?}) must not exceed max_settle ({:?}) — \
             config-layer validate_settle enforces max_settle >= 4 × settle; a \
             breach here means a caller bypassed config validation",
            identity.max_settle(),
        );
        let anchor = match kind {
            None => AnchorClassification::Unclassified { witness: None },
            Some(ResourceKind::Dir) => AnchorClassification::Dir {
                current: None,
                settled: SettledState::Unset,
            },
            Some(ResourceKind::File) => AnchorClassification::File {
                current: None,
                settled: SettledState::Unset,
            },
            Some(ResourceKind::Unknown) => {
                debug_assert!(
                    false,
                    "Profile::new: Resource::kind() yields Unknown→None, never Some(Unknown)",
                );
                AnchorClassification::Unclassified { witness: None }
            }
        };
        Self {
            resource,
            cfg: ProfileConfig::new(identity),
            settle,
            anchor,
            state: ProfileState::Idle,
            contributions: TreeContributions {
                anchor_claim: AnchorClaim::None,
                watch_root_parent: None,
            },
            absorb: None,
            absorb_count: 0,
        }
    }

    /// Graft a Dir-shaped `current` into the anchor classification. Sole legitimate writer of the
    /// Dir `current` slot.
    ///
    /// - From `Unclassified`: classify as `Dir`, carrying any survival witness forward into
    ///   `settled` (recovery: `Witness(h)`; fresh: `Unset`). The witness must survive
    ///   classification so the post-recovery drift verdict still has a reference.
    /// - From `Dir`: overwrite `current`, leaving `settled` untouched (a re-graft within the same
    ///   materialised epoch — fresh or mid-recovery).
    /// - From `File`: a `File`-kinded Profile receiving a `Dir` graft is a dispatcher routing
    ///   breach. The certifier's inline kind guard catches this and routes through
    ///   `finalize_anchor_lost` (which clears to `Unclassified`) *before* any graft, so this arm is
    ///   defensively dead; `debug_assert!` flags a future boundary bypass and release builds
    ///   re-classify rather than construct an illegal pair.
    pub fn install_dir_current(&mut self, snapshot: Arc<crate::snapshot::tree::DirSnapshot>) {
        match &mut self.anchor {
            AnchorClassification::Dir { current, .. } => {
                *current = Some(snapshot);
            }
            AnchorClassification::Unclassified { witness } => {
                let settled = witness.map_or(SettledState::Unset, SettledState::Witness);
                self.anchor = AnchorClassification::Dir {
                    current: Some(snapshot),
                    settled,
                };
            }
            AnchorClassification::File { .. } => {
                debug_assert!(
                    false,
                    "install_dir_current: kind mismatch (File-kinded Profile \
                     received a Dir graft — dispatcher boundary bypassed)",
                );
                self.anchor = AnchorClassification::Dir {
                    current: Some(snapshot),
                    settled: SettledState::Unset,
                };
            }
        }
    }

    /// Graft a File-shaped `current` into the anchor classification. Symmetric with
    /// [`Self::install_dir_current`]: carries the survival witness forward from `Unclassified`,
    /// overwrites `current` from `File` leaving `settled` untouched, and treats a `Dir`-kinded
    /// Profile as the defensively-dead dispatcher breach.
    pub fn install_file_current(&mut self, leaf: crate::snapshot::tree::LeafEntry) {
        match &mut self.anchor {
            AnchorClassification::File { current, .. } => {
                *current = Some(leaf);
            }
            AnchorClassification::Unclassified { witness } => {
                let settled = witness.map_or(SettledState::Unset, SettledState::Witness);
                self.anchor = AnchorClassification::File {
                    current: Some(leaf),
                    settled,
                };
            }
            AnchorClassification::Dir { .. } => {
                debug_assert!(
                    false,
                    "install_file_current: kind mismatch (Dir-kinded Profile \
                     received a File graft — dispatcher boundary bypassed)",
                );
                self.anchor = AnchorClassification::File {
                    current: Some(leaf),
                    settled: SettledState::Unset,
                };
            }
        }
    }

    /// Settle the live `current` snapshot as the new baseline: `settled := Snapshot(current)`. Any
    /// survival witness is *consumed* — the `Witness → Snapshot` move is the structural end of the
    /// loss→recovery window (no separate witness-clear step exists).
    ///
    /// Called only at a **terminal pin**, after a successful graft where `current.is_some()` holds:
    /// - `dispatch_rebase_ok` on [`QuiescenceVerdict::Stable`] — both `StableReason::Natural` (two
    ///   settle-spaced equal post-command samples) and `StableReason::Forced` (the bounded
    ///   rebase-loop terminal — pin the freshest observation against the ceiling).
    /// - the Seed-Ok recovery pin — the `EmitMode::SeedDrift` seal in the engine's
    ///   `fire_and_settle`, and the silent `SilentPin` pin — reached from the
    ///   [`QuiescenceVerdict::Stable`] Seed verdicts (both `Natural` and `Forced`).
    ///
    /// The rebase-loop [`QuiescenceVerdict::Retry`] arm (not yet at the ceiling) and the Seed
    /// [`QuiescenceVerdict::Retry`] arm graft (or skip) but **do not** rebase: the witness-survival
    /// contract — the survival witness must outlive an unbounded re-batch / rebase loop and be
    /// consumed only at the eventual terminal pin, so this consumer must never run on a looping arm.
    pub fn rebase_baseline(&mut self) {
        match &mut self.anchor {
            AnchorClassification::Dir { current, settled } => {
                debug_assert!(
                    current.is_some(),
                    "rebase_baseline: Dir current must be set at every post-graft caller",
                );
                if let Some(c) = current {
                    *settled = SettledState::Snapshot(Arc::clone(c));
                }
            }
            AnchorClassification::File { current, settled } => {
                debug_assert!(
                    current.is_some(),
                    "rebase_baseline: File current must be set at every post-graft caller",
                );
                if let Some(c) = current {
                    *settled = SettledState::Snapshot(c.clone());
                }
            }
            AnchorClassification::Unclassified { .. } => {
                debug_assert!(
                    false,
                    "rebase_baseline: called on an Unclassified anchor (no current to settle)",
                );
            }
        }
    }

    /// Clear the anchor classification at anchor loss, capturing the
    /// survival witness in one move: `File`/`Dir` ⇒ `Unclassified {
    /// witness: settled.to_hash() }`. The witness is the settled reference's hash (`Snapshot`
    /// digests; `Witness` passes through; `Unset` ⇒ `None`), so a post-recovery Seed-Ok can still
    /// detect drift after the baseline snapshot is gone. Idempotent against an
    /// already-`Unclassified` anchor: the prior witness is preserved, never overwritten with
    /// `None`. Inverse of [`Self::materialize_anchor`].
    pub fn clear_anchor_classification(&mut self) {
        let witness = match &self.anchor {
            AnchorClassification::Unclassified { witness } => *witness,
            AnchorClassification::File { settled, .. } => settled.to_hash(),
            AnchorClassification::Dir { settled, .. } => settled.to_hash(),
        };
        self.anchor = AnchorClassification::Unclassified { witness };
    }

    /// Atomically install a descent-materialised anchor: transition `Pending → Idle`, install the
    /// claim, and classify the anchor with the discovered `kind`, **carrying the survival witness
    /// forward** (`Unclassified { witness } ⇒ File/Dir { current: None, settled: Witness(h) | Unset
    /// }`). Reached only through [`ProfileMap::materialize_anchor`] (which reconciles the carrier
    /// count across the `Pending → Idle` edge); the engine's descent terminus launches the Seed
    /// burst on the next statement — the `Idle` written here is a structural intermediate, never
    /// observed. The whole sequence runs under one `&mut self` so no reader sees a partial write.
    /// Inverse of [`Self::clear_anchor_classification`].
    ///
    /// Debug-asserts the fresh-materialisation preconditions (`state == Pending`, no claim, anchor
    /// `Unclassified`); release builds compile the asserts out and still classify atomically.
    pub fn materialize_anchor(&mut self, kind: ResourceKind) {
        debug_assert!(
            matches!(self.state, ProfileState::Pending(_)),
            "materialize_anchor: state must be Pending (was {:?})",
            self.state.discriminant(),
        );
        debug_assert!(
            matches!(self.contributions.anchor_claim, AnchorClaim::None),
            "materialize_anchor: anchor_claim must be None",
        );
        debug_assert!(
            matches!(self.anchor, AnchorClassification::Unclassified { .. }),
            "materialize_anchor: anchor must be Unclassified (already classified)",
        );
        let witness = match &self.anchor {
            AnchorClassification::Unclassified { witness } => *witness,
            AnchorClassification::File { .. } | AnchorClassification::Dir { .. } => None,
        };
        self.state = ProfileState::Idle;
        self.contributions.anchor_claim = AnchorClaim::Held;
        self.anchor = match kind {
            ResourceKind::Dir => AnchorClassification::Dir {
                current: None,
                settled: witness.map_or(SettledState::Unset, SettledState::Witness),
            },
            ResourceKind::File => AnchorClassification::File {
                current: None,
                settled: witness.map_or(SettledState::Unset, SettledState::Witness),
            },
            ResourceKind::Unknown => {
                debug_assert!(
                    false,
                    "materialize_anchor: kind Unknown (descent feeds a real dirent kind)",
                );
                AnchorClassification::Unclassified { witness }
            }
        };
        self.debug_assert_anchor_coherent();
    }

    /// Debug-time coherence tripwire for the multi-field classification coordinators (this
    /// `materialize_anchor` and the engine's `discard_anchor_state`).
    ///
    /// The snapshot-shape (`kind ⇔ current` variant) and baseline/witness-exclusion invariants are
    /// *structural* — no representable `AnchorClassification` violates them, so there is nothing to
    /// check there. What remains is the cross-axis invariants the type system does not cover,
    /// asserted here so a future coordinator that leaves a `Pending` / `Parked` Profile classified
    /// (or holding the anchor claim) trips at the write site rather than latently at the next
    /// dispatch / reap:
    /// - `Pending ⇒ Unclassified` — during descent the anchor is not probed; the descent prefix,
    ///   not the anchor, carries the watch.
    /// - `Pending ⇒ ¬AnchorClaim::Held` — the descent prefix carries the STRUCTURE watch; the anchor
    ///   claim is installed only at materialisation. (`reap_profile`'s trichotomy depends on this.)
    /// - `Parked ⇒ Unclassified ∧ ¬AnchorClaim::Held` — the mirror pair: a park is anchorless rest,
    ///   entered only after `discard_anchor_state` (or from a `Pending` descent, where the pair
    ///   already held), so a classified or claim-holding park is a coordinator regression.
    pub fn debug_assert_anchor_coherent(&self) {
        if matches!(self.state, ProfileState::Pending(_) | ProfileState::Parked) {
            debug_assert!(
                matches!(self.anchor, AnchorClassification::Unclassified { .. }),
                "anchor coherence: a {:?} Profile must be Unclassified",
                self.state.discriminant(),
            );
            debug_assert!(
                matches!(self.contributions.anchor_claim, AnchorClaim::None),
                "anchor coherence: a {:?} Profile must not hold the anchor claim",
                self.state.discriminant(),
            );
        }
    }

    /// General-purpose **push** `state` writer: installs the given `new` and returns the prior via
    /// `mem::replace`. Reached (through [`ProfileMap::transition_state`]) by the
    /// install-a-given-state paths — `start_seed_burst` / `start_standard_burst` / descent
    /// materialisation / the claims-ledger Idle reset — which discard the returned prior, relying
    /// only on its drop (the claims reset drops a disarmed `Pending` descent this way). The
    /// **transform** dual — compute `new` from the *consumed* prior, for the typed fire-boundary
    /// moves — is [`Self::map_state`]. Preconditions live at the engine boundary (`require_idle` /
    /// `require_active_pre_fire`), not here.
    ///
    /// [`Self::materialize_anchor`] is the single documented bypass — a three-field atomic `Pending
    /// → (Idle, AnchorClaim::Held, classified)` write; [`ProfileMap::materialize_anchor`] is its
    /// counter-reconciling wrapper (the bypass is a genuine `Pending → Idle` carrier-count edge).
    pub const fn transition_state(&mut self, new: ProfileState) -> ProfileState {
        std::mem::replace(&mut self.state, new)
    }

    /// Transform `state` in place: extract the prior by value, hand it to `f`, and install the
    /// [`ProfileState`] `f` returns (alongside its auxiliary `R`, threaded back out). The
    /// **transform** dual of [`Self::transition_state`]'s **push** — for the callers that must
    /// *consume* the prior to compute the next: the typed fire-boundary moves
    /// [`PreFireBurst::into_post_fire`] / [`PostFireBurst::into_pre_fire_residual`] and the
    /// burst-end `finish_burst_to_idle`. `transition_state` cannot serve them — it wants `new` up
    /// front, but `new = f(old)` needs `old` extracted first.
    ///
    /// The `Idle` the `mem::replace` parks while `f` runs is never observed: the engine step is
    /// synchronous and single-threaded, and `f`'s returned state overwrites it before this returns.
    /// A panic inside `f` would leave `state == Idle` with the prior burst dropped — but in release
    /// the typed moves don't panic, and the swap-to-Idle dance this replaces had the identical
    /// property, so no regression and no `catch_unwind`.
    ///
    /// Not `const` — it invokes `f`, which `const fn` cannot; `transition_state` stays `const`.
    /// Preconditions live at the engine boundary, not here.
    pub fn map_state<R>(&mut self, f: impl FnOnce(ProfileState) -> (ProfileState, R)) -> R {
        let prior = std::mem::replace(&mut self.state, ProfileState::Idle);
        let (next, r) = f(prior);
        self.state = next;
        r
    }

    /// Install the anchor claim. Idempotent against `Held`. Production caller:
    /// `Engine::bootstrap_immediate`. (The descent-materialised claim rides
    /// [`Self::materialize_anchor`]'s bundled write instead.)
    pub const fn install_anchor_claim_held(&mut self) {
        self.contributions.anchor_claim = AnchorClaim::Held;
    }

    /// Release the anchor claim. Idempotent against `None`. Production caller:
    /// `Engine::release_anchor_claim`, which wraps this with the Tree-side `sub_watch`.
    pub const fn release_anchor_claim_now(&mut self) {
        self.contributions.anchor_claim = AnchorClaim::None;
    }

    /// The cached watch-root parent Resource, if this Profile owes a `ContribKey::ProfileParent`
    /// STRUCTURE contribution there. `None` for a root anchor. Read seam over the release-ledger
    /// field; `Engine::set_watch_root_parent` uses it for the cache-coherence and idempotence
    /// checks, `classify_event_carriers` for the anchor-recovery channel.
    #[must_use]
    pub const fn watch_root_parent(&self) -> Option<ResourceId> {
        self.contributions.watch_root_parent
    }

    /// Cache the watch-root parent id. The single write seam, wrapped by
    /// `Engine::set_watch_root_parent` (which also installs the Tree-side `add_watch` and the
    /// cache-coherence `debug_assert!`). Plain set — idempotence and coherence are the engine
    /// wrapper's concern, not duplicated here.
    pub const fn set_watch_root_parent(&mut self, parent: ResourceId) {
        self.contributions.watch_root_parent = Some(parent);
    }

    /// Take the cached watch-root parent, clearing it — the symmetric deferred-release primitive
    /// (`Engine::release_watch_root_parent_claim` keys the `sub_watch` removal off the returned
    /// id). Idempotent: a second call returns `None`.
    pub const fn take_watch_root_parent(&mut self) -> Option<ResourceId> {
        self.contributions.watch_root_parent.take()
    }

    /// Borrow the pre-fire burst payload iff `state == Active(PreFire(_), _)` — the `&self` mirror
    /// of [`Self::pre_fire_burst_mut`]. A read of the state's structural shape, never a transition;
    /// the engine's pre-fire dispatch reads (the [`PreFirePhase::Verifying`] target, the Seed
    /// first-fire witness `dirty`) route through this instead of re-matching `state()` inline.
    #[must_use]
    pub const fn pre_fire_burst(&self) -> Option<&PreFireBurst> {
        match &self.state {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(pre),
            _ => None,
        }
    }

    /// Borrow the pre-fire burst payload iff `state == Active(PreFire(_), _)` — a read of the
    /// state's structural shape, *not* a variant transition (the variant-level move still routes
    /// through [`Self::transition_state`]). Sole production caller surface: `burst.rs` named
    /// helpers — the single-source-of-mutation rule for `Active(_)` phase fields, inherited by the
    /// symmetric [`Self::post_fire_burst_mut`].
    pub const fn pre_fire_burst_mut(&mut self) -> Option<&mut PreFireBurst> {
        match &mut self.state {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(pre),
            _ => None,
        }
    }

    /// Borrow the post-fire burst payload iff `state == Active(PostFire(_), _)` — the `&self`
    /// mirror of [`Self::post_fire_burst_mut`], symmetric with [`Self::pre_fire_burst`]. The
    /// engine's post-fire dispatch reads (the forced-ceiling `retry_streak` consult) route through
    /// this instead of re-matching `state()` inline.
    #[must_use]
    pub const fn post_fire_burst(&self) -> Option<&PostFireBurst> {
        match &self.state {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => Some(post),
            _ => None,
        }
    }

    /// Symmetric with [`Self::pre_fire_burst_mut`] for the post-fire payload.
    pub const fn post_fire_burst_mut(&mut self) -> Option<&mut PostFireBurst> {
        match &mut self.state {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => Some(post),
            _ => None,
        }
    }

    /// Borrow the state machine. The universal read path — every `&self` [`ProfileState`]
    /// projection (`discriminant`, `burst_finish`, `detach_lifecycle`, `timer_token`,
    /// `is_draining`, `descent_state`) routes through this.
    #[must_use]
    pub const fn state(&self) -> &ProfileState {
        &self.state
    }

    #[must_use]
    pub const fn anchor_claim(&self) -> AnchorClaim {
        self.contributions.anchor_claim
    }

    /// Anchor kind discriminant — the sum's variant projected back to the engine's
    /// `Option<ResourceKind>` shape. `Unclassified ⇒ None`; `File ⇒ Some(File)`; `Dir ⇒ Some(Dir)`.
    #[must_use]
    pub const fn kind(&self) -> Option<ResourceKind> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { .. } => Some(ResourceKind::File),
            AnchorClassification::Dir { .. } => Some(ResourceKind::Dir),
        }
    }

    /// The Profile's user-declared event-class mask. Invariant for the Profile's lifetime (folds
    /// into `config_hash`). Stable read seam over the frozen identity.
    #[must_use]
    pub const fn events(&self) -> ClassSet {
        self.cfg.identity.events()
    }

    /// The frozen [`ScanConfig`] half of the Profile identity. Borrow for the named scope
    /// predicates (`accepts*` / `descends_into` coverage reads, the witness-class requirement) —
    /// consumers never destructure the shape. Not `const`: the read derefs through the identity's
    /// `Arc`. The probe wire takes the sharing handle via [`Self::config_shared`] instead.
    #[must_use]
    pub fn config(&self) -> &ScanConfig {
        self.cfg.identity.config()
    }

    /// The same frozen [`ScanConfig`] behind its sharing handle — for the probe-emission choke,
    /// which ships the config cross-thread on `ProbeRequest::Subtree` as a refcount bump. Every
    /// in-engine reader borrows through [`Self::config`] instead.
    #[must_use]
    pub const fn config_shared(&self) -> &Arc<ScanConfig> {
        self.cfg.identity.config_shared()
    }

    /// The Tree slot this Profile anchors at — the slot axis of the `(resource, config_hash)`
    /// partition key. Write-once at [`Self::new`]; see the field rustdoc for the load-bearing
    /// invariant.
    #[must_use]
    pub const fn resource(&self) -> ResourceId {
        self.resource
    }

    /// The lifetime-stable canonical config hash — the config axis of the `(resource, config_hash)`
    /// partition key. A read through the frozen identity, which seals the hash at its own
    /// construction ([`ProfileIdentity::new`]).
    #[must_use]
    pub const fn config_hash(&self) -> u64 {
        self.cfg.identity.config_hash()
    }

    /// The settle-deadline ceiling — the identity half of the burst timings (folds into
    /// `config_hash`; invariant for the Profile's lifetime, in deliberate contrast to the mutable
    /// `settle`).
    #[must_use]
    pub const fn max_settle(&self) -> Duration {
        self.cfg.identity.max_settle()
    }

    /// True iff covered Leaves need their own FDs (the event mask carries CONTENT or METADATA).
    /// Invariant for the Profile's lifetime — the reconciler reads it to decide per-file watch
    /// installation.
    #[must_use]
    pub const fn has_per_file_fds(&self) -> bool {
        self.cfg.has_per_file_fds
    }

    /// True iff settle-window silence is a sufficient quiescence witness for this Profile — the
    /// events mask covers the classes its scan shape requires
    /// ([`ScanConfig::quiescence_witness_classes`]). The criterion is shape-owned: the shape
    /// determines the proof object (subtree content hash vs match set), the proof object determines
    /// which change classes could cross a settle window invisibly, and the per-class rationale (with
    /// the kernel-event-vocabulary assumption) lives at the [`ClassSet`] constants. This method is
    /// the composition of the two frozen identity halves and holds no shape knowledge itself.
    ///
    /// `false` signals that fire-bearing bursts require the hash-equality witness across two
    /// consecutive Authoritative samples — the Layer-C safety net for events-incomplete masks.
    /// Invariant for the Profile's lifetime (both inputs fold into `config_hash`).
    ///
    /// See also `Engine::owes_proof_from` — the orthogonal predicate selecting *which* bursts owe a
    /// proof. The two compose at the witness-selection join inside `Engine::certify_probe_response`.
    /// The emission-time consumer is [`Self::event_chains_prove_quiescence`] — the probe target and
    /// obligation reach this primitive only through it, never directly.
    /// Not `const`: reads the shape through [`Self::config`]'s `Arc` deref.
    #[must_use]
    pub fn events_witness_quiescence(&self) -> bool {
        self.events()
            .contains(self.config().quiescence_witness_classes())
    }

    /// True iff the event-dirty chains alone can carry this burst's quiescence proof: the burst
    /// re-verifies against a trusted prior ([`BurstIntent::Standard`]) AND the event stream names
    /// every change class that could move the proof object invisibly
    /// ([`Self::events_witness_quiescence`]). `false` ⇒ the probe must be anchor-rooted and
    /// `WholeSubtree`-obligated: no region may be skipped against the baseline, because either no
    /// trusted prior exists (Seed) or the stream cannot witness in-place writes — the two
    /// emission-equivalent reasons for a full fresh read.
    ///
    /// The emission-time twin of the verdict floor's witness selection
    /// (`Engine::certify_probe_response`, which engages the hash channel iff the burst owes proof
    /// AND `!events_witness_quiescence()`). Sharing this one source is what makes "hash channel
    /// engaged ⇒ both samples were full fresh anchor-rooted reads of the proof object" hold by
    /// construction: a `Chains`-obligated sample would `Arc`-clone every off-chain frame with an
    /// unchanged mtime from the stale baseline, so two such samples agree *by construction* on
    /// exactly the regions an events-incomplete mask cannot witness — the channel would certify a
    /// tree that demonstrably changed. The two halves are individually necessary: a chains-scoped
    /// frame leaves out-of-LCA regions out of frame, and an anchor-rooted `Chains` walk still
    /// skips off-chain frames.
    ///
    /// Exactly two consumers — the pre-fire probe-target rule (`pre_fire_target`) and the emission
    /// choke's obligation arm (`Engine::probe_emission_request`); they key the LCA-vs-anchor and
    /// `Chains`-vs-`WholeSubtree` decisions on the same call, so target and obligation cannot
    /// diverge. The verdict floor deliberately does *not* consume this predicate: its question
    /// (`owes_proof && !events_witness`) differs — an events-complete triggered Seed walks
    /// `WholeSubtree` yet folds `EventsReliable`.
    #[must_use]
    pub fn event_chains_prove_quiescence(&self, intent: BurstIntent) -> bool {
        matches!(intent, BurstIntent::Standard) && self.events_witness_quiescence()
    }

    /// The substitution-side projection of `ScanConfig.exclude` (source strings, builder-canonical
    /// order). Returned by reference so the effect emitter `Arc::clone`s it rather than rebuilding.
    #[must_use]
    pub const fn exclude_strings(&self) -> &Arc<[CompactString]> {
        &self.cfg.exclude_strings
    }

    /// The settled baseline as an owned [`TreeSnapshot`] — `Some` only in active mode (a settled
    /// `Snapshot`). The sum stores the inner payload, not a `TreeSnapshot`, so this mints the
    /// wrapper (Arc bump for Dir, copy for File). `Unclassified`, a not-yet-settled anchor, and the
    /// loss-window witness all yield `None`. Hash-only readers should prefer
    /// [`Self::baseline_hash`] (no re-wrap).
    #[must_use]
    pub fn baseline(&self) -> Option<TreeSnapshot> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { settled, .. } => settled.snapshot(),
            AnchorClassification::Dir { settled, .. } => settled.snapshot(),
        }
    }

    /// The live `current` snapshot as an owned [`TreeSnapshot`]. Minted on demand (Arc bump for Dir,
    /// copy for File) — the sum cannot lend a `&TreeSnapshot` it does not store in that shape. Hot
    /// Dir readers that only need the inner `Arc` should prefer [`Self::current_dir`] (no re-wrap);
    /// presence-only readers [`Self::current_is_some`]; hash-only readers [`Self::current_hash`].
    #[must_use]
    pub fn current(&self) -> Option<TreeSnapshot> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { current, .. } => {
                current.as_ref().map(AnchorPayload::rewrap)
            }
            AnchorClassification::Dir { current, .. } => {
                current.as_ref().map(AnchorPayload::rewrap)
            }
        }
    }

    /// Borrow the live Dir `current` snapshot's `Arc` directly — the reconcile / probe hot path
    /// that wants `Arc::clone`, not an owned `TreeSnapshot` re-wrap. `None` for File-kinded,
    /// Unclassified, or current-absent anchors.
    #[must_use]
    pub const fn current_dir(&self) -> Option<&Arc<crate::snapshot::tree::DirSnapshot>> {
        match &self.anchor {
            AnchorClassification::Dir {
                current: Some(arc), ..
            } => Some(arc),
            _ => None,
        }
    }

    /// Borrow the settled Dir baseline's `Arc` directly — symmetric with [`Self::current_dir`] for
    /// the settled `Snapshot`. `None` unless the anchor is `Dir` in active mode.
    #[must_use]
    pub const fn baseline_dir(&self) -> Option<&Arc<crate::snapshot::tree::DirSnapshot>> {
        match &self.anchor {
            AnchorClassification::Dir {
                settled: SettledState::Snapshot(arc),
                ..
            } => Some(arc),
            _ => None,
        }
    }

    /// The settled baseline's anchor-rooted hash — `Some` only in active mode (a settled
    /// `Snapshot`). The hash-only complement of [`Self::baseline`] (no `TreeSnapshot` re-wrap). The
    /// Snapshot-only narrower complement of [`Self::settled_hash`], which also folds the
    /// loss-window `Witness` and the `Unclassified { witness }` arms. `Unclassified`, a
    /// not-yet-settled anchor, and the loss-window witness all yield `None`.
    #[must_use]
    pub fn baseline_hash(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { settled, .. } => settled.snapshot_hash(),
            AnchorClassification::Dir { settled, .. } => settled.snapshot_hash(),
        }
    }

    /// The live `current` snapshot's anchor-rooted hash. The hash-only complement of
    /// [`Self::current`] (no `TreeSnapshot` re-wrap; the presence-only sibling is
    /// [`Self::current_is_some`]). `Unclassified` and a current-absent anchor both yield `None`.
    #[must_use]
    pub fn current_hash(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { current, .. } => {
                current.as_ref().map(AnchorPayload::payload_hash)
            }
            AnchorClassification::Dir { current, .. } => {
                current.as_ref().map(AnchorPayload::payload_hash)
            }
        }
    }

    /// The settled anchor-rooted hash the post-recovery drift verdict compares `current` against —
    /// one total function over the sum: active-mode `Snapshot` digests its payload, the loss-window
    /// `Witness` passes its retained hash through, the `Unclassified` arm yields its carried
    /// witness, and a not-yet-settled anchor yields `None`. The disjoint union of
    /// [`Self::baseline_hash`] (Snapshot arm), [`Self::survival_witness`] (Witness arm), and the
    /// `Unclassified { witness }` carried hash — each input arm contributes to exactly one summand.
    #[must_use]
    pub fn settled_hash(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { witness } => *witness,
            AnchorClassification::File { settled, .. } => settled.to_hash(),
            AnchorClassification::Dir { settled, .. } => settled.to_hash(),
        }
    }

    /// The loss-window survival witness: `Some(h)` iff the settled reference is *currently* a
    /// not-yet-consumed `Witness` (the pre-loss anchor-rooted hash retained across an anchor-loss
    /// window), not an active baseline `Snapshot` and not `Unset`.
    ///
    /// **Deliberately narrower than [`Self::settled_hash`]; the two must not be unified.**
    /// `settled_hash` is the *total* drift oracle — "what hash does the post-recovery verdict diff
    /// `current` against" — and so folds `Snapshot`, `Witness`, and the pre-classification
    /// `Unclassified { witness }` into one value. This accessor answers the strictly narrower
    /// question "is the anchor *right now* sitting on a live loss→recovery witness", true solely
    /// between the witness lift ([`Self::materialize_anchor`] / the `install_*_current` `Unclassified
    /// { witness } ⇒ classified { Witness }` arms) and its consumption ([`Self::rebase_baseline`],
    /// `Witness ⇒ Snapshot`). `settled_hash`'s `Snapshot` arm (an active baseline is not a survival
    /// witness) and its `Unclassified` arm (recovery has not completed) would each mis-answer it.
    ///
    /// `Unclassified ⇒ None` is correct on both counts above and, at the sole consumer — a Seed-Ok
    /// past `apply_snapshot`, which has classified the anchor — unreachable.
    #[must_use]
    pub const fn survival_witness(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { settled, .. } => settled.witness_hash(),
            AnchorClassification::Dir { settled, .. } => settled.witness_hash(),
        }
    }

    /// Whether a live `current` snapshot is present, without minting (or `Arc`-bumping) one. The
    /// zero-cost presence check for readers that branch on "has the anchor been grafted yet?"
    /// rather than consuming the snapshot.
    #[must_use]
    pub const fn current_is_some(&self) -> bool {
        matches!(
            &self.anchor,
            AnchorClassification::File {
                current: Some(_),
                ..
            } | AnchorClassification::Dir {
                current: Some(_),
                ..
            }
        )
    }

    /// Whether this Profile can possibly *carry* an `FsEvent` dispatch responsibility — the
    /// membership predicate of [`ProfileMap`]'s `nonsteady` carrier count, and the single source
    /// both that counter's delta and its debug full-scan tripwire read.
    ///
    /// A carrier is either a `Pending` descent (`current_prefix == R`) or a `Parked` Profile
    /// (recovery via `watch_root_parent == Some(R)` or its own co-claimed anchor slot). This is a
    /// **pure state predicate** over exactly those two variants: every true carrier satisfies it
    /// (soundness — the count-gate never under-counts; a channel-less park is counted too, a sound
    /// over-approximation), and it is *tight* in the dimension that matters — a healthy `Idle`
    /// Profile and a steady `Active` burst are excluded, so a quiet watcher coexisting with a storm
    /// does not pin the count above zero.
    ///
    /// Because the predicate reads only `state`, every edge flows through the [`ProfileMap`]
    /// chokepoints: [`ProfileMap::transition_state`] / [`ProfileMap::map_state`] (the push and
    /// transform writers) and [`ProfileMap::materialize_anchor`] (the wrapper over
    /// [`Profile::materialize_anchor`]'s three-field atomic write, whose `Pending → Idle` move is a
    /// genuine `true → false` counting edge). [`ProfileMap::attach`] / [`ProfileMap::detach`]
    /// derive the membership edges from the predicate directly.
    #[must_use]
    pub const fn is_nonsteady(&self) -> bool {
        matches!(self.state, ProfileState::Pending(_) | ProfileState::Parked)
    }

    /// Whether a settled baseline `Snapshot` is present, without minting (or `Arc`-bumping) one —
    /// the zero-cost presence complement of [`Self::baseline`], exactly as
    /// [`Self::current_is_some`] is of [`Self::current`]. [`Self::baseline`] yields `Some` only for
    /// a settled `Snapshot`, so this matches that arm directly.
    ///
    /// A loss-window `Witness` and a not-yet-settled anchor both yield `false`: neither is a
    /// *trustworthy settled baseline*. This is the load-bearing distinction for the burst-fork
    /// question "do I have a settled baseline to debounce against, or must I re-Seed?" —
    /// `current_is_some` answered it only because a settled baseline and a live `current` were once
    /// installed atomically; once they decouple (a Seed grafting `current` while deferring the pin)
    /// the fork must read presence of the *baseline*, not of `current`.
    #[must_use]
    pub const fn baseline_is_some(&self) -> bool {
        matches!(
            &self.anchor,
            AnchorClassification::File {
                settled: SettledState::Snapshot(_),
                ..
            } | AnchorClassification::Dir {
                settled: SettledState::Snapshot(_),
                ..
            }
        )
    }

    /// Mutable descent payload — thin delegator to [`ProfileState::descent_state_mut`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        self.state.descent_state_mut()
    }

    /// Disarm this Profile's in-flight probe and return its prior correlation — thin delegator to
    /// [`ProfileState::take_probe`], joining the in-place state-mutator family beside
    /// [`Self::descent_state_mut`] / [`Self::mark_active_for_reap`].
    #[must_use]
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        self.state.take_probe()
    }

    /// Flip an Active burst's directive to `Reap`. `true` iff the flip landed (Active). Delegates
    /// to [`ProfileState::mark_active_for_reap`].
    #[must_use]
    pub const fn mark_active_for_reap(&mut self) -> bool {
        self.state.mark_active_for_reap()
    }

    /// Revive a zombie burst (`Reap → ReturnToIdle`). `true` iff a zombie was revived. Delegates to
    /// [`ProfileState::clear_active_reap`].
    #[must_use]
    pub const fn clear_active_reap(&mut self) -> bool {
        self.state.clear_active_reap()
    }

    /// The single in-life mutator of [`PostFirePhase::Awaiting`]'s `outstanding` — pure delegation
    /// through the state machine, the no-public-setter seam shared with [`Self::clear_active_reap`].
    /// The floor is enforced by the owner, [`PostFireBurst::note_effect_completion`].
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        self.state.note_effect_completion()
    }

    /// The single in-life mutator of the pre-fire and post-fire `last_certified_hash` carriers —
    /// cat-(b) cascade entry, pure delegation through the state machine. The dispatch in
    /// [`ActiveBurst::advance_certified_sample`] routes to whichever burst variant is live;
    /// non-Active states fold to `None`.
    ///
    /// **Authoritative-only contract** sits at the caller (the verdict choke in
    /// `certify_probe_response`): only call after extracting a `ProofAuthority::Authoritative`
    /// response hash. The returned prior is the [`QuiescenceWitness::HashChannel`] `prior` input.
    /// Callable regardless of the verdict outcome ([`QuiescenceVerdict::Stable`] or
    /// [`QuiescenceVerdict::Retry`]) — the carrier tracks the last walker-certified sample, not the
    /// last fire-eligible one.
    #[must_use]
    pub const fn advance_certified_sample(&mut self, hash: u128) -> Option<u128> {
        self.state.advance_certified_sample(hash)
    }

    /// Arm (or re-arm) the operator `absorb` window **and** retro-latch any in-flight pre-fire
    /// burst — one operation for one operator event. Arming while a pre-fire burst is already
    /// batching the replication's events must fold that burst too (the reverse race: events arrive
    /// before the signal), so the set and the retro-latch are inseparable. The latch delegates
    /// through [`ProfileState::latch_fold`] — a no-op unless `Active(PreFire)`, since `Idle` /
    /// `Pending` / post-fire have no pre-fire consequence to override; the window still stands for
    /// the next burst's birth consult.
    ///
    /// **Last-writer-wins, idempotent latch.** A re-arm overwrites the window wholesale and re-drives
    /// the set-only latch; a burst born under the prior window stays latched and folds per the
    /// *current* window's mode at fold time ([`Self::note_absorb_fold`] reads the live mode).
    pub const fn arm_absorb(&mut self, expiry: Instant, mode: AbsorbMode) {
        self.absorb = Some(AbsorbWindow { expiry, mode });
        self.state.latch_fold();
    }

    /// Record one absorbed fold: bump the per-Profile count and, for a live
    /// [`AbsorbMode::ConsumeOnFirst`] window, retire it. One operation for one fold event — the
    /// consolidation mirroring `SubRegistry::record_fired`'s bump-and-stamp.
    ///
    /// The count bumps **unconditionally**: the fold happened, even if the window already went
    /// inert by time between the burst's birth consult and this fold (the latch, frozen at birth,
    /// still folds). The consume guards on a live `ConsumeOnFirst` window and reads the **current**
    /// mode, so a burst born under an old mode retires per the operator's latest intent.
    /// Saturating, to bound at `u64::MAX` rather than wrap.
    pub const fn note_absorb_fold(&mut self) {
        self.absorb_count = self.absorb_count.saturating_add(1);
        if matches!(
            self.absorb,
            Some(AbsorbWindow {
                mode: AbsorbMode::ConsumeOnFirst,
                ..
            })
        ) {
            self.absorb = None;
        }
    }

    /// Borrow the armed window **iff it is live** at `now` (`now < expiry`) — the lone owner of the
    /// liveness predicate. Both the burst-birth consult ([`Self::absorb_window_live`]) and the
    /// `show` projection derive from this, so `now < expiry` is written in exactly one place, not
    /// re-implemented at the projection site across the crate boundary. An expired window reads
    /// `None` without being cleared — lazy expiry, no `&mut`, so the read composes inside the
    /// shared immutable birth borrow.
    #[must_use]
    pub fn absorb_window_if_live(&self, now: Instant) -> Option<&AbsorbWindow> {
        self.absorb.as_ref().filter(|w| now < w.expiry)
    }

    /// The burst-birth consult: `true` iff a window is live at `now` — the boolean projection of
    /// [`Self::absorb_window_if_live`]. The single read that freezes `PreFireBurst::fold_latched`
    /// at construction.
    #[must_use]
    pub fn absorb_window_live(&self, now: Instant) -> bool {
        self.absorb_window_if_live(now).is_some()
    }

    /// Borrow the armed window **without** live-gating — the raw, lossless accessor for tests and
    /// inspection that must tell an inert-but-uncleared window apart from no window. Production
    /// projection live-gates through [`Self::absorb_window_if_live`]; this exposes an expired
    /// window too. `None` iff no window is currently armed.
    #[must_use]
    pub const fn absorb_window(&self) -> Option<&AbsorbWindow> {
        self.absorb.as_ref()
    }

    /// Count of folds this Profile has absorbed — projected per-Sub at the `show` boundary
    /// alongside the per-Sub fire counters.
    #[must_use]
    pub const fn absorb_count(&self) -> u64 {
        self.absorb_count
    }

    /// Take the live `current` snapshot, leaving the arm's `current` `None` and `settled` untouched
    /// — the covered-descendant claim-release primitive. The returned `Dir` snapshot's entries
    /// *are* the descendant membership set the caller (`Engine::release_descendant_claim`) walks
    /// via wholesale deletion. Idempotent (a second call finds `None`); `File` has no descendants
    /// and `Unclassified` has no snapshot, both short-circuit to `None`. Not subsumed by
    /// [`Self::clear_anchor_classification`]: it runs first and is also called standalone from the
    /// `dispatch_*_vanished/failed` + `reap_profile` sites.
    pub fn take_current(&mut self) -> Option<TreeSnapshot> {
        match &mut self.anchor {
            AnchorClassification::Dir { current, .. } => current.take().map(TreeSnapshot::Dir),
            AnchorClassification::File { current, .. } => current.take().map(TreeSnapshot::File),
            AnchorClassification::Unclassified { .. } => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ProfileMap {
    profiles: SlotMap<ProfileId, Profile>,
    by_resource: SecondaryMap<ResourceId, SmallVec<[(u64, ProfileId); 1]>>,
    /// Live count of Profiles satisfying [`Profile::is_nonsteady`] — the O(1) carrier gate the
    /// engine reads before the O(P) `classify_event_carriers` scan. `is_nonsteady` is a pure
    /// `state` predicate (`Pending ∨ Parked`), so it is maintained at exactly the `state`
    /// chokepoints: [`Self::attach`] / [`Self::detach`] (the membership edges),
    /// [`Self::transition_state`] and [`Self::map_state`] (the push and transform `state` writers),
    /// and [`Self::materialize_anchor`] (the wrapper over [`Profile::materialize_anchor`]'s
    /// `transition_state` bypass — a true `Pending → Idle` counting edge). A debug full-scan
    /// tripwire in `Engine::classify_event_carriers` is the desync net: a missed `+` (under-count)
    /// would false-skip a real carrier and is caught there; a missed `−` (over-count) only degrades
    /// the gate to the status-quo scan (perf, never correctness).
    nonsteady: usize,
}

impl ProfileMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up an existing Profile by `(resource, config_hash)`. Returns `None` if no Profile at
    /// this resource matches the hash.
    #[must_use]
    pub fn find(&self, resource: ResourceId, config_hash: u64) -> Option<ProfileId> {
        self.by_resource
            .get(resource)?
            .iter()
            .find(|(h, _)| *h == config_hash)
            .map(|(_, id)| *id)
    }

    /// Insert a fresh Profile and write back-references on both the Tree (`Resource.profiles`) and
    /// the `ProfileMap` (`by_resource`). Caller has verified `find` returns `None` for
    /// `(profile.resource, profile.config_hash)`; a debug-build assertion guards against repeat.
    ///
    /// Panics if `profile.resource` is stale (no live Tree slot). The Engine must construct the
    /// Resource before attaching a Profile to it.
    pub fn attach(&mut self, tree: &mut Tree, profile: Profile) -> ProfileId {
        let resource = profile.resource;
        let hash = profile.config_hash();
        // Derived from the actual birth state, not assumed: a fresh `Profile::new` is `Idle` with
        // the anchor absent (nonsteady), but reading the predicate keeps `nonsteady` exact even if
        // a future construction path births a different state.
        let born_nonsteady = profile.is_nonsteady();
        debug_assert!(
            self.find(resource, hash).is_none(),
            "ProfileMap::attach called twice for the same (resource, config_hash) — caller must `find` first",
        );
        let id = self.profiles.insert(profile);
        if born_nonsteady {
            self.nonsteady += 1;
        }
        // SecondaryMap::entry returns None only if the key has been removed from a primary-tracked
        // SlotMap with a generation that no longer matches. For a freshly-minted ResourceId, we
        // expect `Some`.
        self.by_resource
            .entry(resource)
            .expect("ProfileMap::attach: resource is stale (slot was reaped)")
            .or_default()
            .push((hash, id));
        tree.get_mut(resource)
            .expect("ProfileMap::attach: resource has no live Tree slot")
            .insert_profile_anchor(hash, id);
        id
    }

    /// Remove a Profile and clear back-references on both indices. The caller is responsible for
    /// any subsequent `tree.try_reap(resource)` once it confirms no other anchors remain.
    pub fn detach(&mut self, tree: &mut Tree, id: ProfileId) -> Option<Profile> {
        let p = self.profiles.remove(id)?;
        if p.is_nonsteady() {
            self.nonsteady = self.nonsteady.saturating_sub(1);
        }
        if let Some(v) = self.by_resource.get_mut(p.resource) {
            v.retain(|(h, pid)| !(*pid == id && *h == p.config_hash()));
        }
        if let Some(r) = tree.get_mut(p.resource) {
            r.remove_profile_anchor(p.config_hash(), id);
        }
        Some(p)
    }

    /// Live carrier-eligibility count — the O(1) value `Engine::classify_event_carriers` consults
    /// before its O(P) scan. `0` ⟺ no Profile can carry an `FsEvent` dispatch (every Profile is in
    /// a steady `Active` burst or a healthy anchored `Idle`), so the scan is provably empty and
    /// skipped.
    #[must_use]
    pub const fn nonsteady(&self) -> usize {
        self.nonsteady
    }

    /// Apply one [`Profile::is_nonsteady`] edge to [`Self::nonsteady`] — the single source of the
    /// carrier-count arithmetic, shared by the three `state`-edge reconcile paths:
    /// [`Self::transition_state`], [`Self::map_state`], and [`Self::materialize_anchor`].
    /// Saturating on the `−` side so a (debug-tripwired) missed `+` upstream degrades the gate to
    /// the status-quo scan rather than underflowing.
    const fn apply_nonsteady_edge(&mut self, before: bool, after: bool) {
        match (before, after) {
            (false, true) => self.nonsteady += 1,
            (true, false) => self.nonsteady = self.nonsteady.saturating_sub(1),
            (false, false) | (true, true) => {}
        }
    }

    /// The **push**-shape counter-reconciling path for a Profile **state** edge — the sibling of
    /// [`Self::map_state`] (the **transform** dual). Delegates to [`Profile::transition_state`]
    /// (the core `state` chokepoint, installing the given `new`) and reconciles [`Self::nonsteady`]
    /// across the edge from [`Profile::is_nonsteady`] read before and after the swap.
    ///
    /// Reached by the install-a-given-state callers — `start_seed_burst` / `start_standard_burst` /
    /// descent materialisation / the claims-ledger Idle reset — which discard the returned prior
    /// (relying only on its drop; the claims reset drops a disarmed `Pending` descent this way).
    /// Returns `None` for a stale id, which the callers branch on via `?` or simply discard. A
    /// missed reconcile is perf-only — the debug full-scan tripwire in
    /// `Engine::classify_event_carriers` surfaces a desync in CI.
    pub fn transition_state(&mut self, id: ProfileId, new: ProfileState) -> Option<ProfileState> {
        let p = self.profiles.get_mut(id)?;
        let before = p.is_nonsteady();
        let prior = p.transition_state(new);
        let after = p.is_nonsteady();
        self.apply_nonsteady_edge(before, after);
        Some(prior)
    }

    /// The counter-reconciling path for a **transform** state edge — the [`Self::transition_state`]
    /// sibling that delegates to [`Profile::map_state`] (consume the prior, install `f`'s result)
    /// rather than [`Profile::transition_state`] (install a given `new`). Reconciles
    /// [`Self::nonsteady`] across the edge identically: read [`Profile::is_nonsteady`] before and
    /// after the swap, apply the one edge via `Self::apply_nonsteady_edge`. The auxiliary `R` that
    /// `f` computed from the prior is threaded back out.
    ///
    /// Returns `None` for a stale id — the same `?` short-circuit `transition_state` offers, which
    /// the typed-move callers branch on. A missed reconcile is perf-only, caught by the debug
    /// full-scan tripwire in `Engine::classify_event_carriers`.
    pub fn map_state<R>(
        &mut self,
        id: ProfileId,
        f: impl FnOnce(ProfileState) -> (ProfileState, R),
    ) -> Option<R> {
        let p = self.profiles.get_mut(id)?;
        let before = p.is_nonsteady();
        let r = p.map_state(f);
        let after = p.is_nonsteady();
        self.apply_nonsteady_edge(before, after);
        Some(r)
    }

    /// The counter-reconciling wrapper over [`Profile::materialize_anchor`]'s three-field atomic
    /// `Pending → (Idle, Held, classified)` write — the chokepoint sibling of
    /// [`Self::transition_state`] / [`Self::map_state`] for the one documented `state` write that
    /// bypasses [`Profile::transition_state`]. Under the pure-state [`Profile::is_nonsteady`]
    /// predicate that bypass is a genuine `true → false` counting edge (`Pending` counted, `Idle`
    /// not), so it must reconcile here or the engine's debug full-recount tripwire fires on the
    /// next delivered event.
    ///
    /// Silently no-ops on a stale id — the same policy as `transition_state`'s `None` return; the
    /// sole caller (`Engine::materialize_profile_anchor`, the descent terminus) operates on an
    /// owner the probe gate just proved live.
    pub fn materialize_anchor(&mut self, id: ProfileId, kind: ResourceKind) {
        let Some(p) = self.profiles.get_mut(id) else {
            return;
        };
        let before = p.is_nonsteady();
        p.materialize_anchor(kind);
        let after = p.is_nonsteady();
        self.apply_nonsteady_edge(before, after);
    }

    #[must_use]
    pub fn get(&self, id: ProfileId) -> Option<&Profile> {
        self.profiles.get(id)
    }

    pub fn get_mut(&mut self, id: ProfileId) -> Option<&mut Profile> {
        self.profiles.get_mut(id)
    }

    /// Iterator over the Profiles attached at `resource`, in `Resource.profiles` insertion order.
    pub fn at(&self, resource: ResourceId) -> impl Iterator<Item = ProfileId> + '_ {
        self.by_resource
            .get(resource)
            .into_iter()
            .flatten()
            .map(|(_, id)| *id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (ProfileId, &Profile)> {
        self.profiles.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ProfileId, &mut Profile)> {
        self.profiles.iter_mut()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Count of Profiles doing something right now (`Pending ∨ Active`) — the operator-facing "in
    /// flight" count distinct from [`Self::len`] (the total). Disjoint from [`Self::nonsteady`]:
    /// `nonsteady` is the engine's carrier-eligibility predicate (`Pending ∨ Parked`), whereas this
    /// projection is the operator's activity predicate. A healthy `Idle` and a `Parked` Profile are
    /// both at rest and uncounted here (a park is a stuck state awaiting recovery, not in-flight
    /// work — `StateLabel::Parked` is its operator surface); a `Pending` is counted by both; an
    /// `Active` burst is counted here and excluded from `nonsteady` (the burst itself is the
    /// dispatch authority — no recovery channel is needed).
    ///
    /// O(N) over [`Self::iter`]. Acceptable for v1: `specter status` is operator-paced (single
    /// request, human latency tolerance). A future cached counter belongs on [`ProfileMap`] itself,
    /// not on the projection helper that consumes it.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.iter()
            .filter(|(_, p)| {
                matches!(
                    p.state(),
                    ProfileState::Pending(_) | ProfileState::Active(_, _)
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AnchorClassification, ClassSet, DescentRemaining, DescentState, Profile, ProfileIdentity,
        ProfileMap, ProfileState, ScanConfig, SettledState,
    };
    use crate::fs_id::FsIdentity;
    use crate::ids::{ProfileId, ResourceId};
    use crate::output::StepOutput;
    use crate::pattern::PatternSpec;
    use crate::probe::ProbeSlot;
    use crate::resource::{ResourceKind, ResourceRole};
    use crate::scan_config::GlobPattern;
    use crate::scan_config::compute_config_hash;
    use crate::snapshot::EntryKind;
    use crate::snapshot::tree::{DirMeta, DirSnapshot, LeafEntry, TreeSnapshot};
    use crate::tree::Tree;
    use compact_str::CompactString;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    fn cfg() -> ScanConfig {
        ScanConfig::builder().build()
    }

    fn glob(source: &str) -> GlobPattern {
        GlobPattern::compile(source).expect("test glob compiles")
    }

    /// Test constructor preserving the pre-decomposition 6-arg call shape: folds `(config,
    /// max_settle, events)` into the [`ProfileIdentity`] the real [`Profile::new`] now takes by
    /// value, so every fixture's exact parameters survive the decomposition unchanged.
    fn mk_profile(
        resource: ResourceId,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        events: ClassSet,
        kind: Option<ResourceKind>,
    ) -> Profile {
        Profile::new(
            resource,
            ProfileIdentity::new(config, max_settle, events),
            settle,
            kind,
        )
    }

    #[test]
    fn new_profile_starts_idle() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(matches!(p.state(), ProfileState::Idle));
        assert!(p.baseline().is_none());
        assert!(!p.current_is_some());
        assert_eq!(p.max_settle(), MAX_SETTLE);
        assert_eq!(p.settle, SETTLE);
        // Absorb state initialises empty: no window armed, zero folds.
        assert!(
            p.absorb_window().is_none(),
            "fresh Profile has no absorb window"
        );
        assert_eq!(p.absorb_count(), 0, "fresh Profile has folded nothing");
    }

    /// `Profile::new` debug-asserts `settle <= max_settle`. The burst lifecycle needs the settle
    /// (quiet-window) timer to expire before the burst deadline; otherwise every burst force-fires
    /// without ever reaching a stable verdict. The config layer's `validate_settle` is the real
    /// boundary (it enforces `max_settle >= 4 × settle`), so reaching construction with `settle >
    /// max_settle` means a caller bypassed config validation — the constructor trips loudly in
    /// debug rather than silently shipping a Profile that forces every burst.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "must not exceed max_settle")]
    fn profile_new_panics_when_settle_exceeds_max_settle() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        // max_settle = 5s, settle = 10s — the rejected-but-representable combination the config
        // layer should have caught upstream.
        let _ = mk_profile(
            r,
            cfg(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            NO_EVENTS,
            None,
        );
    }

    /// `has_per_file_fds` defaults to false when `events` excludes both CONTENT and METADATA. The
    /// flag is invariant for the Profile's lifetime — set once at construction from the events mask.
    #[test]
    fn new_profile_initialises_has_per_file_fds_false_for_empty_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!p.has_per_file_fds());
        assert_eq!(p.events(), ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (the `subtree-root` default), so
    /// in-place edits surface as events through per-file FDs instead of waiting for a probe.
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        assert!(p.has_per_file_fds());
        assert_eq!(p.events(), ClassSet::CONTENT);
    }

    /// `has_per_file_fds` is also true when METADATA is in the mask (a metadata-only watch needs
    /// per-file FDs for chmod / nlink signals).
    #[test]
    fn new_profile_has_per_file_fds_when_metadata_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert!(p.has_per_file_fds());
    }

    /// STRUCTURE-only watch does not flip `has_per_file_fds` — directory entries are observed at
    /// the parent dir's FD, not at per-file FDs.
    #[test]
    fn new_profile_has_per_file_fds_false_for_structure_only() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE, None);
        assert!(!p.has_per_file_fds());
    }

    /// For the `Subtree` shape, [`Profile::events_witness_quiescence`] is true iff the mask covers
    /// [`ClassSet::IN_PLACE_WRITES`]. Masks lacking the in-place-writes vocabulary cannot witness
    /// an in-place write over a settle window, so settle-window silence does not prove quiescence
    /// on those masks. The predicate is the (per-Profile) gate on whether the verdict floor's
    /// settle-natural fire path is sound; events-incomplete Profiles need the hash-equality
    /// channel. (The shape dispatch — a `MatchChain` Profile under the identical masks — is pinned
    /// by [`events_witness_quiescence_dispatches_on_scan_shape`].)
    #[test]
    fn events_witness_quiescence_tracks_in_place_writes_mask() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        let empty = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::EMPTY, None);
        assert!(
            !empty.events_witness_quiescence(),
            "an empty events mask catches nothing — silence proves nothing",
        );

        let structure = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE, None);
        assert!(
            !structure.events_witness_quiescence(),
            "STRUCTURE alone misses IN_PLACE_WRITES (the scp regression)",
        );

        let metadata = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert!(
            !metadata.events_witness_quiescence(),
            "METADATA alone drops IN_PLACE_WRITES at the per-Profile class filter",
        );

        let in_place = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            ClassSet::IN_PLACE_WRITES,
            None,
        );
        assert!(
            in_place.events_witness_quiescence(),
            "IN_PLACE_WRITES subscribes to in-place writes — settle-silence proves quiescence",
        );

        let structure_and_in_place = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            ClassSet::STRUCTURE | ClassSet::IN_PLACE_WRITES,
            None,
        );
        assert!(structure_and_in_place.events_witness_quiescence());

        let structure_and_metadata = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            ClassSet::STRUCTURE | ClassSet::METADATA,
            None,
        );
        assert!(
            !structure_and_metadata.events_witness_quiescence(),
            "the predicate is the IN_PLACE_WRITES mask: without it, no settle-natural fire",
        );
    }

    /// The witness criterion is shape-owned: a `MatchChain` Profile's proof object is the match
    /// set, whose changes are all STRUCTURE point events — so the STRUCTURE-only mask that fails
    /// the `Subtree` witness (the sibling test above) suffices here, folding discovery bursts via
    /// `EventsReliable` (N=1) instead of the two-sample hash channel. A chain mask *without*
    /// STRUCTURE falls back to the hash channel — conservative composition, no panic shape, no
    /// attach-time validation needed at this layer.
    #[test]
    fn events_witness_quiescence_dispatches_on_scan_shape() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let chain = || {
            ScanConfig::MatchChain(Arc::new(
                PatternSpec::parse("/srv/*/log").expect("test pattern parses"),
            ))
        };

        let chain_structure = mk_profile(r, chain(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE, None);
        assert!(
            chain_structure.events_witness_quiescence(),
            "STRUCTURE covers MEMBERSHIP_CHANGES — settle-silence witnesses the match set",
        );

        let chain_content = mk_profile(r, chain(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        assert!(
            !chain_content.events_witness_quiescence(),
            "a chain mask missing STRUCTURE cannot witness membership — hash-channel fallback",
        );
    }

    #[test]
    fn config_hash_matches_compute_config_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let c = cfg();
        let expected = compute_config_hash(&c, MAX_SETTLE, NO_EVENTS);
        let p = mk_profile(r, c, MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(p.config_hash(), expected);
    }

    /// Different `events` mask produces different `config_hash` (partition-by-mask).
    #[test]
    fn config_hash_partitions_by_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p_content = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        let p_meta = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert_ne!(p_content.config_hash(), p_meta.config_hash());
    }

    #[test]
    fn attach_writes_both_indices() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let h = p.config_hash();
        let pid = profiles.attach(&mut tree, p);

        assert_eq!(profiles.find(r, h), Some(pid));
        assert_eq!(tree.get(r).unwrap().profiles(), &[(h, pid)]);
    }

    #[test]
    fn attach_anchors_resource_against_reap() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );

        assert!(
            !tree.try_reap(r, &mut StepOutput::default()),
            "Profile-anchored resource must not reap",
        );
    }

    #[test]
    fn detach_clears_back_references() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let h = p.config_hash();
        let pid = profiles.attach(&mut tree, p);

        let detached = profiles.detach(&mut tree, pid);
        assert!(detached.is_some(), "detach yields the removed Profile");
        assert!(profiles.find(r, h).is_none());
        assert!(tree.get(r).unwrap().profiles().is_empty());
    }

    #[test]
    fn detach_then_reap_succeeds_when_no_other_anchors() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pid = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );

        profiles.detach(&mut tree, pid);
        assert!(tree.try_reap(r, &mut StepOutput::default()));
        assert!(tree.get(r).is_none());
    }

    #[test]
    fn at_iterates_profiles_attached_at_resource() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        let pid_a = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS, None),
        );
        // Different max_settle ⇒ different config_hash ⇒ distinct Profile.
        let pid_b = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS, None),
        );

        let mut got: Vec<_> = profiles.at(r).collect();
        got.sort();
        let mut expected = vec![pid_a, pid_b];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn distinct_resources_get_distinct_profiles() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r1 = tree.ensure_root("a", ResourceRole::User);
        let r2 = tree.ensure_root("b", ResourceRole::User);

        let p1 = profiles.attach(
            &mut tree,
            mk_profile(r1, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        let p2 = profiles.attach(
            &mut tree,
            mk_profile(r2, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        assert_ne!(p1, p2);
        assert_eq!(profiles.len(), 2);
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "called twice")]
    fn attach_duplicate_panics_in_debug() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("x", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        // Caller failed to `find` first; second attach hits debug_assert.
        let _pid2 = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
    }

    /// [`ProfileMap::active_count`] reports the count of Profiles **not** in [`ProfileState::Idle`]
    /// — the operator-facing "in flight" tally surfaced via `specter status`. Empty map, fresh
    /// attach, and round-trip Idle → Pending → Idle each pin one row of the counting contract;
    /// other non-Idle variants (Active) share the same `!matches!(_, Idle)` predicate, so pinning a
    /// single non-Idle variant is sufficient.
    #[test]
    fn active_count_counts_only_non_idle_profiles() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        assert_eq!(profiles.active_count(), 0, "empty ⇒ 0");

        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pid = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        // Fresh Profile is `Idle` ⇒ uncounted. Distinguishes the counter from `len()` (which would
        // report 1 here).
        assert_eq!(
            profiles.active_count(),
            0,
            "Idle Profile is excluded from active_count",
        );
        assert_eq!(profiles.len(), 1, "len() counts every Profile");

        let pending = ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
            false,
        ));
        let _prior = profiles
            .transition_state(pid, pending)
            .expect("known id transitions");
        assert_eq!(
            profiles.active_count(),
            1,
            "Pending Profile is counted by active_count",
        );

        let _prior = profiles
            .transition_state(pid, ProfileState::Idle)
            .expect("known id transitions");
        assert_eq!(
            profiles.active_count(),
            0,
            "transitioning back to Idle excludes the Profile again",
        );
    }

    // -----------------------------------------------------------------------
    // rebase_baseline / capture_witness_at_loss
    // -----------------------------------------------------------------------

    fn empty_dir_snapshot() -> Arc<DirSnapshot> {
        Arc::new(DirSnapshot::new(
            DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
            0,
            BTreeMap::new(),
        ))
    }

    fn empty_leaf_entry() -> LeafEntry {
        LeafEntry::synthetic(EntryKind::File, 0, UNIX_EPOCH, FsIdentity::synthetic(0, 0))
    }

    #[test]
    fn rebase_baseline_settles_current_as_baseline() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());
        assert!(p.baseline().is_none(), "no baseline pre-rebase");
        let current_hash = p.current().expect("current set").hash();

        p.rebase_baseline();

        assert_eq!(
            p.baseline().expect("baseline settled").hash(),
            current_hash,
            "baseline matches the rebased current",
        );
    }

    #[test]
    fn rebase_baseline_consumes_survival_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        // Recovery shape: a classified Dir carrying a live current and a survival witness (baseline
        // cleared at the prior loss).
        let snap = empty_dir_snapshot();
        p.anchor = AnchorClassification::Dir {
            current: Some(Arc::clone(&snap)),
            settled: SettledState::Witness(0xdead_beef),
        };
        assert_eq!(p.settled_hash(), Some(0xdead_beef), "witness pre-rebase");

        p.rebase_baseline();

        assert_eq!(
            p.settled_hash(),
            Some(TreeSnapshot::Dir(snap).hash()),
            "rebase replaces the witness with the settled current hash",
        );
        assert!(p.baseline().is_some(), "active mode after rebase");
    }

    // -----------------------------------------------------------------------
    // install_dir_current / install_file_current — classifying graft
    // -----------------------------------------------------------------------

    #[test]
    fn install_dir_current_classifies_and_sets_current() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(p.kind(), None, "fresh Profile is unclassified");
        assert!(!p.current_is_some());

        p.install_dir_current(empty_dir_snapshot());

        assert_eq!(
            p.kind(),
            Some(crate::resource::ResourceKind::Dir),
            "kind is the sum discriminant after the graft",
        );
        assert!(p.current_dir().is_some(), "Dir current borrowable");
        assert!(matches!(p.current(), Some(TreeSnapshot::Dir(_))));
    }

    #[test]
    fn install_file_current_classifies_and_sets_current() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("file", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.install_file_current(empty_leaf_entry());

        assert_eq!(p.kind(), Some(crate::resource::ResourceKind::File));
        assert!(matches!(p.current(), Some(TreeSnapshot::File(_))));
        assert!(p.current_dir().is_none(), "File has no Dir borrow");
    }

    /// Re-grafting a Dir current on a Dir-classified Profile keeps the discriminant and leaves
    /// `settled` untouched (a within-epoch re-graft, fresh or mid-recovery).
    #[test]
    fn install_dir_current_reinstall_preserves_settled() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());
        p.rebase_baseline();
        let settled = p.settled_hash();

        // Second graft with a fresh (equal-content) snapshot.
        p.install_dir_current(empty_dir_snapshot());

        assert_eq!(p.kind(), Some(crate::resource::ResourceKind::Dir));
        assert_eq!(
            p.settled_hash(),
            settled,
            "re-graft leaves the settled baseline untouched",
        );
    }

    /// Grafting onto an `Unclassified` anchor that carries a survival witness (the post-loss
    /// recovery shape) classifies it *and* carries the witness forward into `settled`, so the
    /// post-recovery drift verdict still has a reference.
    #[test]
    fn install_dir_current_carries_witness_from_unclassified() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.anchor = AnchorClassification::Unclassified {
            witness: Some(0x00c0_ffee),
        };

        p.install_dir_current(empty_dir_snapshot());

        assert_eq!(p.kind(), Some(crate::resource::ResourceKind::Dir));
        assert!(p.current_is_some());
        assert_eq!(
            p.settled_hash(),
            Some(0x00c0_ffee),
            "witness carried forward as Witness(settled)",
        );
        assert!(
            p.baseline().is_none(),
            "a carried witness is not an active baseline",
        );
    }

    /// Cross-arm misuse: grafting a `Dir` onto a `File`-classified Profile panics in debug builds.
    /// Production paths never reach this branch — the certifier's inline kind guard catches the
    /// routing breach at the verdict floor first.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "install_dir_current: kind mismatch")]
    fn install_dir_current_panics_on_file_kinded_profile_in_debug() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_file_current(empty_leaf_entry());
        // Boundary-bypass: a future caller skips the certifier's inline kind guard; the graft's
        // debug_assert fires.
        p.install_dir_current(empty_dir_snapshot());
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "install_file_current: kind mismatch")]
    fn install_file_current_panics_on_dir_kinded_profile_in_debug() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());
        p.install_file_current(empty_leaf_entry());
    }

    // -----------------------------------------------------------------------
    // exclude_strings projection
    // -----------------------------------------------------------------------

    /// `Profile.exclude_strings` mirrors `ScanConfig.exclude` in source-string form, sorted
    /// lexicographically. The builder sorts at `build()`, so the projection inherits the canonical
    /// order regardless of insertion order.
    #[test]
    fn profile_new_projects_exclude_strings_in_canonical_order() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let cfg = ScanConfig::builder()
            .exclude(glob("z"))
            .exclude(glob("a"))
            .exclude(glob("m"))
            .build();

        let p = mk_profile(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS, None);

        let actual: Vec<&str> = p
            .exclude_strings()
            .iter()
            .map(CompactString::as_str)
            .collect();
        assert_eq!(actual, vec!["a", "m", "z"]);
    }

    /// `Profile.exclude_strings` is empty (zero-length slice) when the `ScanConfig` has no excludes
    /// — pin so consumers can rely on the projection always being populated.
    #[test]
    fn profile_new_exclude_strings_empty_for_no_excludes() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(p.exclude_strings().is_empty());
    }

    /// The Arc on `Profile.exclude_strings` is the substitution-side handle shared across every Sub
    /// joined to this Profile. Two clones of the field point at the same allocation; the
    /// `bytes-per-Arc` cost is constant regardless of Sub fanout.
    #[test]
    fn profile_exclude_strings_arc_shared_across_siblings() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let cfg = ScanConfig::builder()
            .exclude(glob("*.tmp"))
            .exclude(glob("*.bak"))
            .build();

        let p = mk_profile(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS, None);

        let initial = Arc::strong_count(p.exclude_strings());
        let sibling_a = Arc::clone(p.exclude_strings());
        let sibling_b = Arc::clone(p.exclude_strings());

        assert!(
            Arc::ptr_eq(&sibling_a, &sibling_b),
            "siblings reading exclude_strings observe one allocation",
        );
        assert_eq!(
            Arc::strong_count(p.exclude_strings()),
            initial + 2,
            "each sibling clone bumps the strong count",
        );
    }

    // -----------------------------------------------------------------------
    // ProfileState projections: timer_token / is_draining / descent_state
    // -----------------------------------------------------------------------

    use super::{
        AbsorbMode, AbsorbWindow, ActiveBurst, AwaitVerdict, BurstFinish, BurstIntent,
        DirtyProvenance, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase,
        QuiescenceVerdict, QuiescenceWitness, StableReason, TimerKind, quiescence_verdict,
    };
    use crate::ids::{ProbeCorrelation, TimerId};
    use crate::op::ProofAuthority;
    use std::collections::BTreeSet;
    use std::num::NonZeroU32;
    use std::path::Path;

    fn tid(n: u64) -> TimerId {
        TimerId::from(n)
    }

    /// Inline twin of `testkit::dirty_provenance` — the `testkit` feature is off for `cargo nextest
    /// run -p specter-core`, so the canonical fixture is unreachable from this module's build.
    /// Mirrors the production ingest contract exactly: each pair is one [`DirtyProvenance::note`]
    /// in slice order (a repeated `ResourceId` is last-writer-wins), paths must be absolute (the
    /// component-LCA relies on every value sharing the root) — a relative path is a fixture bug,
    /// caught loudly in dev/CI and inert in release.
    fn dirty_prov(entries: &[(ResourceId, &str)]) -> DirtyProvenance {
        let mut dirty = DirtyProvenance::new();
        for &(id, path) in entries {
            debug_assert!(
                path.starts_with('/'),
                "dirty_prov: '{path}' must be an absolute path",
            );
            dirty.note(id, Arc::from(Path::new(path)));
        }
        dirty
    }

    /// The captured-path set a [`DirtyProvenance`] built from `entries` projects to
    /// (`DirtyProvenance::chains`) — the observable the migrated residual / mutation tests assert
    /// against now that the provenance has no `PartialEq` and no field peek.
    fn expected_chains(entries: &[&str]) -> BTreeSet<Arc<Path>> {
        entries.iter().map(|p| Arc::from(Path::new(*p))).collect()
    }

    /// `n` distinct `ResourceId`s from a throwaway slotmap — core has no `Tree`, and these tests
    /// only need the keys to differ.
    fn rids(n: usize) -> Vec<ResourceId> {
        let mut sm = slotmap::SlotMap::<ResourceId, ()>::with_key();
        (0..n).map(|_| sm.insert(())).collect()
    }

    // `DirtyProvenance`'s own contract. The engine's `pre_fire_target` / emission tests exercise it
    // transitively; these pin the type directly — above all the component-wise LCA the path-LCA
    // scope rests on.

    #[test]
    fn dirty_provenance_lca_path_is_component_wise_not_byte_prefix() {
        // `/w/a` is NOT a prefix of `/w/ab`: a byte-prefix LCA would wrongly root the probe at
        // `/w/a` and clip `/w/ab`. Component-wise, the only shared ancestor is `/w`.
        let r = rids(2);
        let dp = dirty_prov(&[(r[0], "/w/a"), (r[1], "/w/ab")]);
        assert_eq!(dp.lca_path().as_deref(), Some(Path::new("/w")));

        // A genuinely divergent pair reduces to its real ancestor.
        let dp = dirty_prov(&[(r[0], "/w/x/a"), (r[1], "/w/y/b")]);
        assert_eq!(dp.lca_path().as_deref(), Some(Path::new("/w")));
    }

    #[test]
    fn dirty_provenance_lca_path_single_entry_is_identity_empty_is_none() {
        // The dominant single-file-edit case: a lone captured path is its own LCA, returned without
        // reallocating (the `Arc::clone(first)` fast path) — pinned via pointer identity.
        let r = rids(1);
        let only: Arc<Path> = Arc::from(Path::new("/w/deep/file.rs"));
        let mut dp = DirtyProvenance::new();
        dp.note(r[0], Arc::clone(&only));
        let lca = dp.lca_path().expect("non-empty");
        assert!(
            Arc::ptr_eq(&lca, &only),
            "a lone path is returned, not reallocated",
        );

        assert!(DirtyProvenance::new().lca_path().is_none(), "empty ⇒ None");
    }

    #[test]
    fn dirty_provenance_notes_key_by_slot_last_writer_wins() {
        // `note` is keyed by slot: a repeat event for one slot overwrites (last-writer-wins, one
        // chain); distinct slots each contribute one chain; `chains()` is exactly the captured set.
        let r = rids(2);
        let mut dp = DirtyProvenance::new();
        assert!(dp.is_empty());
        dp.note(r[0], Arc::from(Path::new("/w/a")));
        dp.note(r[0], Arc::from(Path::new("/w/a2"))); // same slot, later event
        dp.note(r[1], Arc::from(Path::new("/w/b")));
        assert!(!dp.is_empty());
        assert_eq!(dp.chains(), expected_chains(&["/w/a2", "/w/b"]));
        dp.clear();
        assert!(dp.is_empty() && dp.lca_path().is_none());
    }

    #[test]
    fn dirty_provenance_chains_non_empty_iff_dirty_non_empty() {
        // Probe-choke post-condition: `chains()` is empty exactly when `dirty` is empty (and
        // conversely, non-empty when `dirty` is non-empty). The engine's probe choke wraps the
        // result with `NonEmptyChainSet::new(...)`; the `None` arm degrades to `WholeSubtree`. For
        // a Standard burst with a recorded trigger (`!dirty.is_empty()`), the projection must yield
        // a non-empty chain set or the engine would silently re-walk the whole subtree under
        // `WholeSubtree`. Forward-defensive: if a future change to `chains()` (a filter, a guard)
        // could leave it empty while `dirty` still carries values, this test catches it before the
        // probe-choke regression lands.
        let r = rids(1);
        let mut dp = DirtyProvenance::new();
        assert!(dp.chains().is_empty(), "empty dirty ⇒ empty chains");

        dp.note(r[0], Arc::from(Path::new("/w/a")));
        assert!(!dp.is_empty());
        assert!(
            !dp.chains().is_empty(),
            "non-empty dirty must project to a non-empty chain set",
        );

        dp.clear();
        assert!(dp.chains().is_empty(), "cleared dirty ⇒ empty chains");
    }

    fn batching_burst(settle: TimerId, deadline: TimerId) -> PreFireBurst {
        PreFireBurst::new(
            deadline,
            PreFirePhase::Batching {
                settle_timer: settle,
            },
            BurstIntent::Standard,
            DirtyProvenance::new(),
            None,
            false,
        )
    }

    fn unit_pre(phase: PreFirePhase, deadline: TimerId) -> PreFireBurst {
        PreFireBurst::new(
            deadline,
            phase,
            BurstIntent::Standard,
            DirtyProvenance::new(),
            None,
            false,
        )
    }

    /// Settle on Batching returns the carried token.
    #[test]
    fn timer_token_settle_on_batching_returns_settle_timer() {
        let pre = batching_burst(tid(7), tid(99));
        assert_eq!(pre.timer_token(TimerKind::Settle), Some(tid(7)));
    }

    /// BurstDeadline on any pre-fire phase returns the burst's deadline, non-Optional by
    /// construction.
    #[test]
    fn timer_token_burst_deadline_lives_on_every_prefire_phase() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        for phase in [
            PreFirePhase::Batching {
                settle_timer: tid(1),
            },
            PreFirePhase::Verifying {
                slot: ProbeSlot::empty(),
                target: r,
            },
            PreFirePhase::Draining,
        ] {
            let pre = unit_pre(phase, tid(42));
            assert_eq!(pre.timer_token(TimerKind::BurstDeadline), Some(tid(42)));
        }
    }

    /// Settle on non-Batching pre-fire phases returns None — the field is structurally absent.
    #[test]
    fn timer_token_settle_is_none_on_verifying_or_draining() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        for phase in [
            PreFirePhase::Verifying {
                slot: ProbeSlot::empty(),
                target: r,
            },
            PreFirePhase::Draining,
        ] {
            let pre = unit_pre(phase, tid(42));
            assert!(pre.timer_token(TimerKind::Settle).is_none());
        }
    }

    /// AwaitGateDeadline is type-impossible on pre-fire — returns None.
    #[test]
    fn timer_token_await_gate_is_none_on_prefire() {
        let pre = batching_burst(tid(1), tid(2));
        assert!(pre.timer_token(TimerKind::AwaitGateDeadline).is_none());
    }

    /// AwaitGateDeadline on Awaiting returns the carried token.
    #[test]
    fn timer_token_await_gate_on_awaiting_returns_gate_deadline() {
        let post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(55),
            },
            DirtyProvenance::new(),
        );
        assert_eq!(
            post.timer_token(TimerKind::AwaitGateDeadline),
            Some(tid(55)),
        );
    }

    /// AwaitGateDeadline on Rebasing returns None — the field doesn't exist on that variant.
    #[test]
    fn timer_token_await_gate_is_none_on_rebasing() {
        let post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );
        assert!(post.timer_token(TimerKind::AwaitGateDeadline).is_none());
    }

    /// Settle / BurstDeadline are type-impossible on post-fire — None for both phases.
    #[test]
    fn timer_token_settle_and_burst_deadline_are_none_on_postfire() {
        for phase in [
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(99),
            },
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            PostFirePhase::Settling {
                settle_timer: tid(77),
            },
        ] {
            let post = PostFireBurst::new(BurstIntent::Standard, phase, DirtyProvenance::new());
            assert!(post.timer_token(TimerKind::Settle).is_none());
            assert!(post.timer_token(TimerKind::BurstDeadline).is_none());
        }
    }

    /// `PostFireSettle` is the post-fire `Settle`: it lives only on `Settling`'s `settle_timer`,
    /// `None` on the other phases.
    #[test]
    fn timer_token_post_fire_settle_lives_only_on_settling() {
        let settling = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Settling {
                settle_timer: tid(31),
            },
            DirtyProvenance::new(),
        );
        assert_eq!(
            settling.timer_token(TimerKind::PostFireSettle),
            Some(tid(31)),
        );

        for phase in [
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(1),
            },
            PostFirePhase::Rebasing(ProbeSlot::empty()),
        ] {
            let post = PostFireBurst::new(BurstIntent::Standard, phase, DirtyProvenance::new());
            assert!(post.timer_token(TimerKind::PostFireSettle).is_none());
        }
    }

    /// The verdict-fold floor [`quiescence_verdict`] projects three axes — `(authority × forced ×
    /// witness)` — onto three variants (`Stable(Natural | Forced)`, `Retry`, `Abandon`). Total,
    /// pure, side-effect-free.
    ///
    /// Cases covered: all reachable shapes; the `Authoritative + !forced +
    /// HashChannel(prior≠response)` row produces `Retry`, ruling out the would-be `Stable(Natural)`
    /// mistake. `Retry.observed_motion` is `true` only on the concrete-disagreement row — the
    /// first-sample `prior=None` and the walker refusal both observed nothing, so they fold
    /// `false` (the streak holds, it does not inflate). The `Undischarged + !forced` arm drops
    /// `first_unread` at the fold — the transient retry arm at both dispatch sites has no consumer
    /// for it (consumption-aligned: an unused `Arc<Path>` is one `Arc::drop` instead of
    /// clone-then-drop downstream).
    ///
    /// - Authoritative × !forced × EventsReliable          → Stable(Natural)
    /// - Authoritative × !forced × HashChannel(prior=None) → Retry { observed_motion: false }
    /// - Authoritative × !forced × HashChannel(p==r)       → Stable(Natural)
    /// - Authoritative × !forced × HashChannel(p≠r)        → Retry { observed_motion: true }
    /// - Authoritative ×  forced × EventsReliable          → Stable(Forced{disagreed=false})
    /// - Authoritative ×  forced × HashChannel(prior=None) → Stable(Forced{disagreed=false})
    /// - Authoritative ×  forced × HashChannel(p==r)       → Stable(Forced{disagreed=false})
    /// - Authoritative ×  forced × HashChannel(p≠r)        → Stable(Forced{disagreed=true})
    /// - Undischarged   × !forced × *                      → Retry { observed_motion: false } (first_unread dropped at the fold)
    /// - Undischarged   ×  forced × *                      → Abandon { first_unread }
    #[test]
    fn quiescence_verdict_folds_three_axes() {
        let unread: std::sync::Arc<std::path::Path> =
            std::sync::Arc::from(std::path::Path::new("first/unread"));
        let undischarged = || ProofAuthority::Undischarged {
            first_unread: std::sync::Arc::clone(&unread),
        };
        let er = QuiescenceWitness::EventsReliable;
        let first = QuiescenceWitness::HashChannel {
            prior: None,
            response: 1,
        };
        let eq = QuiescenceWitness::HashChannel {
            prior: Some(7),
            response: 7,
        };
        let neq = QuiescenceWitness::HashChannel {
            prior: Some(7),
            response: 8,
        };

        // Authoritative + !forced — witness selects Stable vs Retry; on Retry, only a concrete
        // disagreement reports observed motion.
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, false, er),
            QuiescenceVerdict::Stable(StableReason::Natural),
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, false, first),
            QuiescenceVerdict::Retry {
                observed_motion: false,
            },
            "first-sample hash channel (prior=None) ⇒ Retry, not Natural; \
             absence of confirmation is not observed motion",
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, false, eq),
            QuiescenceVerdict::Stable(StableReason::Natural),
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, false, neq),
            QuiescenceVerdict::Retry {
                observed_motion: true,
            },
        );

        // Authoritative + forced — ceiling bypass. Disagreement bit reads the witness: `true` only
        // on Some(p) != response.
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, true, er),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: false,
            }),
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, true, first),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: false,
            }),
            "first-sample channel on forced fold ⇒ absence of confirmation, not observed disagreement",
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, true, eq),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: false,
            }),
        );
        assert_eq!(
            quiescence_verdict(ProofAuthority::Authoritative, true, neq),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: true,
            }),
        );

        // Undischarged — witness ignored; forced selects Retry vs Abandon. !forced drops first_unread
        // at the fold (transient arm — no consumer); forced carries it verbatim on Abandon. A
        // refusal observed nothing, so it never reports motion.
        assert_eq!(
            quiescence_verdict(undischarged(), false, er),
            QuiescenceVerdict::Retry {
                observed_motion: false,
            },
            "Undischarged + !forced ⇒ Retry (first_unread dropped at the fold; no motion observed)",
        );
        let v = quiescence_verdict(undischarged(), true, neq);
        assert!(
            matches!(&v, QuiescenceVerdict::Abandon { first_unread }
                if first_unread.as_ref() == std::path::Path::new("first/unread")),
            "Undischarged + forced ⇒ Abandon carrying first_unread verbatim; got {v:?}",
        );
    }

    /// ActiveBurst delegates to the held inner type.
    #[test]
    fn active_burst_timer_token_dispatches_by_lifecycle() {
        let pre = ActiveBurst::PreFire(batching_burst(tid(3), tid(4)));
        assert_eq!(pre.timer_token(TimerKind::Settle), Some(tid(3)));
        assert_eq!(pre.timer_token(TimerKind::BurstDeadline), Some(tid(4)));
        assert!(pre.timer_token(TimerKind::AwaitGateDeadline).is_none());

        let post = ActiveBurst::PostFire(PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(9),
            },
            DirtyProvenance::new(),
        ));
        assert_eq!(post.timer_token(TimerKind::AwaitGateDeadline), Some(tid(9)));
        assert!(post.timer_token(TimerKind::Settle).is_none());
        assert!(post.timer_token(TimerKind::BurstDeadline).is_none());
    }

    /// ProfileState::Idle owns no timers.
    #[test]
    fn profile_state_timer_token_idle_returns_none_for_every_kind() {
        let s = ProfileState::Idle;
        for k in [
            TimerKind::Settle,
            TimerKind::BurstDeadline,
            TimerKind::AwaitGateDeadline,
            TimerKind::PostFireSettle,
            TimerKind::RebaseCeiling,
        ] {
            assert!(s.timer_token(k).is_none());
        }
    }

    /// ProfileState::Pending owns no timers (descent uses the probe channel for correlation, not a
    /// heap timer).
    #[test]
    fn profile_state_timer_token_pending_returns_none_for_every_kind() {
        let s = ProfileState::Pending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
            false,
        ));
        for k in [
            TimerKind::Settle,
            TimerKind::BurstDeadline,
            TimerKind::AwaitGateDeadline,
            TimerKind::PostFireSettle,
            TimerKind::RebaseCeiling,
        ] {
            assert!(s.timer_token(k).is_none());
        }
    }

    /// ProfileState::Active delegates to the held ActiveBurst.
    #[test]
    fn profile_state_timer_token_active_delegates_to_burst() {
        let state = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(11), tid(12))),
            BurstFinish::ReturnToIdle,
        );
        assert_eq!(state.timer_token(TimerKind::Settle), Some(tid(11)));
        assert_eq!(state.timer_token(TimerKind::BurstDeadline), Some(tid(12)));
        assert!(state.timer_token(TimerKind::AwaitGateDeadline).is_none());
    }

    /// `is_draining` is true exactly on `Active(PreFire(Draining), _)`.
    #[test]
    fn is_draining_is_true_only_on_active_prefire_draining() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        // Active PreFire Draining — true.
        let draining = ProfileState::Active(
            ActiveBurst::PreFire(unit_pre(PreFirePhase::Draining, tid(1))),
            BurstFinish::ReturnToIdle,
        );
        assert!(draining.is_draining());

        // BurstFinish doesn't influence the predicate.
        let draining_reap = ProfileState::Active(
            ActiveBurst::PreFire(unit_pre(PreFirePhase::Draining, tid(1))),
            BurstFinish::Reap,
        );
        assert!(draining_reap.is_draining());

        // Every other shape — false.
        for state in [
            ProfileState::Idle,
            ProfileState::Pending(DescentState::new(
                r,
                DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
                ProbeSlot::empty(),
                false,
            )),
            ProfileState::Active(
                ActiveBurst::PreFire(unit_pre(
                    PreFirePhase::Verifying {
                        slot: ProbeSlot::empty(),
                        target: r,
                    },
                    tid(1),
                )),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PreFire(batching_burst(tid(1), tid(2))),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst::new(
                    BurstIntent::Standard,
                    PostFirePhase::Awaiting {
                        outstanding: 1,
                        gate_deadline: tid(3),
                    },
                    DirtyProvenance::new(),
                )),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst::new(
                    BurstIntent::Standard,
                    PostFirePhase::Rebasing(ProbeSlot::empty()),
                    DirtyProvenance::new(),
                )),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst::new(
                    BurstIntent::Standard,
                    PostFirePhase::Settling {
                        settle_timer: tid(4),
                    },
                    DirtyProvenance::new(),
                )),
                BurstFinish::ReturnToIdle,
            ),
        ] {
            assert!(!state.is_draining(), "expected !is_draining for {state:?}");
        }
    }

    /// `descent_state` borrows the inner state in `Pending`, returns `None` for every other variant.
    #[test]
    fn descent_state_returns_some_only_on_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let descent = DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
            false,
        );
        let pending = ProfileState::Pending(descent);
        assert!(pending.descent_state().is_some());

        assert!(ProfileState::Idle.descent_state().is_none());
        let active = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2))),
            BurstFinish::ReturnToIdle,
        );
        assert!(active.descent_state().is_none());
    }

    /// `descent_state_mut` lets a caller advance the descent in place when the state is `Pending`.
    #[test]
    fn descent_state_mut_lets_caller_advance_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut state = ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a"), CompactString::from("b")])
                .expect("non-empty"),
            ProbeSlot::empty(),
            false,
        ));

        {
            let d = state.descent_state_mut().expect("Pending carries descent");
            d.remaining_components_mut().advance();
        }

        let d = state.descent_state().expect("still Pending");
        assert_eq!(
            d.remaining_components().iter().cloned().collect::<Vec<_>>(),
            vec![CompactString::from("b")]
        );

        // Mutator returns None on non-Pending states.
        let mut idle = ProfileState::Idle;
        assert!(idle.descent_state_mut().is_none());
    }

    /// `probe_correlation` projects the Pending descent slot; `take_probe` consumes it once and
    /// idles it. Both are total over the state space — Idle and Active carry no descent slot.
    #[test]
    fn probe_correlation_and_take_probe_track_pending_slot() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let c = ProbeCorrelation::from(42);

        let armed = || {
            ProfileState::Pending(DescentState::new(
                r,
                DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
                ProbeSlot::armed(c),
                false,
            ))
        };

        // Pending + armed ⇒ projects the correlation.
        let mut s = armed();
        assert_eq!(s.probe_correlation(), Some(c));

        // take_probe consumes exactly once and idles the slot.
        assert_eq!(s.take_probe(), Some(c));
        assert_eq!(s.probe_correlation(), None, "slot idled after take");
        assert_eq!(s.take_probe(), None, "second take is a None no-op");

        // Pending + empty ⇒ no correlation, no consume.
        let mut idle_pending = ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
            false,
        ));
        assert_eq!(idle_pending.probe_correlation(), None);
        assert_eq!(idle_pending.take_probe(), None);

        // Idle / Active hold no descent slot — total projection ⇒ None.
        assert_eq!(ProfileState::Idle.probe_correlation(), None);
        assert_eq!(ProfileState::Idle.take_probe(), None);
        let mut active = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2))),
            BurstFinish::ReturnToIdle,
        );
        assert_eq!(active.probe_correlation(), None);
        assert_eq!(active.take_probe(), None);
    }

    // -----------------------------------------------------------------------
    // State-machine setter / accessor API (clear_anchor_classification, materialize_anchor,
    // transition_state, anchor_claim setters, burst projections, read accessors, delegators,
    // take_current)
    // -----------------------------------------------------------------------

    use super::AnchorClaim;

    fn pending(r: ResourceId) -> ProfileState {
        ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("seg")]).expect("non-empty"),
            ProbeSlot::empty(),
            false,
        ))
    }

    fn active_prefire() -> ProfileState {
        ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2))),
            BurstFinish::ReturnToIdle,
        )
    }

    fn active_postfire() -> ProfileState {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst::new(
                BurstIntent::Standard,
                PostFirePhase::Rebasing(ProbeSlot::empty()),
                DirtyProvenance::new(),
            )),
            BurstFinish::ReturnToIdle,
        )
    }

    fn awaiting_post(outstanding: u32) -> PostFireBurst {
        PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Awaiting {
                outstanding,
                gate_deadline: tid(9),
            },
            DirtyProvenance::new(),
        )
    }

    /// The zero-edge: 3 → 2 → 1 → 0 reports `Decremented` until the last completion, then
    /// `LastReached`.
    #[test]
    fn note_effect_completion_counts_down_then_last_reached() {
        let mut post = awaiting_post(3);
        assert_eq!(post.note_effect_completion(), AwaitVerdict::Decremented);
        assert_eq!(post.note_effect_completion(), AwaitVerdict::Decremented);
        assert_eq!(post.note_effect_completion(), AwaitVerdict::LastReached);
        assert!(matches!(
            post.phase,
            PostFirePhase::Awaiting { outstanding: 0, .. }
        ));
    }

    /// A single outstanding effect hits zero on its first completion.
    #[test]
    fn note_effect_completion_single_is_last_reached() {
        assert_eq!(
            awaiting_post(1).note_effect_completion(),
            AwaitVerdict::LastReached
        );
    }

    /// Rebasing carries no outstanding-effect counter, so a late completion in Rebasing folds to
    /// `NotAwaiting`.
    #[test]
    fn note_effect_completion_on_rebasing_is_not_awaiting() {
        let mut post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );
        assert_eq!(post.note_effect_completion(), AwaitVerdict::NotAwaiting);
    }

    /// Over-completion (more `EffectComplete`s than emitted Effects) is an invariant breach — the
    /// dev/CI floor backstop.
    #[test]
    #[should_panic(expected = "outstanding underflow")]
    fn note_effect_completion_underflow_trips_assert() {
        let _ = awaiting_post(0).note_effect_completion();
    }

    /// `Profile` delegates through the state machine: `NotAwaiting` for every non-Awaiting state,
    /// the live verdict on `Active(Awaiting)`.
    #[test]
    fn note_effect_completion_delegates_through_profile() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert_eq!(p.note_effect_completion(), AwaitVerdict::NotAwaiting);
        p.transition_state(pending(r));
        assert_eq!(p.note_effect_completion(), AwaitVerdict::NotAwaiting);
        p.transition_state(active_prefire());
        assert_eq!(p.note_effect_completion(), AwaitVerdict::NotAwaiting);

        p.transition_state(ProfileState::Active(
            ActiveBurst::PostFire(awaiting_post(2)),
            BurstFinish::ReturnToIdle,
        ));
        assert_eq!(p.note_effect_completion(), AwaitVerdict::Decremented);
        assert_eq!(p.note_effect_completion(), AwaitVerdict::LastReached);
    }

    fn rebasing_post(intent: BurstIntent, residual: DirtyProvenance) -> PostFireBurst {
        PostFireBurst::new(
            intent,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            residual,
        )
    }

    /// The residual provenance seeds the restart: the typed move re-arms a fresh `Batching`
    /// Standard burst with the engine-minted timers and the anchor placeholder, carries the
    /// captured paths over whole (so the restarted burst's first verify obligates over them), and
    /// opens a fresh quiescence sequence — the restarted burst is constructed by
    /// [`PostFireBurst::into_pre_fire_residual`] from scratch, so any sample-sequence bookkeeping
    /// on the prior pre-fire burst does not survive the move.
    #[test]
    fn into_pre_fire_residual_seeds_a_fresh_batching_burst() {
        let mut tree = Tree::new();
        let anchor = tree.ensure_root("anchor", ResourceRole::User);
        let c1 = tree
            .ensure_child(anchor, "c1", ResourceRole::User)
            .expect("live");
        let c2 = tree
            .ensure_child(anchor, "c2", ResourceRole::User)
            .expect("live");
        let residual = dirty_prov(&[(c1, "/w/c1"), (c2, "/w/c2")]);
        let now = std::time::Instant::now();

        let pre = rebasing_post(BurstIntent::Standard, residual).into_pre_fire_residual(
            tid(7),
            tid(8),
            now,
            false,
        );

        assert_eq!(pre.burst_deadline, tid(7));
        assert!(matches!(
            pre.phase,
            PreFirePhase::Batching { settle_timer } if settle_timer == tid(8)
        ));
        assert_eq!(pre.intent, BurstIntent::Standard);
        assert!(!pre.forced);
        assert_eq!(
            pre.dirty.chains(),
            expected_chains(&["/w/c1", "/w/c2"]),
            "the move preserves the residual's captured paths",
        );
        assert_eq!(pre.last_event_time, Some(now));
    }

    /// A Seed-origin residual restarts just as a Standard one does: the move is origin-agnostic and
    /// *sets* `intent: Standard` (a restarted debounce burst is Standard by definition). This is
    /// the closed Seed-residual event-loss — a Seed drift → fire → rebase with absorbed events
    /// rejoins the Standard debounce lifecycle instead of being dropped. No origin gate, no panic;
    /// the reconfirm is a fresh query, so there is no per-origin balance to keep.
    #[test]
    fn into_pre_fire_residual_seed_origin_restarts_as_standard() {
        let mut tree = Tree::new();
        let anchor = tree.ensure_root("anchor", ResourceRole::User);
        let c1 = tree
            .ensure_child(anchor, "c1", ResourceRole::User)
            .expect("live");
        let residual = dirty_prov(&[(c1, "/w/c1")]);
        let pre = rebasing_post(BurstIntent::Seed, residual).into_pre_fire_residual(
            tid(1),
            tid(2),
            std::time::Instant::now(),
            false,
        );
        assert_eq!(
            pre.intent,
            BurstIntent::Standard,
            "Seed origin is rewritten to Standard — a restarted debounce burst is Standard",
        );
        assert!(matches!(
            pre.phase,
            PreFirePhase::Batching { settle_timer } if settle_timer == tid(2)
        ));
        assert_eq!(pre.burst_deadline, tid(1));
        assert_eq!(
            pre.dirty.chains(),
            expected_chains(&["/w/c1"]),
            "the move preserves the residual's captured path across origins",
        );
    }

    /// An empty residual is a misuse — the restart would have no seed and would mask a caller that
    /// failed to gate on a non-empty fire-tail.
    #[test]
    #[should_panic(expected = "empty residual")]
    fn into_pre_fire_residual_empty_residual_trips_assert() {
        let _ = rebasing_post(BurstIntent::Standard, DirtyProvenance::new())
            .into_pre_fire_residual(tid(1), tid(2), std::time::Instant::now(), false);
    }

    /// The pre-fire N=2 sample carrier drops by omission at the fire boundary:
    /// `PreFireBurst::into_post_fire` constructs a fresh `PostFireBurst::new` whose
    /// `last_certified_hash` is `None`, regardless of any prior pre-fire sample sequence. Pinning
    /// this structurally guards a future refactor that might accidentally thread the pre-fire
    /// carrier across the boundary — the post-fire rebase loop samples a different tree
    /// (post-command, not pre-), so cross-carrying a hash would be a category error.
    #[test]
    fn pre_fire_carrier_drops_at_into_post_fire() {
        let mut pre = batching_burst(tid(7), tid(99));
        assert_eq!(
            pre.advance_certified_sample(0xCAFE_F00D_u128),
            None,
            "first sample on a fresh burst returns None",
        );

        // The carrier is sealed (`pub(crate)` newtype) — assert the drop behaviorally: a surviving
        // carrier would make the post-fire burst's first advance return `Some(0xCAFE_F00D)`.
        let mut post = pre.into_post_fire(NonZeroU32::new(1).unwrap(), tid(55));
        assert_eq!(
            post.advance_certified_sample(0x1234_u128),
            None,
            "the typed fire move drops the pre-fire carrier by omission — \
             the post-fire rebase loop opens its own fresh sample sequence",
        );
    }

    /// The post-fire N=2 sample carrier drops by omission at the fire-tail restart:
    /// `PostFireBurst::into_pre_fire_residual` constructs a fresh `PreFireBurst` whose
    /// `last_certified_hash` is `None`, regardless of any prior post-fire sample sequence. The
    /// restarted Standard burst samples a third tree (post-rebase, re-debounced) and opens its own
    /// fresh quiescence sequence.
    #[test]
    fn post_fire_carrier_drops_at_into_pre_fire_residual() {
        let mut tree = Tree::new();
        let anchor = tree.ensure_root("anchor", ResourceRole::User);
        let c1 = tree
            .ensure_child(anchor, "c1", ResourceRole::User)
            .expect("live");
        let residual = dirty_prov(&[(c1, "/w/c1")]);

        let mut post = rebasing_post(BurstIntent::Standard, residual);
        assert_eq!(
            post.advance_certified_sample(0xDEAD_BEEF_u128),
            None,
            "first sample on a fresh burst returns None",
        );

        // Sealed carrier — assert the drop behaviorally: a surviving carrier would make the
        // restarted burst's first advance return `Some(0xDEAD_BEEF)`.
        let mut pre = post.into_pre_fire_residual(tid(1), tid(2), std::time::Instant::now(), false);
        assert_eq!(
            pre.advance_certified_sample(0x5678_u128),
            None,
            "the typed fire-tail restart drops the post-fire carrier by omission — \
             the restarted pre-fire burst opens its own fresh sample sequence",
        );
    }

    /// A fold *replaces* the fire, so a latched pre-fire burst must never reach the fire move.
    /// `into_post_fire`'s entry `debug_assert` is the structural dual of the verdict-time
    /// `AbsorbFold` override — reaching it means a classify-routing bug let a latched burst cross
    /// the fire boundary. Debug-only: the assert is compiled out in release.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "fold-latched burst must not fire")]
    fn into_post_fire_panics_on_latched_burst_in_debug() {
        let mut pre = batching_burst(tid(7), tid(99));
        pre.latch_fold();
        let _ = pre.into_post_fire(NonZeroU32::new(1).unwrap(), tid(55));
    }

    /// The fire-tail restart **threads** `fold_latched`, unlike the carriers it drops — a still-live
    /// absorb window's latch survives the move so the restarted Standard burst keeps folding.
    /// Complements `post_fire_carrier_drops_at_into_pre_fire_residual`, which threads `false`.
    #[test]
    fn into_pre_fire_residual_threads_fold_latched() {
        let mut tree = Tree::new();
        let anchor = tree.ensure_root("anchor", ResourceRole::User);
        let c1 = tree
            .ensure_child(anchor, "c1", ResourceRole::User)
            .expect("live");
        let residual = dirty_prov(&[(c1, "/w/c1")]);

        let pre = rebasing_post(BurstIntent::Standard, residual).into_pre_fire_residual(
            tid(1),
            tid(2),
            std::time::Instant::now(),
            true,
        );
        assert!(
            pre.fold_latched.is_latched(),
            "the restart threads the latch — a live absorb window keeps folding the restarted burst",
        );
    }

    // -----------------------------------------------------------------------
    // absorb window + fold-latch cascade
    // -----------------------------------------------------------------------

    /// `latch_fold` is a set-only monotone latch that reaches its target only through
    /// `Active(PreFire)`; `burst_fold_latched` reads it back. The cascade is asymmetric by
    /// construction: PostFire has no latch (no-op), and `Idle` / `Pending` have no in-flight burst
    /// to override (no-op). An unlatched `Active(PreFire)` reads `false` — the latch is the only
    /// thing that flips it.
    #[test]
    fn latch_fold_cascade_reaches_only_active_prefire() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        // Active(PreFire): latch flips the read.
        let mut prefire = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2))),
            BurstFinish::ReturnToIdle,
        );
        assert!(
            !prefire.burst_fold_latched(),
            "fresh PreFire reads unlatched",
        );
        prefire.latch_fold();
        assert!(prefire.burst_fold_latched(), "latch flips the PreFire read");

        // Active(PostFire): no latch field — latch is a no-op.
        let mut postfire = ProfileState::Active(
            ActiveBurst::PostFire(rebasing_post(BurstIntent::Standard, DirtyProvenance::new())),
            BurstFinish::ReturnToIdle,
        );
        postfire.latch_fold();
        assert!(
            !postfire.burst_fold_latched(),
            "PostFire has no latch — the cascade no-ops",
        );

        // Idle / Pending: no in-flight burst — latch is a no-op.
        for mut state in [ProfileState::Idle, pending(r)] {
            state.latch_fold();
            assert!(
                !state.burst_fold_latched(),
                "no in-flight pre-fire burst ⇒ latch no-ops, read stays false",
            );
        }
    }

    /// `arm_absorb` is set-plus-retro-latch in one operation: it sets the window unconditionally
    /// AND drives the latch cascade. On `Active(PreFire)` the in-flight burst retro-latches; on a
    /// state with no in-flight pre-fire burst (`Idle`) the window still stands for the next burst's
    /// birth consult but nothing latches. Re-arm is last-writer-wins over the whole window (mode
    /// AND expiry).
    #[test]
    fn arm_absorb_sets_window_and_retro_latches_active_prefire() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let base = std::time::Instant::now();

        // (i) Active(PreFire): window set AND in-flight burst retro-latched.
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(active_prefire());
        assert!(!p.state().burst_fold_latched(), "unlatched before arm");
        p.arm_absorb(base + SETTLE, AbsorbMode::ConsumeOnFirst);
        assert_eq!(
            p.absorb_window(),
            Some(&AbsorbWindow {
                expiry: base + SETTLE,
                mode: AbsorbMode::ConsumeOnFirst,
            }),
        );
        assert!(
            p.state().burst_fold_latched(),
            "arm retro-latches the in-flight PreFire burst",
        );

        // (ii) Idle: window stands for the next birth, nothing latches.
        let mut idle = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        idle.arm_absorb(base + SETTLE, AbsorbMode::PersistUntil);
        assert!(idle.absorb_window().is_some(), "window armed on Idle");
        assert!(
            !idle.state().burst_fold_latched(),
            "Idle has no in-flight burst — nothing retro-latches",
        );

        // (iii) Re-arm: last-writer-wins over mode AND expiry.
        let mut q = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        q.arm_absorb(base + SETTLE, AbsorbMode::ConsumeOnFirst);
        q.arm_absorb(base + MAX_SETTLE, AbsorbMode::PersistUntil);
        assert_eq!(
            q.absorb_window(),
            Some(&AbsorbWindow {
                expiry: base + MAX_SETTLE,
                mode: AbsorbMode::PersistUntil,
            }),
            "re-arm overwrites the window wholesale",
        );
    }

    /// `note_absorb_fold` bumps `absorb_count` **unconditionally** (the fold happened), then
    /// retires the window only when the current window is a `ConsumeOnFirst`. `PersistUntil`
    /// survives the bump; an unarmed (`None`) window stays `None` while the count still advances.
    #[test]
    fn note_absorb_fold_bumps_count_and_consumes_only_consume_on_first() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let base = std::time::Instant::now();

        // ConsumeOnFirst: count bumps, window retires.
        let mut consume = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        consume.arm_absorb(base + SETTLE, AbsorbMode::ConsumeOnFirst);
        consume.note_absorb_fold();
        assert_eq!(consume.absorb_count(), 1);
        assert!(
            consume.absorb_window().is_none(),
            "ConsumeOnFirst retires on the first fold",
        );

        // PersistUntil: count bumps, window stands.
        let mut persist = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        persist.arm_absorb(base + SETTLE, AbsorbMode::PersistUntil);
        persist.note_absorb_fold();
        assert_eq!(persist.absorb_count(), 1);
        assert!(
            persist.absorb_window().is_some(),
            "PersistUntil survives the fold",
        );

        // No window: count still bumps unconditionally, stays None.
        let mut unarmed = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        unarmed.note_absorb_fold();
        assert_eq!(
            unarmed.absorb_count(),
            1,
            "the bump is unconditional — the fold happened even with no live window",
        );
        assert!(unarmed.absorb_window().is_none());
    }

    /// `absorb_window_live` live-gates on `now < expiry`: `None` is never live; an armed window is
    /// live strictly before its expiry and inert at-or-after it (lazy expiry — the read never
    /// clears the window).
    #[test]
    fn absorb_window_live_gates_on_expiry() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let base = std::time::Instant::now();

        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!p.absorb_window_live(base), "no window armed ⇒ never live");

        p.arm_absorb(base + SETTLE, AbsorbMode::PersistUntil);
        assert!(p.absorb_window_live(base), "before expiry ⇒ live");
        assert!(
            !p.absorb_window_live(base + SETTLE),
            "at expiry ⇒ inert (now < expiry is strict)",
        );
        assert!(
            !p.absorb_window_live(base + SETTLE + SETTLE),
            "beyond expiry ⇒ inert",
        );
    }

    /// `absorb_window` borrows the armed window **without** live-gating — the contrast with
    /// `absorb_window_live`. Arming with an already-past expiry leaves the window observably `Some`
    /// for the projection surface, while the live gate reads `false`.
    #[test]
    fn absorb_window_does_not_live_gate() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let now = std::time::Instant::now();
        let past = now
            .checked_sub(SETTLE)
            .expect("monotonic clock past origin");

        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.arm_absorb(past, AbsorbMode::PersistUntil);
        assert_eq!(
            p.absorb_window(),
            Some(&AbsorbWindow {
                expiry: past,
                mode: AbsorbMode::PersistUntil,
            }),
            "absorb_window borrows the armed window regardless of expiry",
        );
        assert!(
            !p.absorb_window_live(now),
            "the same window is not live — the gate, not the borrow, applies expiry",
        );
    }

    #[test]
    fn profile_new_threads_kind_param() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let classified = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
            Some(ResourceKind::Dir),
        );
        assert_eq!(classified.kind(), Some(ResourceKind::Dir));
        let unprobed = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(unprobed.kind(), None);
    }

    #[test]
    fn read_accessors_project_the_sum() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        // Fresh: Unclassified — every anchor projection empty.
        assert!(matches!(p.state(), ProfileState::Idle));
        assert_eq!(p.anchor_claim(), AnchorClaim::None);
        assert_eq!(p.kind(), None);
        assert!(p.baseline().is_none());
        assert!(p.current().is_none());
        assert!(!p.current_is_some());
        assert!(p.current_dir().is_none());
        assert!(p.baseline_dir().is_none());
        assert_eq!(p.settled_hash(), None);

        // Graft + rebase: Dir in active mode (state E).
        let snap = empty_dir_snapshot();
        let h = TreeSnapshot::Dir(Arc::clone(&snap)).hash();
        p.install_dir_current(snap);
        p.rebase_baseline();
        p.install_anchor_claim_held();

        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        assert_eq!(p.kind(), Some(ResourceKind::Dir));
        assert!(matches!(p.baseline(), Some(TreeSnapshot::Dir(_))));
        assert!(matches!(p.current(), Some(TreeSnapshot::Dir(_))));
        assert!(p.current_is_some());
        assert!(p.current_dir().is_some(), "Dir current borrowable");
        assert!(p.baseline_dir().is_some(), "Dir baseline borrowable");
        assert_eq!(p.settled_hash(), Some(h), "settled hash = baseline hash");
    }

    #[test]
    fn transition_state_replaces_and_returns_prior() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        let prior = p.transition_state(pending(r));
        assert!(matches!(prior, ProfileState::Idle));
        assert!(matches!(p.state(), ProfileState::Pending(_)));

        let prior = p.transition_state(ProfileState::Idle);
        assert!(matches!(prior, ProfileState::Pending(_)));
        assert!(matches!(p.state(), ProfileState::Idle));
    }

    /// [`ProfileMap::map_state`] is the transform dual of [`ProfileMap::transition_state`]: it hands
    /// the prior to the closure by value, installs the [`ProfileState`] the closure computes from it,
    /// threads the auxiliary `R` back out, and reconciles [`ProfileMap::nonsteady`] across the one
    /// resulting edge in a single reconcile. A stale id short-circuits to `None` without running the
    /// closure: the `?` the fire-boundary callers (`finish_burst_to_idle`'s `.flatten()`) branch on.
    #[test]
    fn map_state_transforms_reconciles_once_and_skips_stale() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let pid = profiles.attach(
            &mut tree,
            mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        // A fresh Profile births steady Idle; the descent entry's `transition_state(Pending)`
        // records the +1.
        assert_eq!(profiles.nonsteady(), 0, "a fresh Profile births steady");
        let _ = profiles.transition_state(pid, pending(r));
        assert_eq!(
            profiles.nonsteady(),
            1,
            "the Idle → Pending push records the carrier edge",
        );

        // Transform Pending → Active(PreFire). The closure consumes the prior by value and computes
        // the next from it; the auxiliary `R` is threaded back out. Pending → Active is a nonsteady
        // true → false edge.
        let aux = profiles.map_state(pid, |prior| {
            assert!(
                matches!(prior, ProfileState::Pending(_)),
                "the closure consumes the prior state by value",
            );
            (active_prefire(), 0xABCD_u32)
        });
        assert_eq!(
            aux,
            Some(0xABCD),
            "the live-id path threads the auxiliary R out"
        );
        assert!(
            matches!(
                profiles.get(pid).unwrap().state(),
                ProfileState::Active(_, _)
            ),
            "the closure's computed state is installed",
        );
        assert_eq!(
            profiles.nonsteady(),
            0,
            "the single Active edge reconciled the carrier count once (1 → 0)",
        );

        // A stale id never runs the closure and returns `None` — the outer `None`
        // `finish_burst_to_idle` flattens against.
        let stale: Option<()> = profiles.map_state(ProfileId::default(), |s| (s, ()));
        assert!(
            stale.is_none(),
            "stale id short-circuits to None without transforming"
        );
        assert_eq!(
            profiles.nonsteady(),
            0,
            "a stale-id no-op leaves the carrier count untouched",
        );
    }

    #[test]
    fn clear_anchor_classification_unclassifies_and_captures_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let snap = empty_dir_snapshot();
        let expected = TreeSnapshot::Dir(Arc::clone(&snap)).hash();
        // Active-mode Dir (state E): graft then settle the baseline.
        p.install_dir_current(snap);
        p.rebase_baseline();

        p.clear_anchor_classification();

        assert_eq!(p.kind(), None, "unclassified after loss");
        assert!(p.baseline().is_none());
        assert!(!p.current_is_some());
        assert_eq!(
            p.settled_hash(),
            Some(expected),
            "witness captured from the settled baseline in one move",
        );
    }

    #[test]
    fn clear_anchor_classification_from_file_baseline_captures_leaf_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("file", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let leaf = empty_leaf_entry();
        let expected = TreeSnapshot::File(leaf.clone()).hash();
        p.install_file_current(leaf);
        p.rebase_baseline();

        p.clear_anchor_classification();

        assert_eq!(p.kind(), None);
        assert_eq!(
            p.settled_hash(),
            Some(expected),
            "File baseline hash captured as the witness",
        );
    }

    #[test]
    fn clear_anchor_classification_is_idempotent_preserving_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        // Already-lost shape: classified but baseline cleared at the prior loss, only the survival
        // witness remains.
        p.anchor = AnchorClassification::Dir {
            current: None,
            settled: SettledState::Witness(0x00c0_ffee),
        };

        p.clear_anchor_classification();
        assert_eq!(p.kind(), None);
        assert_eq!(p.settled_hash(), Some(0x00c0_ffee), "witness carried");

        // Second clear (already Unclassified) must preserve, not null.
        p.clear_anchor_classification();
        assert_eq!(
            p.settled_hash(),
            Some(0x00c0_ffee),
            "idempotent clear preserves the prior witness",
        );
    }

    #[test]
    fn materialize_anchor_classifies_from_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(pending(r));

        p.materialize_anchor(ResourceKind::Dir);

        assert!(matches!(p.state(), ProfileState::Idle));
        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        assert_eq!(p.kind(), Some(ResourceKind::Dir));
        assert!(!p.current_is_some(), "materialised, not yet grafted");
        assert_eq!(p.settled_hash(), None, "fresh: no witness, no baseline");
    }

    /// Recovery path: descent re-materialises an anchor that lost its baseline. The survival
    /// witness held on the `Unclassified` anchor must survive classification so the post-recovery
    /// Seed-Ok drift verdict still has a reference (states B → C).
    #[test]
    fn materialize_anchor_carries_survival_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(pending(r));
        p.anchor = AnchorClassification::Unclassified {
            witness: Some(0xfeed_face),
        };

        p.materialize_anchor(ResourceKind::Dir);

        assert_eq!(p.kind(), Some(ResourceKind::Dir));
        assert!(!p.current_is_some());
        assert_eq!(
            p.settled_hash(),
            Some(0xfeed_face),
            "witness carried forward through materialisation",
        );
        assert!(p.baseline().is_none(), "witness is not an active baseline");
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "state must be Pending")]
    fn materialize_anchor_panics_when_not_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        // Fresh Profile is Idle, not Pending — precondition breach.
        p.materialize_anchor(ResourceKind::Dir);
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "anchor_claim must be None")]
    fn materialize_anchor_panics_when_claim_held() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(pending(r));
        p.install_anchor_claim_held();
        p.materialize_anchor(ResourceKind::Dir);
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "anchor must be Unclassified")]
    fn materialize_anchor_panics_when_already_classified() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(pending(r));
        p.anchor = AnchorClassification::Dir {
            current: None,
            settled: SettledState::Unset,
        };
        p.materialize_anchor(ResourceKind::Dir);
    }

    #[test]
    fn anchor_claim_setters_are_idempotent() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.install_anchor_claim_held();
        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        p.install_anchor_claim_held();
        assert_eq!(
            p.anchor_claim(),
            AnchorClaim::Held,
            "idempotent against Held"
        );

        p.release_anchor_claim_now();
        assert_eq!(p.anchor_claim(), AnchorClaim::None);
        p.release_anchor_claim_now();
        assert_eq!(
            p.anchor_claim(),
            AnchorClaim::None,
            "idempotent against None"
        );
    }

    #[test]
    fn pre_fire_burst_mut_some_only_on_prefire_and_mutation_persists() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert!(p.pre_fire_burst_mut().is_none(), "Idle has no pre-fire");
        p.transition_state(pending(r));
        assert!(p.pre_fire_burst_mut().is_none(), "Pending has no pre-fire");
        p.transition_state(active_postfire());
        assert!(p.pre_fire_burst_mut().is_none(), "PostFire has no pre-fire");

        p.transition_state(active_prefire());
        let pre = p.pre_fire_burst_mut().expect("PreFire carries the payload");
        pre.forced = true;
        assert!(
            p.pre_fire_burst_mut().expect("still PreFire").forced,
            "mutation through the projection persists",
        );
    }

    #[test]
    fn post_fire_burst_mut_some_only_on_postfire_and_mutation_persists() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert!(p.post_fire_burst_mut().is_none(), "Idle has no post-fire");
        p.transition_state(active_prefire());
        assert!(
            p.post_fire_burst_mut().is_none(),
            "PreFire has no post-fire"
        );

        p.transition_state(active_postfire());
        let post = p
            .post_fire_burst_mut()
            .expect("PostFire carries the payload");
        post.final_window_residual
            .note(r, Arc::from(Path::new("/w/anchor")));
        assert!(
            p.post_fire_burst_mut()
                .expect("still PostFire")
                .final_window_residual
                .chains()
                .contains(Path::new("/w/anchor")),
            "mutation through the projection persists",
        );
    }

    #[test]
    fn delegators_route_to_profile_state() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        // descent_state_mut: Some only on Pending; advancing persists.
        assert!(p.descent_state_mut().is_none());
        p.transition_state(ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a"), CompactString::from("b")])
                .expect("non-empty"),
            ProbeSlot::empty(),
            false,
        )));
        p.descent_state_mut()
            .expect("Pending carries descent")
            .remaining_components_mut()
            .advance();
        assert_eq!(
            p.descent_state_mut()
                .expect("still Pending")
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("b")],
        );

        // take_probe delegates to ProfileState::take_probe: arming the Pending descent slot through
        // the typed mint edge, then taking it idles the machine state (the take is the linear
        // consume, so the slot is never dropped armed).
        p.descent_state_mut()
            .expect("still Pending")
            .arm_probe(ProbeCorrelation::from(7));
        assert_eq!(p.take_probe(), Some(ProbeCorrelation::from(7)));
        assert_eq!(p.take_probe(), None, "delegate idled the slot");

        // mark/clear_active_for_reap delegate the bool semantics.
        let mut q = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!q.mark_active_for_reap(), "Idle cannot be marked");
        assert!(!q.clear_active_reap(), "Idle has nothing to clear");
        q.transition_state(active_prefire());
        assert!(q.mark_active_for_reap(), "Active flips to Reap");
        assert!(q.mark_active_for_reap(), "already-Reap is idempotent true");
        assert!(q.clear_active_reap(), "zombie revived");
        assert!(!q.clear_active_reap(), "nothing left to clear");
    }

    #[test]
    fn take_current_takes_leaves_settled_and_is_idempotent() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());
        p.rebase_baseline();
        let settled = p.settled_hash();

        let taken = p.take_current();
        assert!(matches!(taken, Some(TreeSnapshot::Dir(_))));
        assert!(!p.current_is_some(), "take leaves current None");
        assert_eq!(
            p.settled_hash(),
            settled,
            "take_current does not disturb the settled baseline",
        );
        assert!(p.take_current().is_none(), "second take is idempotent");

        // Unclassified short-circuits to None.
        let mut q = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(q.take_current().is_none(), "Unclassified has no current");
    }

    /// Guarded random walk over the public anchor mutators, asserting after every op that the
    /// projection surface stays consistent with the underlying sum and that every reachable shape is
    /// one of the documented states. The snapshot-shape and baseline/witness-exclusion invariants are
    /// *structural* (no representable `AnchorClassification` violates them) — these assertions are
    /// the defense-in-depth tripwire that would catch a future flat-field regression or a projection
    /// bug. Guards respect each mutator's documented precondition so a step trips the consistency
    /// check, never a precondition `debug_assert!`. Deterministic xorshift64 PRNG, seed pinned in the
    /// fn name; 16 fresh Profiles so the one-shot `materialize_anchor` is exercised.
    #[test]
    fn anchor_projection_consistent_under_random_api_walk_seed_0x5eed_f00d() {
        struct XorShift64(u64);
        impl XorShift64 {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next_u64() % n
            }
        }

        fn is_dir(s: &TreeSnapshot) -> bool {
            matches!(s, TreeSnapshot::Dir(_))
        }

        // Every public projection must agree with the underlying sum, and the shape must be one of
        // the eight reachable rows.
        fn assert_invariants(p: &Profile, op: &str) {
            let current = p.current();
            let baseline = p.baseline();

            // Snapshot-shape: when both present they share a variant; kind tracks the current
            // variant.
            if let (Some(b), Some(c)) = (&baseline, &current) {
                assert_eq!(is_dir(b), is_dir(c), "baseline/current variant after {op}");
            }
            if let (Some(k), Some(c)) = (p.kind(), &current) {
                assert_eq!(
                    matches!(k, ResourceKind::Dir),
                    is_dir(c),
                    "kind/current variant after {op}",
                );
            }

            // Cheap predicate ⇔ owned accessor.
            assert_eq!(
                p.current_is_some(),
                current.is_some(),
                "current_is_some disagrees with current() after {op}",
            );

            // Typed-borrow projections agree with the owned views.
            assert_eq!(
                p.current_dir().is_some(),
                matches!(&current, Some(TreeSnapshot::Dir(_))),
                "current_dir disagrees with current() after {op}",
            );
            assert_eq!(
                p.baseline_dir().is_some(),
                matches!(&baseline, Some(TreeSnapshot::Dir(_))),
                "baseline_dir disagrees with baseline() after {op}",
            );
            if let Some(d) = p.current_dir() {
                assert_eq!(
                    TreeSnapshot::Dir(Arc::clone(d)).hash(),
                    current.as_ref().unwrap().hash(),
                    "current_dir hash disagrees with current() after {op}",
                );
            }

            // Reachable-state membership + projection ⇔ sum coherence.
            match &p.anchor {
                AnchorClassification::Unclassified { witness } => {
                    assert_eq!(p.kind(), None, "Unclassified ⇒ kind None after {op}");
                    assert!(
                        baseline.is_none() && current.is_none(),
                        "Unclassified ⇒ no snapshot after {op}",
                    );
                    assert_eq!(
                        p.settled_hash(),
                        *witness,
                        "Unclassified settled_hash is the carried witness after {op}",
                    );
                }
                // `baseline` is exposed iff `settled` is an active `Snapshot`; a `Witness` is a
                // survival hash, not a live baseline. `Snapshot` xor `Witness` is structural, so
                // this can never observe both. The File / Dir arms differ only in the `settled`
                // payload type, so each computes its own expectation.
                AnchorClassification::File { settled, .. } => {
                    assert_eq!(
                        baseline.is_some(),
                        matches!(settled, SettledState::Snapshot(_)),
                        "baseline() ⇔ settled Snapshot (File) after {op}",
                    );
                    let expected = match settled {
                        SettledState::Unset => None,
                        SettledState::Snapshot(_) => baseline.as_ref().map(TreeSnapshot::hash),
                        SettledState::Witness(h) => Some(*h),
                    };
                    assert_eq!(
                        p.settled_hash(),
                        expected,
                        "settled_hash disagrees with settled (File) after {op}",
                    );
                }
                AnchorClassification::Dir { settled, .. } => {
                    assert_eq!(
                        baseline.is_some(),
                        matches!(settled, SettledState::Snapshot(_)),
                        "baseline() ⇔ settled Snapshot (Dir) after {op}",
                    );
                    let expected = match settled {
                        SettledState::Unset => None,
                        SettledState::Snapshot(_) => baseline.as_ref().map(TreeSnapshot::hash),
                        SettledState::Witness(h) => Some(*h),
                    };
                    assert_eq!(
                        p.settled_hash(),
                        expected,
                        "settled_hash disagrees with settled (Dir) after {op}",
                    );
                }
            }
        }

        let mut master = XorShift64(0x5EED_F00D_D1CE_C0DE);
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        for _ in 0..16 {
            let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
            let mut rng = XorShift64(master.next_u64() | 1);
            assert_invariants(&p, "construction");

            for _ in 0..512 {
                // Ops 0..=5 are the six anchor-classification mutators (install_dir / install_file
                // / clear / materialize / rebase / take_current); ops 6..=9 drive the orthogonal
                // ProfileState axis so the materialize precondition (`Pending`) is reachable
                // mid-walk.
                match rng.below(10) {
                    0 => {
                        // Precondition: not File-classified (cross-arm graft is a
                        // dispatcher-boundary breach).
                        if !matches!(p.kind(), Some(ResourceKind::File)) {
                            p.install_dir_current(empty_dir_snapshot());
                            assert_invariants(&p, "install_dir_current");
                        }
                    }
                    1 => {
                        if !matches!(p.kind(), Some(ResourceKind::Dir)) {
                            p.install_file_current(empty_leaf_entry());
                            assert_invariants(&p, "install_file_current");
                        }
                    }
                    2 => {
                        p.clear_anchor_classification();
                        assert_invariants(&p, "clear_anchor_classification");
                    }
                    3 => {
                        // Precondition: Pending, no claim, Unclassified.
                        let pending = matches!(p.state(), ProfileState::Pending(_));
                        if pending && p.anchor_claim() == AnchorClaim::None && p.kind().is_none() {
                            let k = if rng.below(2) == 0 {
                                ResourceKind::Dir
                            } else {
                                ResourceKind::File
                            };
                            p.materialize_anchor(k);
                            assert_invariants(&p, "materialize_anchor");
                        }
                    }
                    4 => {
                        // Precondition: a live current to settle.
                        if p.current_is_some() {
                            p.rebase_baseline();
                            assert_invariants(&p, "rebase_baseline");
                        }
                    }
                    5 => {
                        // Total across the sum, no precondition: Unclassified ⇒ no-op `None`;
                        // File/Dir ⇒ takes `current`, leaving `settled` untouched (states E→F /
                        // D→C). The only route to state F (`current` None ∧ `settled` Snapshot) in
                        // the interleaved walk — the standalone unit test can't catch a *sequenced*
                        // coherence regression.
                        p.take_current();
                        assert_invariants(&p, "take_current");
                    }
                    6 => {
                        p.transition_state(ProfileState::Idle);
                        assert_invariants(&p, "transition_state(Idle)");
                    }
                    7 => {
                        p.transition_state(pending(r));
                        assert_invariants(&p, "transition_state(Pending)");
                    }
                    8 => {
                        p.transition_state(active_prefire());
                        assert_invariants(&p, "transition_state(PreFire)");
                    }
                    _ => {
                        p.transition_state(active_postfire());
                        assert_invariants(&p, "transition_state(PostFire)");
                    }
                }
            }
        }
    }

    /// `Profile::new`'s `kind` → sum projection is total: `None` ⇒ `Unclassified` (state A),
    /// `Some(Dir)` / `Some(File)` ⇒ a classified anchor with no snapshot or baseline (state C′).
    #[test]
    fn profile_new_projects_kind_to_initial_state() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        let a = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(matches!(
            a.anchor,
            AnchorClassification::Unclassified { witness: None }
        ));
        assert_eq!(a.kind(), None);
        assert_eq!(a.settled_hash(), None);

        let c_dir = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
            Some(ResourceKind::Dir),
        );
        assert!(matches!(
            c_dir.anchor,
            AnchorClassification::Dir {
                current: None,
                settled: SettledState::Unset
            }
        ));
        assert_eq!(c_dir.kind(), Some(ResourceKind::Dir));
        assert!(!c_dir.current_is_some());
        assert_eq!(c_dir.settled_hash(), None);

        let c_file = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
            Some(ResourceKind::File),
        );
        assert!(matches!(
            c_file.anchor,
            AnchorClassification::File {
                current: None,
                settled: SettledState::Unset
            }
        ));
        assert_eq!(c_file.kind(), Some(ResourceKind::File));
    }

    /// `Some(ResourceKind::Unknown)` is defensively dead — the sole production caller threads
    /// `Resource::kind()` which maps `Unknown → None`. Release builds degrade to `Unclassified`
    /// (same shape as `None`) rather than constructing an illegal state; debug builds trip the
    /// `debug_assert!`.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "Resource::kind() yields Unknown→None")]
    fn profile_new_unknown_kind_is_defensively_dead() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let _ = mk_profile(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
            Some(ResourceKind::Unknown),
        );
    }

    /// `settled_hash` is the one total drift reference across the sum: not-yet-settled ⇒ `None`;
    /// active baseline ⇒ its digest; loss-window witness ⇒ the retained hash; carried after a clear.
    #[test]
    fn settled_hash_is_total_across_the_sum() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(p.settled_hash(), None, "Unclassified, no witness");

        let snap = empty_dir_snapshot();
        let h = TreeSnapshot::Dir(Arc::clone(&snap)).hash();
        p.install_dir_current(snap);
        assert_eq!(p.settled_hash(), None, "grafted but not settled (Unset)");

        p.rebase_baseline();
        assert_eq!(p.settled_hash(), Some(h), "active baseline digest");

        p.clear_anchor_classification();
        assert_eq!(p.settled_hash(), Some(h), "witness carried across loss");
        assert_eq!(p.kind(), None);
    }

    /// `debug_assert_anchor_coherent` enforces the residual cross-axis invariant `Pending | Parked
    /// ⇒ Unclassified ∧ ¬Held`. The happy path (every shape outside `Pending` / `Parked`, or either
    /// while `Unclassified`) is silent; a classified `Pending` trips (the `Parked` trip is the same
    /// assert).
    #[test]
    fn anchor_coherent_is_silent_on_reachable_shapes() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.debug_assert_anchor_coherent(); // Idle + Unclassified
        p.transition_state(pending(r));
        p.debug_assert_anchor_coherent(); // Pending + Unclassified ✓
        p.transition_state(ProfileState::Parked);
        p.debug_assert_anchor_coherent(); // Parked + Unclassified ✓
        p.transition_state(ProfileState::Idle);
        p.install_dir_current(empty_dir_snapshot());
        p.debug_assert_anchor_coherent(); // Idle + classified ✓
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "Pending Profile must be Unclassified")]
    fn anchor_coherent_trips_on_classified_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.transition_state(pending(r));
        p.anchor = AnchorClassification::Dir {
            current: None,
            settled: SettledState::Unset,
        };
        p.debug_assert_anchor_coherent();
    }
}
