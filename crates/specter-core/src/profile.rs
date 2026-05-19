//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is computed at construction from
//! `(config, max_settle)` and is the lifetime-stable identity of the Profile.
//! `ProfileMap` keeps `(resource, config_hash) → ProfileId` and updates
//! `Resource.profiles` in lockstep — `attach`/`detach` are the only mutators
//! of either index.

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
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One fire cycle, split by the fire-transition boundary.
///
/// A burst lives `Idle → Active(ActiveBurst) → Idle`. The fire transition
/// (`Verifying → Awaiting`) is a typed state-machine move from
/// [`PreFireBurst`] to [`PostFireBurst`]: the two sides have disjoint
/// valid mutators, valid timers, valid probe responses, and accumulator
/// semantics. Encoding the split at the type level means a field that
/// has no post-fire consumer (e.g. `forced`, `probe_target`,
/// `last_event_time`) cannot leak across the boundary by construction.
///
/// **Pre-fire** (`Batching | Verifying | Draining`): event-driven
/// debounce window, in-flight verify or self-stable / descendants-pending
/// idle. Carries the event-provenance accumulator (`dirty`), the
/// settle-deadline source of truth (`last_event_time`), and the
/// quiescence proof (`certified`).
///
/// **Post-fire** (`Awaiting | Rebasing | RebaseSettling`): the actuator
/// gate, then the *structural mirror* of the pre-fire loop —
/// `RebaseSettling ⇄ Rebasing` is `Batching ⇄ Verifying`, folding
/// [`PostFireBurst`]'s own [`CertifiedPrior`] N=2 proof, bounded by
/// its own `rebase_ceiling` (pre-fire's `burst_deadline` analogue),
/// over the *post-command* tree. The pre-fire fields that encode a
/// fire decision do not cross the boundary — the typed
/// [`PreFireBurst::into_post_fire`] move drops them, and the
/// `BurstDeadline` timer becomes structurally irrelevant
/// ([`PostFireBurst::timer_token`] folds it to `None` for post-fire
/// phases, so the engine's stale-drain lazily collects the heap
/// entry). The pre-fire `certified` does **not** carry across: the
/// post-fire burst opens its own fresh proof, because the tree the
/// rebase loop must prove quiescent is the one the command just
/// mutated — and the rebase probe walks `WholeSubtree` (the
/// post-command tree has no trustworthy prior to skip against, exactly
/// as Seed). Its one fresh accumulator is the post-fire
/// `dirty`, which `absorb_event_into_fire_tail` feeds; it is
/// no longer a proof-obligation source (the `WholeSubtree` walk
/// observes everything regardless), only the fire-tail residual restart
/// seed, reset at every `Rebasing` re-entry so a `Stable` terminal
/// restarts only on the genuine final-window race.
#[derive(Debug)]
pub enum ActiveBurst {
    PreFire(PreFireBurst),
    PostFire(PostFireBurst),
}

/// Event provenance accumulated across a burst's pre-fire life (and,
/// for the post-fire fire-tail, the residual restart seed).
///
/// Key = the live engine slot the event named. Value = that slot's
/// path, `Arc::clone`d at ingest from the already-resolved live
/// `&Resource` (the `watch_demand > 0` gate proved the slot live).
/// Where an event landed is a *historical fact* — immutable from the
/// instant of ingest and immune to the slot later being reaped
/// (delete-recreate at the same path). A reaped key never invalidates
/// its captured path.
///
/// The Standard pre-fire proof obligation derives from the **values**,
/// never the keys: [`Self::chains`] is the dirty root→leaf chains the
/// walker must freshly observe, and [`Self::lca_path`] is their
/// component-wise lowest common ancestor — the tightest directory the
/// probe can root at without excluding a chain. Sourcing both from the
/// captured paths is what makes an empty `Chains` over a fully
/// reaped-id set unconstructable: liveness never filters the
/// projection.
///
/// The map is keyed by the slot, not reduced to a bare path set,
/// for two reasons: per-slot **last-writer-wins** dedup (a slot
/// firing N events contributes one entry, not N — see
/// [`Self::note`]), and retaining the live-slot id as the cheap
/// basis for any future caller needing *current* liveness rather
/// than history (today none on the Standard pre-fire path — the
/// projection reads only the values). No public setter —
/// [`Self::note`] is the sole accumulator edge.
#[derive(Debug, Default)]
pub struct DirtyProvenance(BTreeMap<ResourceId, Arc<Path>>);

impl DirtyProvenance {
    /// An empty accumulator. `const` for the `burst.rs` constructors
    /// and the typed post-fire move.
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Record that an `FsEvent` named `id` at `path`. The sole
    /// accumulator edge. `path` is an `Arc::clone` of the live
    /// `&Resource`'s materialised path captured at the ingest site —
    /// total by construction (the `watch_demand > 0` gate proved the
    /// slot live), so no fallible `path_of`, no `Option`. Last-writer
    /// -wins per id; ids are stable, so a repeat event for one slot
    /// re-stores the identical path.
    pub fn note(&mut self, id: ResourceId, path: Arc<Path>) {
        self.0.insert(id, path);
    }

    /// No event recorded yet. The Seed first-fire witness
    /// (`seed_owes_first_fire`) and the fire-tail residual restart gate
    /// read this; a Standard pre-fire burst is non-empty by
    /// construction (its constructor notes the trigger).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Drop every recorded event — the per-`Rebasing`-entry fire-tail
    /// residual reset. Behaviour-preserving swap-in for the prior
    /// `BTreeSet::clear`.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// The dirty root→leaf chains for [`crate::ProofObligation::Chains`]:
    /// every captured path, `BTreeSet`-ordered for deterministic replay.
    /// Every captured path is at-or-under the burst's probe target by
    /// construction (the target is the live id at [`Self::lca_path`], or
    /// the anchor fallback — both ancestors-or-equal of every value), so
    /// the prior "intersect with the target subtree" filter is a
    /// tautology and is gone. Never empty for a Standard pre-fire burst.
    #[must_use]
    pub fn chains(&self) -> BTreeSet<Arc<Path>> {
        self.0.values().map(Arc::clone).collect()
    }

    /// The component-wise lowest common ancestor of every captured path
    /// — the tightest directory the walker can root at without
    /// excluding a chain. `None` iff empty.
    ///
    /// Component-wise (not byte-prefix) is load-bearing: `/a` must not
    /// match `/ab`. Sound because v1 forbids symlinks / cross-filesystem,
    /// so a shared component prefix is genuine Tree ancestry. A lone
    /// captured path (the dominant single-file-edit case) returns itself
    /// with no allocation; the engine resolves the result to a live id
    /// and promotes a File leaf to its parent Dir.
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

/// Longest shared **component** prefix of two paths, borrowed from `a`.
/// Walks `Path::components` in lockstep, then strips `a`'s trailing
/// components past the divergence via `Path::parent` (each step a
/// sub-slice of `a`, so the result keeps `a`'s lifetime). Component
/// -wise, so `/a` is never a prefix of `/ab`. Both inputs are absolute
/// (materialised from the root chain) and share at least the root, so
/// the result is never empty.
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

/// Pre-fire lifecycle — every phase before the fire transition.
///
/// Fields are split across three roles:
/// - **Burst-scoped invariants** (`intent`, `forced`, `burst_deadline`,
///   `probe_target`): survive every pre-fire phase transition.
/// - **Pre-fire event state** (`dirty`, `last_event_time`): populated
///   by `event_drives_batching` on every `FsEvent`, for both intents
///   (both burst constructors are Batching-first). `dirty`'s captured
///   paths are the obligation + scope basis re-projected at each
///   `transition_to_verifying` for a Standard burst, and live-but-inert
///   for a Seed (anchor target + `WholeSubtree`); `last_event_time` is
///   the settle deadline's source of truth for both.
/// - **Quiescence proof** (`certified`): the [`CertifiedPrior`] floor —
///   the prior certified sample of the N=2 stability sequence, never an
///   event-accumulator.
///
/// `dirty` is preserved across the burst's pre-fire lifetime because
/// the obligation + scope are re-projected from it at every reconfirm
/// (`Draining → Verifying`) — the *projection* mutates, the captured
/// -path *basis* doesn't.
///
/// `certified` is preserved verbatim across every pre-fire phase swap
/// (`event_drives_batching`, `transition_to_draining`,
/// `unstable_response_drives_batching`) so the two samples may straddle
/// re-batches; its advance/withhold contract lives on
/// [`CertifiedPrior`].
///
/// `probe_target` is the resource id of the latest emitted probe.
/// Initialised to the Profile's anchor at burst start; overwritten by
/// `transition_to_verifying` (LCA for Standard, anchor for Seed) and by
/// `transition_to_rebasing` (anchor unconditionally). Non-Optional —
/// the anchor is a meaningful pre-probe initial value, and every
/// reader either knows it's been written or correctly treats it as the
/// fallback. The prior `Option<ResourceId>` with a `unwrap_or(anchor)`
/// fallback at every reader was the same semantics with one extra
/// nullability layer.
///
/// `last_event_time` is the source of truth for the settle deadline:
/// the settle timer is scheduled once on Batching entry and reschedules
/// on expiry only when `last_event_time` has advanced since. Event
/// arrivals while already in Batching update this field but do **not**
/// re-insert a fresh heap entry — heap inserts are bounded to one per
/// `last_event_time + settle` boundary, regardless of event density.
/// Seeded `Some(burst-start)` by *both* burst constructors (both are
/// Batching-first); the `Option` survives only because
/// `on_settle_expired` reads it defensively and folds a `None` straight
/// to the `Verifying` transition.
#[derive(Debug)]
pub struct PreFireBurst {
    pub burst_deadline: TimerId,
    pub phase: PreFirePhase,
    pub intent: BurstIntent,
    pub forced: bool,
    /// Event provenance — every `FsEvent` that drove (or is driving)
    /// this burst, captured `(slot, path)` at ingest. Constructed with
    /// the trigger by *both* `start_standard_burst` (always — its
    /// trigger is mandatory) and `start_seed_burst` (iff the Seed has a
    /// triggering `FsEvent`; empty otherwise), then `event_drives_batching`
    /// notes each later FsEvent during the pre-fire phases
    /// (`Batching | Verifying | Draining`), for *both* intents.
    ///
    /// **Two intent-specific consumers.**
    /// - *Standard* projects the captured **paths** to the
    ///   `ProofObligation::Chains` and their component-LCA (resolved to
    ///   a live id by `pre_fire_target`) to the probe target — both
    ///   immune to slot reaping because they read history, not current
    ///   liveness.
    /// - *Seed* targets the anchor and carries
    ///   `ProofObligation::WholeSubtree` unconditionally, so this is
    ///   **not** its probe-target / obligation source; instead its
    ///   *non-emptiness is the first-fire witness*. A fresh, never-fired
    ///   Seed fires its `SubtreeRoot` Subs iff it observed activity
    ///   (`!dirty.is_empty()`, the engine's `seed_owes_first_fire`
    ///   gate); empty ⇔ no activity ⇔ restart-safe silent pin (a daemon
    ///   restart over a static tree must not re-fire — Specter persists
    ///   no baseline, so every restart is a fresh Seed). A recovery Seed
    ///   (`any_fired`) ignores this and uses the drift oracle instead.
    pub dirty: DirtyProvenance,
    /// Latest probe target. Initialised to the Profile's anchor at
    /// burst start. Overwritten by `transition_to_verifying` to the
    /// `pre_fire_target` result (File anchor → anchor; Seed → anchor;
    /// Standard → the live id at the component-LCA of `dirty`'s captured
    /// paths, a File leaf promoted to its parent Dir, anchor on any
    /// resolution miss). `transition_to_rebasing` targets the anchor
    /// unconditionally but does not write this field (the post-fire
    /// phases live on `PostFireBurst`, which has no `probe_target` —
    /// Rebasing's target is structurally fixed).
    ///
    /// **Draining → Verifying reconfirm.** Recomputed via the same
    /// `pre_fire_target` rule because `dirty` is preserved across the
    /// burst's pre-fire lifetime: production code only ever *notes* into
    /// it, so the captured-path basis is identical at the reconfirm.
    /// The probe target and obligation derive from the captured paths
    /// (history), so a slot reaped during Draining cannot collapse
    /// either — only the live-id *resolution* may fall back to the
    /// anchor, which is strictly wider and never clips a chain.
    pub probe_target: ResourceId,
    /// Wall-clock instant of the most recent `FsEvent` that drove this
    /// burst. The **source of truth** for the Batching settle deadline:
    /// the live settle timer's heap entry pins to a fixed deadline
    /// (`burst-start + settle`, or `prior_last_event + settle` after a
    /// reschedule), but the deadline the burst is *waiting for* is
    /// `last_event_time + settle`. The on-expiry reschedule check
    /// reconciles the two — if `now − last_event_time < settle` the
    /// expiry handler reschedules a fresh entry at `last_event_time +
    /// settle` and stays in Batching; otherwise it transitions to
    /// Verifying.
    ///
    /// **Lifecycle.**
    /// - `Some(now)` from `start_standard_burst` — the burst-start
    ///   `FsEvent` is the first event and seeds the field.
    /// - `Some(now)` from `start_seed_burst` — both burst constructors
    ///   are Batching-first, so the Seed burst seeds this with its
    ///   start instant exactly as Standard does. There is no first
    ///   `FsEvent` for a Seed, but the burst-start instant is the
    ///   principled seed: the settle timer was scheduled at `now +
    ///   settle`, so on its expiry `expiry_now − last_event_time ≥
    ///   settle` holds by construction and the burst transitions to
    ///   `Verifying` cleanly; a fresh `FsEvent` during the Seed
    ///   `Batching` debounces via `event_drives_batching` identically
    ///   to a Standard burst.
    /// - Updated by `event_drives_batching` on every event.
    /// - **Pinned to `Some(now)`** by
    ///   `unstable_response_drives_batching` — the verify just
    ///   responded, and pinning the timestamp removes the `Instant`
    ///   monotonicity dependency from the reschedule correctness
    ///   argument.
    /// - **Preserved** across `transition_to_verifying` (the reconfirm
    ///   path) and `transition_to_draining` — phase swaps without
    ///   semantic resets.
    ///
    /// **Distinct from the watcher's `last_event_at`.** The watcher's
    /// field is per-watcher, scoped to drain-cadence recency. This field
    /// is per-burst, scoped to settle-deadline reschedule. Different
    /// consumers, different cadences.
    pub last_event_time: Option<Instant>,
    /// The N=2 quiescence proof — the prior `Authoritative` sample.
    /// Constructed fresh ([`CertifiedPrior::new`]) by the category-(a)
    /// `burst.rs` constructors (`start_standard_burst`,
    /// `start_seed_burst`) and reset by
    /// [`PostFireBurst::into_pre_fire_residual`]; thereafter advanced
    /// only via [`CertifiedPrior::advance`]. The advance/withhold
    /// contract and the no-setter invariant live on the type — see
    /// [`CertifiedPrior`].
    pub certified: CertifiedPrior,
}

/// Pre-fire phase discriminator.
///
/// `Batching` carries its own correlation token (`settle_timer: TimerId`)
/// because timer correlation is per-Burst and has no peer slot to live
/// on. `Verifying` carries a [`ProbeSlot`]: the pre-fire stability
/// probe's liveness *and* identity live on the phase, so a verify in
/// flight without its correlation is unconstructable and I5 ("at most
/// one outstanding probe") is a representability property of the single
/// slot. `Draining` carries no correlation token of its own: its exit
/// is driven by a fresh query over the live tree
/// ([`ProfileState::in_active_standard_burst`]), swept at every
/// `finish_burst_to_idle` — no per-phase token, no cached counter.
#[derive(Debug)]
pub enum PreFirePhase {
    /// Activity-gap detection. `settle_timer` is the armed debounce
    /// timer; an `FsEvent` reschedules it (`event_drives_batching`),
    /// timer expiry advances to `Verifying` (`transition_to_verifying`).
    Batching { settle_timer: TimerId },
    /// Pre-fire stability probe. The [`ProbeSlot`] is armed with the
    /// correlation the response must echo while the probe is in flight;
    /// it is empty only for the transient post-Cancel window before the
    /// burst re-arms `Batching` (`event_drives_batching`). Consuming the
    /// response disarms the slot exactly once — the structural
    /// consume-once guarantee. Constructing the variant *requires* a
    /// slot, so a verify phase without a correlation cannot exist:
    ///
    /// ```compile_fail
    /// use specter_core::PreFirePhase;
    /// // `Verifying` is not unit — a bare tag is not a `PreFirePhase`.
    /// let _: PreFirePhase = PreFirePhase::Verifying;
    /// ```
    ///
    /// ```
    /// use specter_core::{PreFirePhase, ProbeSlot};
    /// let _ = PreFirePhase::Verifying(ProbeSlot::empty());
    /// ```
    Verifying(ProbeSlot),
    /// Self-stable; descendants pending. The stable snapshot lives on
    /// `Profile.current` — `dispatch_standard_ok` updates `current` to
    /// the stable response immediately before transitioning here, so the
    /// tree-reconcile / Watch side keeps a faithful baseline. The
    /// reconfirm probe (Draining → Verifying, fired by the
    /// `finish_burst_to_idle` sweep once no covered descendant is still
    /// in an Active Standard burst) derives its quiescence verdict from
    /// the carrier proof ([`CertifiedPrior::advance`] over the burst's
    /// `certified` prior), not from a re-comparison against
    /// `Profile.current` — the verdict no longer depends on the
    /// splice-mutated snapshot. Holding a duplicate `TreeSnapshot` on
    /// the variant would only invite drift between the two references.
    Draining,
}

/// Post-fire lifecycle — the structural mirror of [`PreFireBurst`].
///
/// Post-fire runs its own quiescence loop over the *post-command*
/// tree, so it carries the same two invariant-bearing fields pre-fire
/// does — a [`CertifiedPrior`] N=2 proof (`certified`) and a loop
/// bound (`rebase_ceiling`, the post-fire analogue of pre-fire's
/// `burst_deadline`). The pre-fire fields that encode a *fire
/// decision* genuinely do not cross the boundary, dropped by leaving
/// them out of [`PreFireBurst::into_post_fire`]:
/// - No `forced`: the fire already happened — no decision left.
/// - No `burst_deadline`: the pre-fire ceiling; `rebase_ceiling` is
///   the post-fire one. The stale pre-fire timer lazy-drops via
///   [`PostFireBurst::timer_token`]'s `Settle | BurstDeadline` arm.
/// - No `probe_target`: Rebasing always targets the Profile's anchor.
/// - No `last_event_time`: the pre-fire settle-deadline source.
///
/// The pre-fire `dirty` (the captured-path basis) also does not cross;
/// the post-fire `dirty` is a *distinct, freshly-empty* provenance
/// accumulator (the fire-tail residual), not the pre-fire one carried
/// over.
///
/// `intent: BurstIntent` survives post-fire so
/// `dispatch_rebase_{vanished,failed}` can tag the `ProbeVanished` /
/// `ProbeFailed` diagnostic with it (Seed-driven drift rebases and
/// Standard-driven post-fire rebases both reach PostFire, and the
/// diagnostic distinguishes them). It is also the field
/// [`ProfileState::in_active_standard_burst`] reads — the reconfirm
/// query treats a post-fire Standard burst as still covering its
/// ancestors, exactly the lifetime the old refcount bracketed. The
/// fire-tail residual restart is **not** gated on it: the reconfirm is
/// a fresh query, not a per-origin refcount, so a Seed origin restarts
/// just as a Standard one does.
///
/// **Single construction seam.** Every `PostFireBurst` is born fresh —
/// `certified` empty, `rebase_ceiling` unstarted — through
/// [`Self::new`]; [`PreFireBurst::into_post_fire`] (the typed fire
/// move) is its only production caller. The post-command tree is a
/// *different tree* than the one the pre-fire carrier proved, so the
/// rebase loop opens its own N=2 sequence rather than inheriting the
/// pre-fire `certified` — the no-cross-fire-carry the typed move
/// exists to enforce.
#[derive(Debug)]
pub struct PostFireBurst {
    pub intent: BurstIntent,
    pub phase: PostFirePhase,
    /// Events absorbed during the post-fire tail
    /// (`Awaiting | Rebasing | RebaseSettling`), captured `(slot, path)`
    /// by `absorb_event_into_fire_tail` in `drive_burst`'s post-fire arm.
    ///
    /// **Not a proof-obligation source.** The rebase probe walks
    /// `WholeSubtree` (the post-command tree has no trustworthy prior),
    /// so it re-observes the whole anchor subtree regardless of this
    /// accumulator — every absorbed event is folded into the rebase
    /// verdict by the next `WholeSubtree` read whether or not it is
    /// recorded here. It is reset at *every* `Rebasing` entry
    /// (`transition_to_rebasing`, the first `Awaiting → Rebasing` edge
    /// and each `RebaseSettling → Rebasing` re-arm), so when the rebase
    /// loop terminates it holds only the events that landed during the
    /// **final** probe round-trip — the genuine final-window race
    /// (a change observed by the sensor's `Stable`-producing walk's
    /// instant but after the engine could fold it).
    ///
    /// **Sole consumer: the final-window restart seed.** At
    /// `dispatch_rebase_ok`'s `Stable` terminal a non-empty residual on
    /// a `ReturnToIdle` burst restarts a fresh debounced burst seeded
    /// from it (`into_pre_fire_residual` moves the whole provenance, so
    /// the restarted Standard burst's first verify has its captured
    /// paths intact), so that final-window change is not lost. A zombie
    /// (`Reap`) burst, an empty residual, or a ceiling terminal (no
    /// restart) drops it at `finish_burst_to_idle`. Without the
    /// per-entry reset, any tree-touching command would leave a
    /// non-empty residual and spuriously restart; with it the restart
    /// fires only for the real race. The restarted burst's settle
    /// window reckons from the rebase-response instant, not the absorbed
    /// events', a bounded ≤ one-`settle` extra re-fire latency.
    pub dirty: DirtyProvenance,
    /// The post-fire rebase loop's N=2 quiescence proof — the prior
    /// `Authoritative` sample of the *post-command* tree. Born fresh
    /// ([`CertifiedPrior::new`]) at [`Self::new`]; advanced only via
    /// [`CertifiedPrior::advance`] when the rebase verify folds a
    /// verdict. Disjoint from the pre-fire carrier's `certified` by
    /// construction (the typed fire move does not carry it across — the
    /// tree the rebase loop proves is the one the command just
    /// mutated). The advance/withhold contract and the no-setter
    /// invariant live on the type — see [`CertifiedPrior`].
    pub certified: CertifiedPrior,
    /// The rebase loop's ceiling-timer lifecycle. Module-private: the
    /// 3-state `armed-once / never (Reached, timer-still-armed)`
    /// invariant is load-bearing, so the only mutators are the typed
    /// edge-methods [`Self::arm_rebase_ceiling`] (NotStarted → Armed,
    /// arm-once) and [`Self::mark_rebase_ceiling_reached`] (Armed →
    /// Reached); the only reader is [`Self::rebase_ceiling_reached`]
    /// plus the in-module `timer_token` projection. No public setter —
    /// the [`CertifiedPrior`] / [`Self::note_effect_completion`]
    /// no-bypass discipline, applied to the ceiling.
    rebase_ceiling: RebaseCeilingState,
}

/// Post-fire phase discriminator — the structural mirror of
/// [`PreFirePhase`].
///
/// `Awaiting` has no pre-fire peer (the actuator gate);
/// `RebaseSettling ⇄ Rebasing` is the post-fire `Batching ⇄ Verifying`
/// loop, settle-spaced over the burst's `certified` proof.
///
/// `Awaiting { outstanding, gate_deadline }`: effects emitted, counter
/// decrements on each `EffectComplete` for this Profile's `DedupKey`s.
/// Reaching zero advances to `Rebasing` (or, when the burst carries
/// [`BurstFinish::Reap`], finishes the burst directly). `gate_deadline`
/// is the recovery timer for an actuator that never reports completion
/// — its expiry forces the burst into `Rebasing` (or, on a zombie
/// burst, directly into [`crate::ProfileState::Idle`] via reap).
///
/// `Rebasing` carries a [`ProbeSlot`]: the post-fire baseline-capture
/// probe's liveness *and* identity live on the phase, so a rebase in
/// flight without its correlation is unconstructable. The rebase
/// response folds the burst's `certified` N=2 proof; a `Stable`
/// verdict rebases `baseline := current` and finishes (or restarts on
/// a non-empty residual), an unstable / unread verdict loops back
/// through `RebaseSettling`.
///
/// `RebaseSettling { spacing_timer }`: the spacing wait between two
/// `Rebasing` samples — the post-fire analogue of
/// [`PreFirePhase::Batching`]. No probe is in flight (the slot lives
/// on `Rebasing`), only the settle-spacing timer, so two consecutive
/// rebase reads are separated by ≥ `settle` and a writer slower than
/// the probe round-trip but faster than `settle` cannot produce a
/// premature `Stable`. `spacing_timer` is the phase's correlation
/// token, exactly as `Batching`'s `settle_timer` is.
#[derive(Debug)]
pub enum PostFirePhase {
    Awaiting {
        outstanding: u32,
        gate_deadline: TimerId,
    },
    /// Post-fire baseline-capture probe at the anchor. The [`ProbeSlot`]
    /// holds the correlation the rebase response must echo while it is
    /// in flight; the single disarm at response dispatch is the
    /// consume-once guarantee. The variant requires the slot, so a
    /// rebase phase without a correlation is unrepresentable:
    ///
    /// ```compile_fail
    /// use specter_core::PostFirePhase;
    /// let _: PostFirePhase = PostFirePhase::Rebasing;
    /// ```
    Rebasing(ProbeSlot),
    /// Settle-spacing wait between two `Rebasing` samples — the
    /// post-fire `Batching`. `spacing_timer` is the armed spacing
    /// timer; its expiry re-enters `Rebasing` for the next sample.
    /// No [`ProbeSlot`]: no probe is in flight during the spacing
    /// window, so a stray `EffectComplete` / probe response here is a
    /// late, untracked arrival (folded to the same routing as
    /// `Rebasing`).
    RebaseSettling { spacing_timer: TimerId },
}

/// The post-fire rebase loop's ceiling-timer lifecycle — the
/// post-fire analogue of pre-fire's `burst_deadline`, made a 3-state
/// sum so the illegal combinations a `(Option<TimerId>, bool)` pair
/// would admit are unconstructable.
///
/// - [`Self::NotStarted`]: the loop has not begun (`Awaiting`). No
///   ceiling timer exists yet — arming it during `Awaiting` would let
///   a slow-but-finite command burn the rebase budget against the
///   actuator gate (a different, `gate_deadline`-bounded concern).
/// - [`Self::Armed`]: the ceiling timer is live, scheduled once at the
///   first `Awaiting → Rebasing` edge. The `RebaseSettling → Rebasing`
///   re-entry does **not** re-arm — *armed once* per loop.
/// - [`Self::Reached`]: the ceiling fired. The terminal is applied
///   with the verdict in hand at the next `Rebasing` response (the
///   forced-mirror of pre-fire `burst_deadline → forced`), so the
///   `(Reached, timer-still-armed)` state a flag-pair would make
///   representable never exists.
///
/// Module-private; mutated only through [`PostFireBurst`]'s typed
/// edge-methods (no public setter — the [`CertifiedPrior`] discipline).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RebaseCeilingState {
    NotStarted,
    Armed(TimerId),
    Reached,
}

/// Verdict of one `EffectComplete` against the post-fire counter.
///
/// Three variants, not a `bool`, because the route is resolved from
/// the same call: "decremented, still in flight" vs "last completion"
/// vs "not even Awaiting" must each be representable.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AwaitVerdict {
    /// Decremented, still `> 0` — more in flight; stay Awaiting.
    Decremented,
    /// Hit zero (pre-decrement `≤ 1`) — last completion; caller routes on.
    LastReached,
    /// Not `Active(PostFire(Awaiting))` — a late/untracked completion.
    NotAwaiting,
}

/// Verdict of one certified probe response against the carrier's
/// accumulated quiescence proof.
///
/// Three variants, not a `bool`, because the route is resolved from the
/// same fold: "two settle-spaced equal samples — fire" vs "the tree is
/// still moving — re-batch" vs "the probe could not discharge its
/// obligation — refuse to fire and surface the unread path" must each
/// be representable. The `Undischarged` payload lifts `first_unread`
/// onto the verdict so the engine switches on the verdict alone and has
/// a single diagnostic source.
///
/// Not `Copy` (unlike [`AwaitVerdict`]) — `Undischarged` carries an
/// `Arc<Path>`. Produced solely by [`CertifiedPrior::advance`]; the
/// engine maps it to a consequence privately and never reconstructs it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum QuiescenceVerdict {
    /// Two consecutive settle-spaced `Authoritative` samples hashed
    /// equal — the tree is observably quiescent; the burst may fire.
    Stable,
    /// No prior sample, or the latest `Authoritative` sample differs
    /// from the prior — the tree is still changing; re-batch.
    Unstable,
    /// The probe returned `Undischarged`: a non-observation lies on an
    /// obligation chain at `first_unread`. No verdict can be derived;
    /// the burst must not fire and the carrier proof is not advanced.
    Undischarged { first_unread: Arc<Path> },
}

/// The N=2 quiescence floor — the prior `Authoritative` sample plus the
/// one operation that folds a fresh certified response into it.
///
/// The type *is* the invariant. The inner `Option<u128>` is private and
/// there is no setter: the sole mutator is [`Self::advance`], and a
/// fresh value ([`Self::new`]) is the only other way to obtain one, so
/// no call site can write an arbitrary hash past the floor. The
/// quiescence proof a burst carries is therefore exactly "what
/// `advance` recorded", with no bypass — the
/// [`PostFireBurst::note_effect_completion`] category-(b) pattern (a
/// total edge-method owning its field's invariant), inverted to the
/// certified-sample side.
///
/// `None` until the first `Authoritative` sample, so the first verify
/// of a sequence is `Unstable` by construction. Advanced only on an
/// `Authoritative` response; an `Undischarged` response leaves it
/// untouched (an unread region must never become the comparison
/// baseline). The carrier preserves it verbatim across every phase
/// swap, so two settle-spaced samples compare to each other rather than
/// to a splice-mutated live snapshot.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct CertifiedPrior(Option<u128>);

impl CertifiedPrior {
    /// A fresh floor — no certified sample yet. `const` for the
    /// category-(a) burst constructors and fixture sites; equivalent to
    /// [`Default::default`].
    #[must_use]
    pub const fn new() -> Self {
        Self(None)
    }

    /// Fold one certified probe response into the proof and return the
    /// verdict — the floor lives here, not at the call site.
    ///
    /// - `Undischarged` ⇒ [`QuiescenceVerdict::Undischarged`] carrying
    ///   `first_unread`; the prior is **not** advanced (an unread
    ///   region must never become the comparison baseline).
    /// - `Authoritative` ⇒ compare `response` to the prior, then record
    ///   `response` as the new prior **unconditionally** (a stable pair
    ///   re-records the same value harmlessly; an unstable sample
    ///   becomes the prior the next probe compares against). `Stable`
    ///   iff a prior sample existed and hashed equal, else `Unstable`
    ///   (a `None` prior ⇒ `Unstable` by construction).
    ///
    /// Total, holds no engine state.
    #[must_use]
    pub fn advance(&mut self, authority: ProofAuthority, response: u128) -> QuiescenceVerdict {
        match authority {
            ProofAuthority::Undischarged { first_unread } => {
                QuiescenceVerdict::Undischarged { first_unread }
            }
            ProofAuthority::Authoritative => {
                let stable = self.0 == Some(response);
                self.0 = Some(response);
                if stable {
                    QuiescenceVerdict::Stable
                } else {
                    QuiescenceVerdict::Unstable
                }
            }
        }
    }
}

impl PreFireBurst {
    /// The `TimerId` armed on this burst for `kind`, or `None` if the
    /// pre-fire shape doesn't carry a slot for `kind`.
    ///
    /// Pre-fire owns:
    /// - [`TimerKind::Settle`] — lives on [`PreFirePhase::Batching`]
    ///   only; the field is absent in `Verifying`/`Draining` and the
    ///   arm returns `None`.
    /// - [`TimerKind::BurstDeadline`] — non-Optional on
    ///   [`PreFireBurst`]; always `Some(self.burst_deadline)`.
    /// - [`TimerKind::AwaitGateDeadline`] /
    ///   [`TimerKind::RebaseSettle`] / [`TimerKind::RebaseCeiling`] —
    ///   type-impossible here (these fields live on [`PostFireBurst`]
    ///   only); the arms return `None` to encode the structural
    ///   absence.
    ///
    /// Consumed via the [`ActiveBurst`] / [`ProfileState`] delegation
    /// chain by the engine's stale-timer filter; each layer only
    /// enumerates the kinds its data shape can actually carry, so the
    /// type-impossible pairs fold to `None` at the leaf without a
    /// wildcard fallthrough.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match kind {
            TimerKind::Settle => match &self.phase {
                PreFirePhase::Batching { settle_timer } => Some(*settle_timer),
                PreFirePhase::Verifying(_) | PreFirePhase::Draining => None,
            },
            TimerKind::BurstDeadline => Some(self.burst_deadline),
            TimerKind::AwaitGateDeadline | TimerKind::RebaseSettle | TimerKind::RebaseCeiling => {
                None
            }
        }
    }

    /// Typed move from pre-fire to post-fire — the fire transition.
    ///
    /// Drops, by leaving them out of the [`PostFireBurst::new`]
    /// construction this delegates to:
    /// - `burst_deadline` — lazy-dropped by
    ///   [`PostFireBurst::timer_token`]'s `None` arm once it expires
    ///   post-fire; the post-fire loop has its own ceiling.
    /// - `forced` — no fire decision left in the post-fire lifecycle.
    /// - `probe_target` — Rebasing always targets the anchor.
    /// - `last_event_time` / `dirty` — pre-fire-only event state.
    ///   Post-fire opens a *fresh, empty* `dirty` (the fire-tail
    ///   residual), not the pre-fire captured-path provenance.
    /// - `certified` — the pre-fire proof is **not** carried across:
    ///   the post-command tree the rebase loop must prove quiescent is
    ///   a different tree than the one the pre-fire carrier proved, so
    ///   `into_post_fire` opens a fresh [`CertifiedPrior`] sequence on
    ///   the post-fire burst (this no-cross-fire-carry is *why* the
    ///   move is typed).
    ///
    /// `intent` is preserved (read by `dispatch_rebase_*` for the
    /// diagnostic).
    #[must_use]
    pub fn into_post_fire(self, outstanding: u32, gate_deadline: TimerId) -> PostFireBurst {
        PostFireBurst::new(
            self.intent,
            PostFirePhase::Awaiting {
                outstanding,
                gate_deadline,
            },
            DirtyProvenance::new(),
        )
    }
}

impl PostFireBurst {
    /// The `TimerId` armed on this burst for `kind`, or `None` if the
    /// post-fire shape doesn't carry a slot for `kind`.
    ///
    /// Post-fire owns:
    /// - [`TimerKind::AwaitGateDeadline`] — lives on
    ///   [`PostFirePhase::Awaiting`]'s `gate_deadline` field; `None`
    ///   once the burst leaves `Awaiting` (the field doesn't exist on
    ///   `Rebasing` / `RebaseSettling`).
    /// - [`TimerKind::RebaseSettle`] — lives on
    ///   [`PostFirePhase::RebaseSettling`]'s `spacing_timer` field;
    ///   `None` in `Awaiting` / `Rebasing` (no spacing wait in flight).
    /// - [`TimerKind::RebaseCeiling`] — lives on the `rebase_ceiling`
    ///   lifecycle, `Some` only while `Armed`; `NotStarted` (the loop
    ///   has not begun) and `Reached` (the terminal already latched)
    ///   both yield `None`, so the just-expired ceiling id lazy-drops.
    /// - [`TimerKind::Settle`] / [`TimerKind::BurstDeadline`] —
    ///   type-impossible here (the fields were dropped at
    ///   [`PreFireBurst::into_post_fire`]); the arm returns `None`
    ///   to encode the structural absence.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match kind {
            TimerKind::AwaitGateDeadline => match &self.phase {
                PostFirePhase::Awaiting { gate_deadline, .. } => Some(*gate_deadline),
                PostFirePhase::Rebasing(_) | PostFirePhase::RebaseSettling { .. } => None,
            },
            TimerKind::RebaseSettle => match &self.phase {
                PostFirePhase::RebaseSettling { spacing_timer } => Some(*spacing_timer),
                PostFirePhase::Awaiting { .. } | PostFirePhase::Rebasing(_) => None,
            },
            TimerKind::RebaseCeiling => match self.rebase_ceiling {
                RebaseCeilingState::Armed(t) => Some(t),
                RebaseCeilingState::NotStarted | RebaseCeilingState::Reached => None,
            },
            TimerKind::Settle | TimerKind::BurstDeadline => None,
        }
    }

    /// Construct a post-fire burst — the single construction seam.
    ///
    /// Born fresh, always: `certified` is an empty [`CertifiedPrior`]
    /// (the rebase loop opens its own N=2 sequence over the
    /// post-command tree) and `rebase_ceiling` is `NotStarted` (the
    /// loop has not begun). Those two invariant-bearing fields take no
    /// parameter precisely because *no* construction path may seed them
    /// — the only mutations are the typed edge-methods, the
    /// [`CertifiedPrior`] no-bypass discipline applied to construction.
    ///
    /// Sole production caller: [`PreFireBurst::into_post_fire`] (the
    /// typed fire move). A private `rebase_ceiling` makes a struct
    /// literal unconstructable outside this module, so this is also the
    /// only cross-crate construction path (fixtures included) — there is
    /// exactly one place a `PostFireBurst` is born.
    #[must_use]
    pub const fn new(intent: BurstIntent, phase: PostFirePhase, dirty: DirtyProvenance) -> Self {
        Self {
            intent,
            phase,
            dirty,
            certified: CertifiedPrior::new(),
            rebase_ceiling: RebaseCeilingState::NotStarted,
        }
    }

    /// Arm the rebase-loop ceiling at the first `Awaiting → Rebasing`
    /// edge. `NotStarted → Armed(timer)`; returns `true` iff it armed.
    ///
    /// **Arm-once.** The `RebaseSettling → Rebasing` re-entry finds
    /// `Armed` (or `Reached`) and is a `false` no-op — the ceiling
    /// bounds the loop from its *start*, not from each sample. The
    /// caller (`transition_to_rebasing`) schedules the timer only when
    /// this returns `true`, so a redundant heap entry is never minted.
    ///
    /// Category-(b) typed edge-method: total, no public setter, returns
    /// the edge — the sibling of [`Self::note_effect_completion`] /
    /// [`CertifiedPrior::advance`] on the ceiling field.
    #[must_use]
    pub const fn arm_rebase_ceiling(&mut self, timer: TimerId) -> bool {
        if matches!(self.rebase_ceiling, RebaseCeilingState::NotStarted) {
            self.rebase_ceiling = RebaseCeilingState::Armed(timer);
            true
        } else {
            false
        }
    }

    /// Latch the rebase-loop ceiling terminal. `Armed → Reached`;
    /// returns `true` iff it latched (the prior state was `Armed`).
    ///
    /// The terminal is *recorded*, not *acted on*, here: the forced
    /// rebase is applied with the verdict in hand at the next
    /// `Rebasing` response (the forced-mirror of pre-fire
    /// `burst_deadline → forced`), so the illegal `(Reached,
    /// timer-still-armed)` state never exists. A `false` return
    /// (`NotStarted` / already `Reached`) is the loud-regression signal
    /// the caller debug-asserts against — `RebaseCeiling` only fires
    /// from `Armed` (`timer_token` filters the other states).
    ///
    /// Category-(b) sibling of [`Self::arm_rebase_ceiling`].
    #[must_use]
    pub const fn mark_rebase_ceiling_reached(&mut self) -> bool {
        if let RebaseCeilingState::Armed(_) = self.rebase_ceiling {
            self.rebase_ceiling = RebaseCeilingState::Reached;
            true
        } else {
            false
        }
    }

    /// Whether the rebase-loop ceiling has latched (`Reached`). The
    /// sole public read of the private `rebase_ceiling` lifecycle —
    /// `dispatch_rebase_ok` consults it to route the verdict to a
    /// looping vs a terminal consequence. `NotStarted` / `Armed` are
    /// both "ceiling not yet reached" ⇒ `false`.
    #[must_use]
    pub const fn rebase_ceiling_reached(&self) -> bool {
        matches!(self.rebase_ceiling, RebaseCeilingState::Reached)
    }

    /// Apply one `EffectComplete`, returning the zero-edge verdict. The
    /// sole in-life mutator of [`PostFirePhase::Awaiting`]'s
    /// `outstanding`: floor and decrement live here on the owner — a
    /// total fn with no public setter that returns the edge, so the
    /// invariant cannot be enforced at a distance. `Rebasing` /
    /// `RebaseSettling` ⇒ [`AwaitVerdict::NotAwaiting`] (the counter
    /// drained at the `Awaiting → Rebasing` edge; a completion arriving
    /// in the rebase loop is a late, untracked arrival). Underflow
    /// (more completions than emitted Effects) trips a `debug_assert!`,
    /// saturates in release.
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
            PostFirePhase::Rebasing(_) | PostFirePhase::RebaseSettling { .. } => {
                AwaitVerdict::NotAwaiting
            }
        }
    }

    /// Typed move from post-fire back to a fresh pre-fire `Batching`
    /// burst — the symmetric inverse of [`PreFireBurst::into_post_fire`].
    ///
    /// Consumes the post-fire burst at the rebase-ok boundary and re-arms
    /// a Standard debounce burst, moving the fire-tail residual `dirty`
    /// provenance over whole: the events `absorb_event_into_fire_tail`
    /// captured while the rebase probe was already in flight. Without
    /// this the residual has no consumer — it drops when the post-fire
    /// burst is torn down, so a descendant change that landed during the
    /// rebase round-trip is seen only by the next unrelated event. The
    /// move keeps the captured paths intact, so the restarted Standard
    /// burst's first verify obligates over them.
    ///
    /// **In-place move, never finish-then-start.** The typed
    /// `PostFire → PreFire` move preserves the watched anchor: it
    /// neither installs nor releases a contribution, so the restarted
    /// burst keeps the original burst's kernel-watch state without a
    /// finish/start round-trip. The single balancing `Unwatch` (if
    /// any) still runs at the restarted burst's eventual reap.
    ///
    /// **Origin-agnostic.** `intent` is *set* (not inherited) to
    /// `Standard` because a restarted debounce burst *is* Standard by
    /// definition. This is precisely where a Seed-origin fire-tail
    /// residual (Seed drift → fire → rebase, with events absorbed while
    /// the rebase probe was in flight) rejoins the Standard debounce
    /// lifecycle rather than being dropped — the closed Seed-residual
    /// event-loss. The reconfirm machinery is a fresh query over the
    /// live tree, not a refcount, so there is no per-origin balance to
    /// preserve and no origin gate.
    ///
    /// `last_event_time` reckons from `now` — the rebase-response
    /// instant, not the absorbed events' (those timestamps are discarded
    /// at absorb). The restarted burst's settle window therefore carries
    /// a bounded ≤ one-`settle` extra re-fire latency in exchange for
    /// never losing the residual. `probe_target` is the anchor
    /// placeholder, overwritten by the next `transition_to_verifying`
    /// exactly as in a fresh `start_standard_burst`. The post-fire
    /// `certified` proof and the `rebase_ceiling` lifecycle are dropped
    /// by omission — they are post-fire-only, and the restarted pre-fire
    /// burst opens its own fresh [`CertifiedPrior`] N=2 sequence and
    /// pre-fire `burst_deadline`, exactly as a fresh
    /// `start_standard_burst`.
    #[must_use]
    pub fn into_pre_fire_residual(
        self,
        burst_deadline: TimerId,
        settle_timer: TimerId,
        anchor: ResourceId,
        now: Instant,
    ) -> PreFireBurst {
        debug_assert!(
            !self.dirty.is_empty(),
            "into_pre_fire_residual: empty residual — the restart has no \
             seed; the caller must gate on a non-empty fire-tail residual",
        );
        let residual = self.dirty;
        PreFireBurst {
            burst_deadline,
            phase: PreFirePhase::Batching { settle_timer },
            intent: BurstIntent::Standard,
            forced: false,
            dirty: residual,
            probe_target: anchor,
            last_event_time: Some(now),
            certified: CertifiedPrior::new(),
        }
    }
}

impl ActiveBurst {
    /// Delegate to the lifecycle-side projection. [`Self::PreFire`]
    /// and [`Self::PostFire`] carry disjoint timer fields by
    /// construction; this dispatcher routes to whichever side the
    /// burst is currently on without re-enumerating the
    /// type-impossible cross-pairs.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match self {
            Self::PreFire(pre) => pre.timer_token(kind),
            Self::PostFire(post) => post.timer_token(kind),
        }
    }

    /// The burst's [`BurstIntent`]. `intent` is a field on **both**
    /// [`PreFireBurst`] and [`PostFireBurst`] (it survives the fire
    /// transition); this is the lifecycle-side projection that reads it
    /// without re-enumerating the cross-pairs — same wildcard-free
    /// PreFire/PostFire shape as [`Self::timer_token`]. Sole consumer:
    /// [`ProfileState::in_active_standard_burst`].
    #[must_use]
    pub const fn intent(&self) -> BurstIntent {
        match self {
            Self::PreFire(pre) => pre.intent,
            Self::PostFire(post) => post.intent,
        }
    }

    /// Delegate to the post-fire counter; [`Self::PreFire`] carries no
    /// fire, folding to [`AwaitVerdict::NotAwaiting`] — same shape-fold
    /// as [`Self::timer_token`], no wildcard.
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        match self {
            Self::PostFire(post) => post.note_effect_completion(),
            Self::PreFire(_) => AwaitVerdict::NotAwaiting,
        }
    }

    /// Delegate the quiescence fold to whichever carrier holds the
    /// burst. Both [`Self::PreFire`] and [`Self::PostFire`] carry their
    /// own [`CertifiedPrior`] N=2 proof — the pre-fire verify loop and
    /// the post-fire rebase loop are the same loop shape over disjoint
    /// carriers — so an Active burst *always* folds a verdict (`Some`
    /// on both arms). The `None ⇔ Idle | Pending` boundary lives one
    /// layer up at [`ProfileState::advance_quiescence`] (no burst ⇒ no
    /// proof to fold), not here.
    ///
    /// The post-fire arm is a legitimate fold site, not a weaker one:
    /// the verdict folds the freshly-walked *response* hash through the
    /// shared [`CertifiedPrior`] floor (never `current` / `baseline`),
    /// so the rebase loop has the same splice-independence the pre-fire
    /// loop does — the floor *carries* the property rather than each
    /// carrier re-deriving it. Same wildcard-free PreFire/PostFire
    /// delegation shape as [`Self::timer_token`].
    #[must_use]
    pub fn advance_quiescence(
        &mut self,
        authority: ProofAuthority,
        response: u128,
    ) -> Option<QuiescenceVerdict> {
        match self {
            Self::PreFire(pre) => Some(pre.certified.advance(authority, response)),
            Self::PostFire(post) => Some(post.certified.advance(authority, response)),
        }
    }

    /// Latch the post-fire rebase-loop ceiling terminal — delegate to
    /// the post-fire side's owner-resident edge. [`Self::PreFire`]
    /// carries no ceiling (the pre-fire loop bounds itself with
    /// `burst_deadline`), so it folds to `false` — same wildcard-free
    /// PreFire/PostFire delegation as [`Self::note_effect_completion`].
    /// The floor (the `Armed → Reached` invariant, no public setter)
    /// lives on the owner, [`PostFireBurst::mark_rebase_ceiling_reached`].
    #[must_use]
    pub const fn mark_rebase_ceiling_reached(&mut self) -> bool {
        match self {
            Self::PostFire(post) => post.mark_rebase_ceiling_reached(),
            Self::PreFire(_) => false,
        }
    }
}

/// Burst-finish directive — *what does the Profile do at burst-end?*
///
/// Carried as the second payload of [`ProfileState::Active`]. Default
/// [`Self::ReturnToIdle`]: the burst completes, the Profile returns to
/// [`ProfileState::Idle`], and the next `FsEvent` may start a fresh
/// burst. [`Self::Reap`] flips the directive: the active burst still
/// runs to completion (so the burst-end Draining-sweep reconfirm runs
/// before the Profile leaves the map), but `finish_burst_to_idle` then
/// routes through `reap_profile` instead of returning the Profile to
/// Idle.
///
/// **Why a payload, not a parallel field on Profile.** The directive
/// is *only* meaningful inside an Active burst. Encoding it as a `bool`
/// alongside [`ProfileState`] (the prior `Profile.reap_pending`) made
/// `(Idle, true)` representable but structurally illegal —
/// discipline enforced by convention rather than by the type system.
/// Folding the directive into the `Active` variant's payload
/// type-bans the illegal combination by construction.
///
/// **Writers.**
/// - [`ProfileState::mark_active_for_reap`] flips
///   [`Self::ReturnToIdle`] → [`Self::Reap`]. Sole callers:
///   `detach_sub_inner` (last Sub detached mid-burst) and
///   `on_anchor_terminal_all_dynamic` (all-dynamic Promoter teardown
///   converged on a still-Active Profile).
/// - [`ProfileState::clear_active_reap`] flips
///   [`Self::Reap`] → [`Self::ReturnToIdle`]. Sole caller:
///   `attach_sub_inner`'s zombie-revival arm — a fresh Sub joining a
///   zombie Profile resurrects it under the new Sub set.
///
/// **Readers.** `emit_effects` (suppress emission), `on_effect_complete`
/// (route last completion to reap vs rebase), `handle_gate_deadline`
/// (route zombie burst directly to finish), and `finish_burst_to_idle`
/// (post-drain reap dispatch).
///
/// The directive is preserved across the fire transition
/// (`PreFireBurst::into_post_fire`'s caller carries it through the
/// `mem::replace`) and across phase transitions within pre-fire
/// (`transition_to_verifying`, `_draining`, etc.) — these helpers
/// mutate the burst's inner state without touching the `Active`
/// variant's outer shape.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum BurstFinish {
    /// Default. Burst-end transitions the Profile to [`ProfileState::Idle`].
    #[default]
    ReturnToIdle,
    /// Burst-end reaps the Profile via `reap_profile`. Set by
    /// [`ProfileState::mark_active_for_reap`]; cleared by
    /// [`ProfileState::clear_active_reap`].
    Reap,
}

/// Where should a Profile land when its last Sub detaches?
///
/// Computed by [`ProfileState::detach_lifecycle`] at the moment the
/// last Sub is removed. The two arms encode the only paths that
/// preserve the burst-end drain ordering:
///
/// - [`Self::ReapNow`]: the Profile is [`ProfileState::Idle`] or
///   [`ProfileState::Pending`] — there is no burst to drain.
///   `reap_profile` runs immediately, releasing the descent prefix
///   (Pending) or the anchor contribution (Idle / materialized).
/// - [`Self::DeferToBurstEnd`]: the Profile is [`ProfileState::Active`]
///   — a burst is in flight whose burst-end Draining-sweep reconfirm
///   must run before reap. The caller flips
///   [`BurstFinish::Reap`] (via [`ProfileState::mark_active_for_reap`])
///   so `finish_burst_to_idle` routes through `reap_profile` once the
///   burst converges.
///
/// Lives in core (not in the engine) because the classification is a
/// projection over [`ProfileState`] — the state knows what its
/// last-Sub-detached outcome must be.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DetachLifecycle {
    /// Profile has no burst — reap synchronously.
    ReapNow,
    /// Profile has an Active burst — mark for reap, drain runs first.
    DeferToBurstEnd,
}

/// Trigger that drove a Profile's reap, threaded into
/// [`crate::Diagnostic::ProfileReaped`]. Two paths converge on the
/// same `reap_profile` machinery:
///
/// - [`Self::Immediate`]: `detach_sub_inner` on an Idle/Pending Profile
///   whose last Sub just detached. No burst to wait on, so reap runs
///   inline. Also reached by `on_anchor_terminal_all_dynamic`'s
///   non-Active arm.
/// - [`Self::DeferredFromBurst`]: `finish_burst_to_idle` honouring the
///   [`BurstFinish::Reap`] directive at burst-end. The Profile spent
///   time as a zombie burst before reaching reap.
///
/// Operators distinguish the two for incident triage: a flood of
/// `DeferredFromBurst` reaps signals churn on Active Profiles, whereas
/// `Immediate` is the steady-state detach path.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ReapTrigger {
    /// Reap runs inline at refcount→0 — no burst to drain.
    Immediate,
    /// Reap runs at burst-end via [`BurstFinish::Reap`] honour.
    DeferredFromBurst,
}

/// Profile state machine.
///
/// Three lifecycle states, mutually exclusive by construction:
/// - `Idle`: no probe in flight, no burst, no descent. Reads/writes baseline
///   and current as-is.
/// - `Pending(DescentState)`: anchor doesn't yet exist on disk; the engine
///   is probing the deepest existing prefix and advancing one path
///   component per response. The anchor's `Profile.resource` slot is
///   `DescentScaffold`-roled and carries no `watch_demand` from this
///   Profile (the prefix carries the `+1`). See `DescentState` invariants.
/// - `Active(ActiveBurst, BurstFinish)`: anchor is materialized; a
///   stability burst is in flight. The second payload is the
///   post-burst directive — see [`BurstFinish`] for the four-site
///   reader / two-site writer surface that drives it. Default
///   ([`BurstFinish::ReturnToIdle`]) returns the Profile to Idle at
///   burst-end; [`BurstFinish::Reap`] dispatches `reap_profile`
///   instead. The pre-Phase-4 shape carried [`BurstFinish::Reap`] as a
///   `pub` boolean on [`Profile`] (the now-deleted `reap_pending`
///   field); the variant payload structurally bans the illegal
///   `(Idle, reap_pending = true)` combination.
///
/// I5 (at most one outstanding probe per Profile) is a
/// **representability** property: the in-flight probe's liveness *and*
/// identity live on the state itself, in the single [`ProbeSlot`] of
/// whichever carrier the Profile currently is — the `Pending` descent
/// slot, the `Active(PreFire(Verifying))` slot, or the
/// `Active(PostFire(Rebasing))` slot. One Profile is exactly one of
/// these carriers, so it holds exactly one slot and two simultaneous
/// probes are unconstructable. The response handler routes by state,
/// gates on [`Self::probe_correlation`], and consumes by disarming
/// that slot once via [`Self::take_probe`] — the structural
/// consume-once guarantee, with no separate side-table to drift
/// against.
#[derive(Debug, Default)]
pub enum ProfileState {
    #[default]
    Idle,
    /// Pending-path descent in flight. The anchor (`Profile.resource`) is
    /// `DescentScaffold`-roled and carries no `watch_demand` from this
    /// Profile; `DescentState.current_prefix` does. When the anchor
    /// materializes (descent's last component arrives) the engine
    /// transitions Pending → Idle (releasing the prefix's contribution and
    /// bumping the anchor's), then immediately Idle → `Active(PreFire(Seed), …)`
    /// via `start_seed_burst`.
    Pending(DescentState),
    /// Stability burst in flight, with a post-burst directive. See
    /// [`BurstFinish`] for the directive's semantics; the default
    /// ([`BurstFinish::ReturnToIdle`]) is set at burst-launch and the
    /// engine flips to [`BurstFinish::Reap`] via
    /// [`Self::mark_active_for_reap`] when the Profile loses its last
    /// Sub mid-burst.
    Active(ActiveBurst, BurstFinish),
}

impl ProfileState {
    /// Variant-tag projection used by diagnostics that need to name
    /// "what state was the Profile actually in" without copying the
    /// payload. The four discriminants line up with the four routing
    /// classes burst helpers care about: `Idle` (pre-burst), `Pending`
    /// (descent in flight), `ActivePreFire` (settling / verifying /
    /// draining), `ActivePostFire` (awaiting / rebasing). The fire
    /// transition (`PreFire → PostFire`) is the only edge that crosses
    /// the third-vs-fourth discriminator, which is exactly the same
    /// boundary the [`ActiveBurst`] type split enforces.
    ///
    /// [`BurstFinish`] is intentionally collapsed at this projection —
    /// zombie and live bursts share routing class because every
    /// burst-helper that consults the discriminant routes identically
    /// for both. Readers that need to distinguish call
    /// [`Self::burst_finish`].
    #[must_use]
    pub const fn discriminant(&self) -> ProfileStateDiscriminant {
        match self {
            Self::Idle => ProfileStateDiscriminant::Idle,
            Self::Pending(_) => ProfileStateDiscriminant::Pending,
            Self::Active(ActiveBurst::PreFire(_), _) => ProfileStateDiscriminant::ActivePreFire,
            Self::Active(ActiveBurst::PostFire(_), _) => ProfileStateDiscriminant::ActivePostFire,
        }
    }

    /// Read the burst-finish directive. `Some(_)` only when the
    /// Profile is in an Active burst; `None` for Idle and Pending
    /// (where the directive is structurally meaningless).
    ///
    /// Read by `emit_effects` (suppress emission on zombie),
    /// `on_effect_complete` (route last completion), `handle_gate_deadline`
    /// (zombie-skip), and indirectly by every test that previously
    /// inspected `Profile.reap_pending`.
    #[must_use]
    pub const fn burst_finish(&self) -> Option<BurstFinish> {
        match self {
            Self::Active(_, finish) => Some(*finish),
            Self::Idle | Self::Pending(_) => None,
        }
    }

    /// Classify the reap path when a Profile's last Sub detaches. Called
    /// by `detach_sub_inner` once no Subs remain on the Profile — the
    /// result chooses between immediate `reap_profile` and
    /// deferred-to-burst-end via [`Self::mark_active_for_reap`].
    ///
    /// Lives on [`ProfileState`] because the choice is a pure
    /// projection over the variant — the engine has no other input
    /// that influences the decision.
    #[must_use]
    pub const fn detach_lifecycle(&self) -> DetachLifecycle {
        match self {
            Self::Idle | Self::Pending(_) => DetachLifecycle::ReapNow,
            Self::Active(_, _) => DetachLifecycle::DeferToBurstEnd,
        }
    }

    /// Flip an Active burst's [`BurstFinish`] from
    /// [`BurstFinish::ReturnToIdle`] to [`BurstFinish::Reap`].
    /// Returns `true` iff the state was [`Self::Active`] and the
    /// directive was set (already-`Reap` returns `true` and is a
    /// silent no-op — idempotent under re-entry).
    ///
    /// **Preconditions, by intent.** Callers have already established
    /// that the state is Active (via [`Self::detach_lifecycle`] or a
    /// `matches!` guard). The `bool` return surfaces "did the flip
    /// land" so callers can `debug_assert!` against a future routing
    /// breach.
    ///
    /// **Sole writers.** `detach_sub_inner` (refcount→0 on Active)
    /// and `on_anchor_terminal_all_dynamic` (all-dynamic Promoter
    /// teardown on Active). No other site has a legitimate need to
    /// mark a burst for reap.
    #[must_use]
    pub const fn mark_active_for_reap(&mut self) -> bool {
        if let Self::Active(_, finish) = self {
            *finish = BurstFinish::Reap;
            true
        } else {
            false
        }
    }

    /// Flip an Active burst's [`BurstFinish`] back from
    /// [`BurstFinish::Reap`] to [`BurstFinish::ReturnToIdle`].
    /// Returns `true` iff the state was [`Self::Active`] *and* the
    /// prior directive was `Reap` — i.e., a zombie burst was revived.
    /// `false` on `(Active, ReturnToIdle)` (normal join — nothing to
    /// clear), Idle, and Pending.
    ///
    /// **Why the precondition narrows to current-Reap.** The clear
    /// path's *only* legitimate trigger is zombie revival in
    /// `attach_sub_inner`. Returning `false` on a non-Reap Active
    /// keeps the bool return informative: the caller branches on it
    /// to emit the [`crate::Diagnostic::ReapPendingCancelled`]
    /// diagnostic and run the post-revival cleanup
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

    /// The live `TimerId` for the requested `kind` slot, or `None` if
    /// the state owns no timer of that kind right now.
    ///
    /// Only [`Self::Active`] Profiles schedule timers — [`Self::Idle`]
    /// and [`Self::Pending`] (descent) own none and return `None` for
    /// every kind. The `Active` arm delegates to
    /// [`ActiveBurst::timer_token`], which in turn routes to whichever
    /// burst-side type ([`PreFireBurst`] or [`PostFireBurst`]) actually
    /// carries the field. Each layer only enumerates the kinds its
    /// data shape can carry, so type-impossible pairs fold to `None`
    /// at the leaf without an explicit wildcard arm.
    ///
    /// Consumed by the engine's `pop_expired` and `on_timer_expired`
    /// gates to distinguish a live timer from a stale heap orphan
    /// (cancelled because the Profile's burst was reset between
    /// `schedule` and pop).
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match self {
            Self::Active(burst, _) => burst.timer_token(kind),
            Self::Idle | Self::Pending(_) => None,
        }
    }

    /// Delegate to the active burst's post-fire counter; `Idle` /
    /// `Pending` own none and fold to [`AwaitVerdict::NotAwaiting`].
    /// Same layered, wildcard-free delegation as [`Self::timer_token`].
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        match self {
            Self::Active(burst, _) => burst.note_effect_completion(),
            Self::Idle | Self::Pending(_) => AwaitVerdict::NotAwaiting,
        }
    }

    /// Delegate the quiescence fold to the active burst. This is the
    /// **sole** `None` site in the chain: `Active` (either side of the
    /// fire boundary) always folds a verdict through its own
    /// [`CertifiedPrior`], so `None ⇔ Idle | Pending` — "no Active
    /// burst", not "not in pre-fire". Same layered, wildcard-free
    /// delegation as [`Self::note_effect_completion`]. The verify /
    /// rebase chokes run only with the carrier proven `Active`, so a
    /// `None` reaching them is a loud structural regression (the
    /// Profile left the burst between the gate and the fold), not a
    /// pre-fire-vs-post-fire distinction.
    #[must_use]
    pub fn advance_quiescence(
        &mut self,
        authority: ProofAuthority,
        response: u128,
    ) -> Option<QuiescenceVerdict> {
        match self {
            Self::Active(burst, _) => burst.advance_quiescence(authority, response),
            Self::Idle | Self::Pending(_) => None,
        }
    }

    /// Latch the active burst's post-fire rebase-loop ceiling terminal.
    /// `Idle` / `Pending` own no burst (so `false`); the burst routes
    /// `PostFire` to its owner-resident edge and `PreFire` to `false`.
    /// Same layered, wildcard-free delegation as
    /// [`Self::note_effect_completion`]; the no-public-setter floor
    /// lives on [`PostFireBurst::mark_rebase_ceiling_reached`].
    #[must_use]
    pub const fn mark_rebase_ceiling_reached(&mut self) -> bool {
        match self {
            Self::Active(burst, _) => burst.mark_rebase_ceiling_reached(),
            Self::Idle | Self::Pending(_) => false,
        }
    }

    /// True iff the state is `Active(PreFire(Draining))`. The
    /// reconfirm cascade (the `Draining → Verifying` re-probe) keys off
    /// this predicate: at every `finish_burst_to_idle` the engine
    /// sweeps the Draining Profiles and reconfirms each whose
    /// covered-descendant query has gone false. `Idle` and `Pending`
    /// are structurally not-Draining; the post-fire arm and the other
    /// pre-fire phases (Batching, Verifying) also return `false`.
    #[must_use]
    pub const fn is_draining(&self) -> bool {
        match self {
            Self::Active(ActiveBurst::PreFire(pre), _) => {
                matches!(pre.phase, PreFirePhase::Draining)
            }
            Self::Idle | Self::Pending(_) | Self::Active(ActiveBurst::PostFire(_), _) => false,
        }
    }

    /// True iff the state is an Active **Standard** burst, in *any*
    /// phase — pre-fire (`Batching | Verifying | Draining`) or post-fire
    /// (`Awaiting | Rebasing | RebaseSettling`). Wildcard-free,
    /// mirroring [`Self::is_draining`].
    ///
    /// This is the per-Profile half of the derived replacement for the
    /// old `dirty_descendants` refcount. The refcount's `+1`
    /// (`start_standard_burst`) / `-1` (`finish_burst_to_idle`) bracketed
    /// a Standard burst's *entire* lifetime — pre-fire through post-fire,
    /// across a fire-tail residual restart (the `+1` was held, never
    /// re-taken). Spanning both pre- and post-fire here is exactly that
    /// lifetime evaluated fresh: a Standard descendant counts as covering
    /// its ancestor from burst start until `finish_burst_to_idle`,
    /// whatever phase it is in. A restarted residual burst is
    /// `intent: Standard` by construction
    /// ([`PostFireBurst::into_pre_fire_residual`]), so it stays counted
    /// with no special accounting. Seed bursts return `false` — they
    /// never contributed to the old refcount.
    ///
    /// Read through [`crate::ProfileState::in_active_standard_burst`] →
    /// `.state()` exactly as [`Self::is_draining`] is (no `Profile`
    /// delegate — the accessor convention is `.state().<pred>()`).
    #[must_use]
    pub const fn in_active_standard_burst(&self) -> bool {
        match self {
            Self::Active(burst, _) => matches!(burst.intent(), BurstIntent::Standard),
            Self::Idle | Self::Pending(_) => false,
        }
    }

    /// Borrow the descent payload if the state is currently
    /// [`Self::Pending`]. `None` for [`Self::Idle`] and
    /// [`Self::Active`] — the descent payload only lives in the
    /// `Pending` variant.
    ///
    /// Symmetric with [`crate::PromoterState::descent_state`]; the
    /// engine's owner-polymorphic `descent_state` dispatcher routes
    /// to either projection through [`crate::ProbeOwner`].
    #[must_use]
    pub const fn descent_state(&self) -> Option<&DescentState> {
        match self {
            Self::Pending(d) => Some(d),
            Self::Idle | Self::Active(_, _) => None,
        }
    }

    /// Mutable counterpart to [`Self::descent_state`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        match self {
            Self::Pending(d) => Some(d),
            Self::Idle | Self::Active(_, _) => None,
        }
    }

    /// The correlation of this state's in-flight probe, or `None` if
    /// the carrier holds none. A total projection over the state space:
    /// the three probe-bearing carriers — a `Pending` descent, an
    /// `Active(PreFire(Verifying))`, an `Active(PostFire(Rebasing))` —
    /// answer from their armed slot; every other state (including a
    /// disarmed slot) yields `None`. Owner-symmetric with
    /// [`crate::PromoterState::probe_correlation`].
    #[must_use]
    pub const fn probe_correlation(&self) -> Option<ProbeCorrelation> {
        match self {
            Self::Active(ActiveBurst::PreFire(burst), _) => match &burst.phase {
                PreFirePhase::Verifying(slot) => slot.correlation(),
                PreFirePhase::Batching { .. } | PreFirePhase::Draining => None,
            },
            Self::Active(ActiveBurst::PostFire(burst), _) => match &burst.phase {
                PostFirePhase::Rebasing(slot) => slot.correlation(),
                PostFirePhase::Awaiting { .. } | PostFirePhase::RebaseSettling { .. } => None,
            },
            Self::Pending(d) => d.probe_correlation(),
            Self::Idle => None,
        }
    }

    /// Disarm whichever probe-bearing carrier this state holds and
    /// return the prior correlation — the single state-level consume.
    /// Total: the three probe-bearing carriers (`Pending` descent,
    /// `Active(PreFire(Verifying))`, `Active(PostFire(Rebasing))`) disarm
    /// their slot; every other state (including an already-disarmed
    /// slot) is a `None` no-op. The disarm leaves the carrier's variant
    /// intact — only the slot empties — so a route computed before this
    /// call stays valid after it. Owner-symmetric with
    /// [`crate::PromoterState::take_probe`].
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        match self {
            Self::Active(ActiveBurst::PreFire(burst), _) => match &mut burst.phase {
                PreFirePhase::Verifying(slot) => slot.disarm(),
                PreFirePhase::Batching { .. } | PreFirePhase::Draining => None,
            },
            Self::Active(ActiveBurst::PostFire(burst), _) => match &mut burst.phase {
                PostFirePhase::Rebasing(slot) => slot.disarm(),
                PostFirePhase::Awaiting { .. } | PostFirePhase::RebaseSettling { .. } => None,
            },
            Self::Pending(d) => d.disarm_probe(),
            Self::Idle => None,
        }
    }
}

/// Variant tag for [`ProfileState`], carried on diagnostics that report
/// state-machine routing breaches without copying the payload.
///
/// The four variants match the four routing classes the engine's burst
/// helpers branch on. They are coarser than the full state enum
/// (`Active(PreFire(Batching{settle_timer}))` collapses to
/// `ActivePreFire`) — sufficient for operator triage, and stable
/// against future phase additions.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ProfileStateDiscriminant {
    /// [`ProfileState::Idle`].
    Idle,
    /// [`ProfileState::Pending`].
    Pending,
    /// [`ProfileState::Active`] with [`ActiveBurst::PreFire`].
    ActivePreFire,
    /// [`ProfileState::Active`] with [`ActiveBurst::PostFire`].
    ActivePostFire,
}

/// State for a Profile undergoing pending-path descent.
///
/// Lives inline on `ProfileState::Pending` for the duration of descent.
///
/// Invariants:
/// - `current_prefix` carries a `+1` `watch_demand` contribution from this
///   Profile (added at descent registration / advancement; dropped at
///   descent end or rewind).
/// - [`DescentRemaining`] is non-empty by type construction — the anchor
///   itself is the last component, and descent transitions Pending → Idle
///   on materialization rather than emptying the path.
///
/// I5 ("at most one outstanding probe per descent") is a
/// representability property: the descent probe's liveness *and*
/// identity live in `probe`, a single [`ProbeSlot`] on this payload.
/// An armed slot is a probe in flight; an empty slot is descent
/// awaiting the next structural event with nothing out. One descent
/// holds exactly one slot, so two simultaneous descent probes are
/// unconstructable.
#[derive(Debug)]
pub struct DescentState {
    /// Deepest existing ancestor currently Watched. The Profile
    /// contributes `+1` to this Resource's `watch_demand`. Module-private:
    /// read via [`Self::current_prefix`], moved via [`Self::advance_to`].
    current_prefix: ResourceId,
    /// Path components from `current_prefix` (exclusive) down to the
    /// anchor (inclusive). Non-empty by type construction;
    /// single-component segments (no `/`). Module-private: reached via
    /// [`Self::remaining_components`] / [`Self::remaining_components_mut`].
    remaining_components: DescentRemaining,
    /// The descent probe slot — a linear [`ProbeSlot`]. Armed while a
    /// probe is in flight at `current_prefix` (carrying the correlation
    /// the response must echo); empty while descent awaits the next
    /// structural event. Module-private: the linear protocol is the
    /// only access path — [`Self::arm_probe`] (mint),
    /// [`Self::probe_correlation`] (read), [`Self::disarm_probe`]
    /// (consume). It cannot be cloned, so it is consumed where it lives.
    probe: ProbeSlot,
}

impl DescentState {
    /// Construct a fresh descent payload. Sole producer pattern used
    /// by `materialize_path_or_pending` (Profile pending arm), the
    /// Promoter attach path's pending arm, and the recovery / rewind
    /// flows in `engine::descent` that re-enter `Pending` after an
    /// anchor-terminal event.
    ///
    /// Field-private; callers route through this constructor so the
    /// invariants on `current_prefix` (Watched, refcounted),
    /// `remaining_components` (non-empty by [`DescentRemaining`]'s
    /// own constructor), and `probe` (the descent's single in-flight
    /// slot) are pinned at a single boundary. Every fresh descent
    /// entry mints a correlation and emits a probe, so an honest
    /// constructor takes the `probe` slot — typically
    /// [`ProbeSlot::armed`] with the just-minted correlation. The
    /// engine's refcount setup runs around this constructor (the
    /// contribution at `current_prefix` is installed by `add_watch`
    /// separately).
    #[must_use]
    pub const fn new(
        current_prefix: ResourceId,
        remaining_components: DescentRemaining,
        probe: ProbeSlot,
    ) -> Self {
        Self {
            current_prefix,
            remaining_components,
            probe,
        }
    }

    /// The deepest currently-Watched ancestor on the descent path.
    /// Carries this Profile / Promoter's `+1 STRUCTURE`
    /// [`crate::ContribKey::ProfileDescent`] /
    /// [`crate::ContribKey::PromoterPrefix`] contribution.
    #[must_use]
    pub const fn current_prefix(&self) -> ResourceId {
        self.current_prefix
    }

    /// Read-only handle to the remaining-path-component chain.
    #[must_use]
    pub const fn remaining_components(&self) -> &DescentRemaining {
        &self.remaining_components
    }

    /// Mutable handle to the remaining-path-component chain. Sole
    /// callers are the engine's descent dispatcher
    /// (`engine::descent::advance_descent` consumes the head via
    /// [`DescentRemaining::advance`]) and the rewind path
    /// (`dispatch_descent_vanished` re-injects the prefix's segment
    /// via [`DescentRemaining::prepend`]).
    pub const fn remaining_components_mut(&mut self) -> &mut DescentRemaining {
        &mut self.remaining_components
    }

    /// Rewrite the descent's current prefix. Used by the engine's
    /// descent dispatcher on forward advance (the prior prefix's
    /// `Ok` response routes here with the newly-Watched child) and
    /// by the rewind path (`Vanished` on the prefix routes here with
    /// the parent that just took over the descent's watch).
    ///
    /// Pairs with the engine's `add_watch` / `sub_watch` calls that
    /// maintain the `+1 STRUCTURE` contribution at the new and old
    /// prefixes respectively; the typed mutator pins that the field
    /// only moves under refcount-aware control.
    pub const fn advance_to(&mut self, new_prefix: ResourceId) {
        self.current_prefix = new_prefix;
    }

    /// Arm the descent's single probe slot with a freshly-minted
    /// correlation — the **mint** edge of the descent's linear-probe
    /// protocol. The engine calls this when re-probing in place
    /// (forward advance, rewind, event re-trigger); fresh-descent entry
    /// instead constructs the slot armed via [`Self::new`]. Asserts the
    /// slot was empty (the response handler or cancel path disarmed it
    /// first) — a double-arm would orphan the prior correlation.
    pub fn arm_probe(&mut self, correlation: ProbeCorrelation) {
        self.probe.arm(correlation, ());
    }

    /// Identity of the descent's in-flight probe, or `None` if idle —
    /// the **read** edge of the linear-probe protocol.
    /// [`crate::ProfileState::probe_correlation`] /
    /// [`crate::PromoterState::probe_correlation`] delegate here for
    /// their descent carrier rather than reaching the private slot.
    #[must_use]
    pub(crate) const fn probe_correlation(&self) -> Option<ProbeCorrelation> {
        self.probe.correlation()
    }

    /// Consume the descent's probe: disarm the slot and return the
    /// prior correlation (`None` if already idle) — the **consume**
    /// edge of the linear-probe protocol, dual of [`Self::arm_probe`].
    ///
    /// Crate-internal by design. The engine-facing "single consume per
    /// owner" law remains the `pub`
    /// [`crate::ProfileState::take_probe`] /
    /// [`crate::PromoterState::take_probe`]; both delegate their
    /// descent arm here, so the consume routes through the typed
    /// protocol instead of a raw field and `probe` stays
    /// module-private. Routing-once is unaffected — the engine still
    /// sees exactly one consume entry point per owner.
    pub(crate) const fn disarm_probe(&mut self) -> Option<ProbeCorrelation> {
        self.probe.disarm()
    }
}

/// Path-component chain from a descent's `current_prefix` down to the
/// anchor.
///
/// Non-emptiness is a type-level invariant: the sole constructor
/// [`DescentRemaining::from_vec`] rejects empty inputs, and the two
/// mutators ([`advance`](Self::advance) and [`prepend`](Self::prepend))
/// preserve non-emptiness by construction. `CompactString` keeps
/// typical-length names (≤24 bytes) inline, so the per-element advance
/// / rewind avoids the heap for the common path.
///
/// **API discipline.**
/// - [`head`](Self::head) is the next segment under consideration —
///   always present by invariant.
/// - [`is_terminal`](Self::is_terminal) is `true` when only the head
///   remains; the descent dispatcher routes through anchor materialization
///   on this edge and never calls [`advance`](Self::advance).
/// - [`advance`](Self::advance) consumes the head and is debug-asserted
///   non-terminal at call time. The terminal arm has already routed
///   through anchor materialization in production, which replaces the
///   `Pending` lifecycle entirely; advance is structurally never
///   reachable there.
/// - [`prepend`](Self::prepend) is the rewind path's mutator: a
///   `Vanished` response on `current_prefix` re-injects the prefix's own
///   segment as the new head while the prefix shifts up one level.
///
/// **Representation.** Components are stored *reverse* of descent order:
/// the logical head (next to consume) is the `Vec`'s last element. The
/// only mutated end is therefore the `Vec`'s O(1) tail — [`advance`] is
/// a `pop`, [`prepend`] a `push` — instead of the O(N) front shifts a
/// forward-order `Vec` forces (`advance` runs every forward descent
/// step). The reversal is an internal detail: every accessor keeps its
/// logical-order signature and semantics; [`iter`](Self::iter) and the
/// [`Debug`] impl present descent order so diagnostics and tests are
/// unaffected.
///
/// [`advance`]: Self::advance
/// [`prepend`]: Self::prepend
#[derive(Eq, PartialEq)]
pub struct DescentRemaining {
    /// Reversed: `inner.last()` is the logical head. Never empty.
    inner: Vec<CompactString>,
}

impl DescentRemaining {
    /// Construct from a `Vec` in descent order. Returns `None` iff `v`
    /// is empty, preserving the non-empty invariant. Sole intended
    /// producer is `materialize_path_or_pending`'s Pending branch, where
    /// the `prefix_idx + 1 < components.len()` gate already guarantees
    /// non-empty; the `None` arm is defense-in-depth against future
    /// callers. The one-time reverse into storage order is O(depth) on
    /// the cold descent-registration path.
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
        // Index the tail (the logical head under the reversed
        // representation) rather than `last().unwrap()` to encode the
        // invariant at the access site — a future maintainer can't
        // weaken `head` to a defensive `Option` without also adjusting
        // the type's construction discipline.
        &self.inner[self.inner.len() - 1]
    }

    /// Number of remaining segments. Always `>= 1` by invariant.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Always `false` — non-emptiness is a type-level invariant
    /// upheld by [`Self::from_vec`] (rejects empty inputs) and the
    /// mutators ([`Self::advance`] / [`Self::prepend`]). Implemented
    /// so the `len() / is_empty()` pair is complete by Rust convention;
    /// production callers should prefer [`Self::is_terminal`].
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// True iff only the head remains (`len() == 1`). The descent
    /// dispatcher's terminal arm consumes the head via anchor
    /// materialization on this edge and never calls [`Self::advance`].
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.inner.len() == 1
    }

    /// Consume the logical head (pop the reversed `Vec`'s tail).
    /// Preserves the non-empty invariant by debug-asserting non-terminal
    /// at entry; release builds clamp on terminal (no-op) rather than
    /// violating the invariant.
    ///
    /// Production callers (`advance_descent` in
    /// `specter-engine::descent`) guard the call with
    /// [`is_terminal`](Self::is_terminal) — `dispatch_descent_ok` routes
    /// the terminal edge through anchor materialization, which replaces
    /// the `Pending` lifecycle before this method would ever be
    /// reachable on a single-element remaining.
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

    /// Rewind by one segment: re-inject `segment` as the new logical
    /// head (push onto the reversed `Vec`'s tail). Used by
    /// `dispatch_descent_vanished`'s rewind branch, where a `Vanished`
    /// response on `current_prefix` shifts the descent up one level and
    /// the vanished prefix's own segment becomes the next-to-consume
    /// component on the way back down.
    pub fn prepend(&mut self, segment: CompactString) {
        self.inner.push(segment);
    }

    /// Iterate the components in descent (logical) order. For test
    /// assertions and diagnostics only — production code uses
    /// [`head`](Self::head) / [`len`](Self::len) /
    /// [`is_terminal`](Self::is_terminal).
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &CompactString> {
        self.inner.iter().rev()
    }
}

impl std::fmt::Debug for DescentRemaining {
    /// Descent (logical) order, hiding the reversed internal storage so
    /// diagnostics read the way the path is consumed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// `Standard` — event-driven burst; preserves baseline; fires Effect on stable.
/// `Seed` — fresh Profile or post-Effect rebase; sets baseline; no Effect.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BurstIntent {
    #[default]
    Standard,
    Seed,
}

/// Discriminator for a scheduled timer's role within a Burst's lifecycle.
///
/// `Settle` — debounce timer armed during [`PreFirePhase::Batching`].
/// Expiry drives Batching → Verifying.
/// `BurstDeadline` — Burst-level max-settle timer armed at Burst start.
/// Expiry sets `PreFireBurst.forced = true` and dispatches by current
/// phase. The timer is carried on [`PreFireBurst`] and is structurally
/// invalid in post-fire phases; once the burst crosses
/// [`PreFireBurst::into_post_fire`] the timer is dropped from the
/// type's field set, and a stale fire is filtered out by the
/// [`PostFireBurst::timer_token`] projection (the engine's stale-drain
/// consumes the projection through [`ProfileState::timer_token`]).
/// `AwaitGateDeadline` — recovery timer armed at
/// [`PostFirePhase::Awaiting`] entry. Expiry indicates the actuator is
/// taking longer than expected (likely a hung child); the engine
/// force-transitions to `Rebasing` to re-establish a baseline against
/// disk reality.
/// `RebaseSettle` — the post-fire mirror of `Settle`: the spacing wait
/// armed during [`PostFirePhase::RebaseSettling`]. Expiry drives
/// `RebaseSettling → Rebasing` for the next N=2 sample. Carried on
/// [`PostFireBurst`]; structurally `None` on pre-fire (the post-fire
/// analogue of how `Settle` is `None` on `Verifying`/`Draining`).
/// `RebaseCeiling` — the post-fire mirror of `BurstDeadline`: the
/// rebase loop's max bound, armed once at the first `Awaiting →
/// Rebasing` edge and tracked on the `rebase_ceiling` lifecycle.
/// Expiry latches the loop's terminal, applied with the verdict in
/// hand at the next `Rebasing` response (the forced-mirror of
/// `BurstDeadline → forced`). Like `BurstDeadline`, it is filtered to
/// `None` once consumed (here: once `Reached`), so the stale entry
/// lazy-drops.
///
/// Carried alongside [`TimerId`] on the engine's heap entry and on
/// [`crate::input::Input::TimerExpired`] so dispatch routes directly on
/// the kind without re-deriving from Profile state. The [`TimerId`]
/// continues to act as the lazy-invalidation epoch — `kind` only narrows
/// the validation slot, it does not replace it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TimerKind {
    #[default]
    Settle,
    BurstDeadline,
    AwaitGateDeadline,
    RebaseSettle,
    RebaseCeiling,
}

/// Lifecycle of a Profile's anchor `watch_demand` contribution.
///
/// Two-state machine:
/// - [`Self::None`] — Profile holds no anchor contribution. Reachable
///   when the Profile is `Pending` (descent prefix carries the
///   STRUCTURE watch instead), `Purged` (`Input::WatchOpRejected`
///   clamped the slot), or freshly constructed pre-attach.
/// - [`Self::Held`] — Profile contributes `+1` (at its `events` mask)
///   to its anchor's `watch_demand`. Set on the path that bumped the counter
///   (immediate-Seed in `attach_sub_inner` or descent's anchor
///   materialization); cleared on the matching decrement (anchor
///   terminal event, reap, clamp purge).
///
/// Encoded as a sum type so the dispatch sites — `release_anchor_claim`,
/// the recompute, every `dispatch_*_vanished` — read the lifecycle
/// directly rather than combining a flag with [`ProfileState`]. The
/// trichotomy "materialized / pending / purged" emerges from
/// `(state, anchor_claim)` rather than from a third variant: every
/// release helper treats Purged identically to None (no contribution
/// to drop), so distinguishing them at the type level adds no
/// dispatch information.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum AnchorClaim {
    #[default]
    None,
    Held,
}

/// The settled reference a *classified* anchor compares fresh probes
/// against, in the one window each variant owns:
///
/// - [`Self::Unset`] — no settled baseline yet. A freshly-classified
///   anchor (resource attach against a known-kind slot, or descent
///   materialisation) before its first successful graft. There is
///   nothing to drift against; the first graft installs the baseline.
/// - [`Self::Snapshot`] — active mode. The last settled snapshot; the
///   drift verdict is `current.hash() != settled.hash()`.
/// - [`Self::Witness`] — loss→recovery window. The anchor vanished and
///   its baseline snapshot was dropped, but the pre-loss
///   anchor-rooted hash is retained so the post-recovery Seed-Ok can
///   still decide whether the tree drifted while the anchor was gone.
///   Consumed (overwritten by [`Self::Snapshot`]) at the next rebase.
///
/// `Snapshot` and `Witness` are mutually exclusive *by construction* —
/// there is no representable value carrying both a live baseline and a
/// survival witness. The "a present baseline implies no survival
/// witness" rule is therefore a type property, not a checked
/// convention.
#[derive(Debug)]
enum SettledState<S> {
    Unset,
    Snapshot(S),
    Witness(u128),
}

/// The per-payload operations the generic [`SettledState`] /
/// [`AnchorClassification`] projections need without re-wrapping a
/// [`TreeSnapshot`] just to read it. Implemented once per concrete
/// anchor payload (`LeafEntry` for File anchors, `Arc<DirSnapshot>`
/// for Dir anchors); keeps the per-kind hash route and the owned
/// re-wrap localised instead of fanned out across the accessors.
trait AnchorPayload {
    /// Anchor-rooted digest — `LeafEntry::leaf_hash` for File,
    /// `DirSnapshot::dir_hash` for Dir.
    fn payload_hash(&self) -> u128;
    /// Owned [`TreeSnapshot`] re-wrap (`Arc` bump for Dir, copy for
    /// File). The sum stores the inner payload, never a
    /// `TreeSnapshot`, so the owned-projection accessors mint the
    /// wrapper on demand.
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
    /// The settled anchor-rooted hash, or `None` when no settled
    /// reference exists yet ([`Self::Unset`]). `Snapshot` digests its
    /// payload; `Witness` returns the retained pre-loss hash directly.
    /// This is also the witness a clear captures: the value that must
    /// survive into [`AnchorClassification::Unclassified`] so a later
    /// recovery can still detect drift.
    fn to_hash(&self) -> Option<u128> {
        match self {
            Self::Unset => None,
            Self::Snapshot(s) => Some(s.payload_hash()),
            Self::Witness(h) => Some(*h),
        }
    }

    /// The owned baseline snapshot — `Some` only in active mode
    /// ([`Self::Snapshot`]). `Unset` (no baseline yet) and `Witness`
    /// (baseline dropped at loss) have no snapshot to lend.
    fn snapshot(&self) -> Option<TreeSnapshot> {
        match self {
            Self::Snapshot(s) => Some(s.rewrap()),
            Self::Unset | Self::Witness(_) => None,
        }
    }

    /// The retained pre-loss hash — `Some` only across the
    /// loss→recovery window (`Witness`). An active `Snapshot`
    /// baseline and `Unset` both yield `None`: neither carries a
    /// survival witness.
    ///
    /// The Witness-only complement of [`Self::snapshot`] (the
    /// Snapshot-only owned projection) within [`Self::to_hash`]'s
    /// domain — the three accessors are one lattice over the sum:
    /// `to_hash` is `Some` iff exactly one of `snapshot` /
    /// `witness_hash` is, never both (the variants are disjoint), so
    /// no arm is double-counted and the witness can never be silently
    /// folded into the active-baseline projection.
    const fn witness_hash(&self) -> Option<u128> {
        match self {
            Self::Witness(h) => Some(*h),
            Self::Unset | Self::Snapshot(_) => None,
        }
    }
}

/// The anchor's on-disk classification and its settled reference, as
/// one sum.
///
/// The discriminant *is* the anchor kind: there is no separately
/// stored `kind` to disagree with the snapshot variant.
/// `current = Dir ⇒ kind = Dir`, `current = File ⇒ kind = File`, and
/// `unclassified ⇒ no snapshot` are structural — an ill-shaped pair
/// cannot be constructed, so the engine's typed probe-dispatch chain
/// is the *only* place kind agreement is decided, and a clear /
/// install sequence cannot leave the pair half-written.
///
/// **`Dir.current` is dual-purpose.** Besides the drift-comparison
/// snapshot, its entries *are* the covered-descendant watch-claim
/// membership set: [`Profile::take_current`] hands the live `Dir`
/// snapshot to the wholesale-deletion walk that releases every
/// per-descendant contribution. A parallel descendant-id collection
/// would duplicate exactly what the snapshot already encodes (and
/// re-introduce the drift surface this sum removes); the live `Dir`
/// snapshot is the single source of that membership.
#[derive(Debug)]
enum AnchorClassification {
    /// Kind not yet known, or cleared at anchor loss. No snapshot is
    /// representable here. `witness` carries the pre-loss
    /// anchor-rooted hash across the loss window (set when the cleared
    /// anchor had a settled reference; `None` for a fresh,
    /// never-classified Profile).
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

/// Frozen config identity plus the three caches that are *total
/// functions* of it. Private fields and a sole constructor make
/// "derived once from a frozen identity, never independently writable"
/// a structural property rather than a documented convention: there is
/// no path to a `ProfileConfig` whose `config_hash` disagrees with its
/// `identity`.
///
/// `identity` ([`ProfileIdentity`] = `{config, max_settle, events}`)
/// is the Profile partition key's config half; `config_hash`,
/// `exclude_strings`, and `has_per_file_fds` are each a pure projection
/// of it, materialised once at [`Self::new`].
#[derive(Debug)]
struct ProfileConfig {
    identity: ProfileIdentity,
    config_hash: u64,
    exclude_strings: Arc<[CompactString]>,
    has_per_file_fds: bool,
}

impl ProfileConfig {
    /// Derive all three caches from a frozen [`ProfileIdentity`]. The
    /// canonical hash route is [`ProfileIdentity::config_hash`];
    /// `exclude_strings` projects `config.exclude` in the
    /// builder-canonical order (already sorted by source, so no
    /// re-sort); `has_per_file_fds` is true iff the event mask carries
    /// CONTENT or METADATA (covered Leaves then need their own FDs).
    fn new(identity: ProfileIdentity) -> Self {
        let config_hash = identity.config_hash();
        let has_per_file_fds = identity
            .events
            .intersects(ClassSet::CONTENT | ClassSet::METADATA);
        let exclude_strings: Arc<[CompactString]> = identity
            .config
            .exclude
            .iter()
            .map(|g| CompactString::from(g.source()))
            .collect();
        Self {
            identity,
            config_hash,
            exclude_strings,
            has_per_file_fds,
        }
    }
}

/// The Profile's deferred-release obligations to the Tree refcount
/// aggregate. The pure-step `Tree` has no `Drop` reach, so each
/// obligation is encoded as a cached id/flag here and released
/// explicitly at detach / reap / purge. Drift between this record and
/// the Tree's contribution map is a **Tree refcount leak**, so every
/// write routes through a typed accessor that keeps the cache and the
/// Tree aggregate in lockstep.
///
/// **Scope boundary — do not widen.** This holds *only* the two
/// homeless cached tokens whose sole purpose is deferred release. It
/// deliberately excludes the other two of the four Tree claims, each
/// of which is a side-effect of a primary concern that owns it:
/// - the **descent-prefix** claim *is* `ProfileState::Pending`'s
///   `DescentState::current_prefix`; release routes through the state
///   machine.
/// - the **1-to-N covered-descendant** claims *are*
///   `AnchorClassification::Dir.current`'s entries (the live snapshot
///   is the membership set; [`Profile::take_current`] hands it to the
///   wholesale-deletion walk).
///
/// Co-locating either here would duplicate that state and re-create
/// the exact drift surface this decomposition removes.
#[derive(Debug)]
struct TreeContributions {
    /// "Do I owe `sub_watch(resource, ProfileAnchor(pid))`?" — the
    /// anchor contribution flag. The reap-time trichotomy
    /// (materialized / pending / purged) emerges from `(state,
    /// anchor_claim)`, so this stays orthogonal to the classification
    /// sum.
    anchor_claim: AnchorClaim,
    /// Cached parent Resource carrying this Profile's
    /// `ContribKey::ProfileParent` STRUCTURE contribution. `None` when
    /// the anchor is itself a Tree root (root rename detection then
    /// unavailable). Also the anchor-loss recovery channel —
    /// deliberately preserved across `discard_anchor_state`; released
    /// only by reap / `WatchOpRejected` purge. A stale cache here
    /// leaks the old parent's `+1`.
    watch_root_parent: Option<ResourceId>,
}

/// One stability state machine per `(Resource, ProfileIdentity)`,
/// decomposed into single-concern substructures.
///
/// Only `resource` (the slot axis) and `settle` (the per-Profile
/// mutable debounce param the engine recomputes as
/// `min(remaining_subs.settles)`) are `pub` fields. Every other
/// concern is module-private, exposing a typed accessor/transition API
/// — the cross-crate write surface is `Profile`'s `pub fn`s, never a
/// field assignment. The substructures that own a cross-field
/// invariant: `ProfileConfig` (frozen identity ⇒ derived caches),
/// `AnchorClassification` (snapshot-shape ⊕ baseline/witness
/// exclusion), `TreeContributions` (deferred Tree releases — drift =
/// refcount leak). The burst state machine needs no such wrapper: it
/// is the plain module-private [`ProfileState`] field `state`, read
/// via [`Self::state`] and transitioned via [`Self::transition_state`]
/// / the typed-move accessors — its variants (`Idle | Pending |
/// Active`) and their payloads (`DescentState`, the `ActiveBurst`
/// split) are themselves the single source of every burst invariant.
/// The `Draining → Verifying` reconfirm is a *fresh query*
/// ([`crate::ProfileState::in_active_standard_burst`] over the live
/// tree), not a cached counter. (Effect fire history is per-Sub —
/// [`crate::Sub::has_fired`] — not a Profile substructure.)
#[derive(Debug)]
pub struct Profile {
    /// The Tree slot this Profile's stability machine anchors at — the
    /// slot axis of the `(resource, config_hash)` partition key. Stays
    /// `pub`: a plain identity read with no invariant to guard.
    pub resource: ResourceId,
    /// Frozen config identity and its derived caches. Read via
    /// [`Self::config`] / [`Self::config_hash`] /
    /// [`Self::exclude_strings`] / [`Self::max_settle`] /
    /// [`Self::events`] / [`Self::has_per_file_fds`]; never
    /// independently writable (sole constructor [`ProfileConfig::new`]).
    cfg: ProfileConfig,
    /// Per-Profile mutable debounce interval. **Not identity** —
    /// `max_settle` is the identity half ([`Self::max_settle`]);
    /// `settle` the engine recomputes as `min(remaining_subs.settles)`
    /// on attach/detach. Stays `pub`: `recompute_profile_settle` writes
    /// it directly and there is no cross-field invariant to guard here
    /// (the config layer's `validate_settle` is the `settle <=
    /// max_settle` boundary; [`Self::new`] debug-asserts it).
    pub settle: Duration,
    /// The anchor's classification and settled reference as one sum
    /// (kind ⊕ live snapshot ⊕ settled baseline ⊕ survival witness).
    /// The discriminant *is* the kind; "no snapshot while
    /// unclassified" and "no baseline while a survival witness is
    /// held" hold by construction. Reads: [`Self::kind`] /
    /// [`Self::current`] / [`Self::baseline`] / [`Self::current_dir`] /
    /// [`Self::baseline_dir`] / [`Self::settled_hash`] /
    /// [`Self::current_is_some`]. Writes: [`Self::install_dir_current`]
    /// / [`Self::install_file_current`] / [`Self::rebase_baseline`] /
    /// [`Self::take_current`] / [`Self::clear_anchor_classification`] /
    /// [`Self::materialize_anchor`]. `Resource.kind` is a separate
    /// Tree-side parallel cache the engine never consults for the
    /// anchor's own kind in any post-attach path.
    anchor: AnchorClassification,
    /// Burst state machine. Module-private; the variant payloads carry
    /// every burst invariant by construction (the [`ActiveBurst`] split
    /// type-bans cross-phase field leaks), so no wrapper or sidecar
    /// counter is needed. Read via [`Self::state`], transitioned via
    /// [`Self::transition_state`] / the typed-move accessors. The
    /// `Draining → Verifying` reconfirm is a fresh query over the live
    /// tree ([`ProfileState::in_active_standard_burst`]), not cached
    /// here.
    state: ProfileState,
    /// Deferred-release obligations to the Tree refcount aggregate
    /// (`anchor_claim`, `watch_root_parent`). Drift = refcount leak.
    /// Read via [`Self::anchor_claim`] / [`Self::watch_root_parent`];
    /// written via [`Self::install_anchor_claim_held`] /
    /// [`Self::release_anchor_claim_now`] /
    /// [`Self::set_watch_root_parent`] /
    /// [`Self::take_watch_root_parent`].
    contributions: TreeContributions,
}

impl Profile {
    /// Construct a fresh Profile: state `Idle` (no burst-finish
    /// directive yet), no baseline/current, no watch-root parent.
    /// (Effect fire history is per-Sub — [`crate::Sub::has_fired`] —
    /// not a Profile concern.)
    ///
    /// `identity` ([`ProfileIdentity`] = `{config, max_settle,
    /// events}`) is the Profile partition key's config half, taken by
    /// value: `ProfileConfig::new` folds it once into the lifetime-
    /// stable `config_hash` plus the `exclude_strings` /
    /// `has_per_file_fds` projections. There is no path to a Profile
    /// with an unset or stale hash. The sole production caller
    /// (`find_or_create_profile`) already holds the `ProfileIdentity`
    /// and moves it straight in — no field unpack, no clone.
    ///
    /// `settle` is the per-Profile mutable debounce interval (recomputed
    /// by the engine as `min(remaining_subs.settles)`), distinct from
    /// the identity's `max_settle`. The `settle <= max_settle` relation
    /// is a `debug_assert!`: the config layer's `validate_settle` is
    /// the real boundary (it enforces `max_settle >= 4 × settle`); a
    /// breach here means a caller bypassed config validation.
    ///
    /// `kind` is the anchor's classified shape at construction,
    /// projected into the `AnchorClassification` sum: `None` ⇒
    /// `Unclassified` (a `DescentScaffold` or freshly-`ensure`d slot;
    /// descent materialisation classifies it via
    /// [`Self::materialize_anchor`], the first Seed-Ok via
    /// [`Self::install_dir_current`] / [`Self::install_file_current`]);
    /// `Some(Dir)` / `Some(File)` ⇒ a classified anchor with no
    /// snapshot or baseline yet (the first probe response grafts the
    /// current snapshot). `Some(Unknown)` is defensively dead:
    /// `Resource::kind()` maps `Unknown → None`, so the sole production
    /// caller never threads `Some(Unknown)`; the arm is debug-asserted
    /// and degrades to `Unclassified` (the same shape as `None`) in
    /// release rather than panicking or constructing an illegal state.
    #[must_use]
    pub fn new(
        resource: ResourceId,
        identity: ProfileIdentity,
        settle: Duration,
        kind: Option<ResourceKind>,
    ) -> Self {
        debug_assert!(
            settle <= identity.max_settle,
            "Profile::new: settle ({settle:?}) must not exceed max_settle ({:?}) — \
             config-layer validate_settle enforces max_settle >= 4 × settle; a \
             breach here means a caller bypassed config validation",
            identity.max_settle,
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
        }
    }

    /// Graft a Dir-shaped `current` into the anchor classification.
    /// Sole legitimate writer of the Dir `current` slot.
    ///
    /// - From `Unclassified`: classify as `Dir`, carrying any survival
    ///   witness forward into `settled` (recovery: `Witness(h)`; fresh:
    ///   `Unset`). The witness must survive classification so the
    ///   post-recovery drift verdict still has a reference.
    /// - From `Dir`: overwrite `current`, leaving `settled` untouched
    ///   (a re-graft within the same materialised epoch — fresh or
    ///   mid-recovery).
    /// - From `File`: a `File`-kinded Profile receiving a `Dir` graft
    ///   is a dispatcher routing breach. The engine's
    ///   `kind_agrees_or_finalize` boundary catches this and routes
    ///   through `finalize_anchor_lost` (which clears to `Unclassified`)
    ///   *before* any graft, so this arm is defensively dead;
    ///   `debug_assert!` flags a future boundary bypass and release
    ///   builds re-classify rather than construct an illegal pair.
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

    /// Graft a File-shaped `current` into the anchor classification.
    /// Symmetric with [`Self::install_dir_current`]: carries the
    /// survival witness forward from `Unclassified`, overwrites
    /// `current` from `File` leaving `settled` untouched, and treats a
    /// `Dir`-kinded Profile as the defensively-dead dispatcher breach.
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

    /// Settle the live `current` snapshot as the new baseline: `settled
    /// := Snapshot(current)`. Any survival witness is *consumed* — the
    /// `Witness → Snapshot` move is the structural end of the
    /// loss→recovery window (no separate witness-clear step exists).
    ///
    /// Called only at a **terminal pin**, after a successful graft where
    /// `current.is_some()` holds:
    /// - `dispatch_rebase_ok` on `Stable` (two settle-spaced equal
    ///   post-command samples) or `Unstable + ceiling Reached` (the
    ///   bounded rebase-loop terminal — pin the freshest observation).
    /// - the Seed **pin path** `seed_pin_body`, reached from
    ///   `dispatch_seed_ok`'s `Stable` and `Unstable + forced` arms.
    ///
    /// The rebase-loop `Unstable`/`Undischarged` arms (not yet at the
    /// ceiling) and the Seed `Unstable + !forced` / `Undischarged` arms
    /// graft (or skip) but **do not** rebase: the witness-survival
    /// contract — the survival witness must outlive an unbounded
    /// re-batch / rebase loop and be consumed only at the eventual
    /// terminal pin, so this consumer must never run on a looping arm.
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
    /// witness: settled.to_hash() }`. The witness is the settled
    /// reference's hash (`Snapshot` digests; `Witness` passes through;
    /// `Unset` ⇒ `None`), so a post-recovery Seed-Ok can still detect
    /// drift after the baseline snapshot is gone. Idempotent against
    /// an already-`Unclassified` anchor: the prior witness is
    /// preserved, never overwritten with `None`. Inverse of
    /// [`Self::materialize_anchor`].
    pub fn clear_anchor_classification(&mut self) {
        let witness = match &self.anchor {
            AnchorClassification::Unclassified { witness } => *witness,
            AnchorClassification::File { settled, .. } => settled.to_hash(),
            AnchorClassification::Dir { settled, .. } => settled.to_hash(),
        };
        self.anchor = AnchorClassification::Unclassified { witness };
    }

    /// Atomically install a descent-materialised anchor: transition
    /// `Pending → Idle`, install the claim, and classify the anchor
    /// with the discovered `kind`, **carrying the survival witness
    /// forward** (`Unclassified { witness } ⇒ File/Dir { current:
    /// None, settled: Witness(h) | Unset }`). Sole caller
    /// `Engine::materialize_profile_anchor`, which launches the Seed
    /// burst on the next statement — the `Idle` written here is a
    /// structural intermediate, never observed. The whole sequence
    /// runs under one `&mut self` so no reader sees a partial write.
    /// Inverse of [`Self::clear_anchor_classification`].
    ///
    /// Debug-asserts the fresh-materialisation preconditions
    /// (`state == Pending`, no claim, anchor `Unclassified`); release
    /// builds compile the asserts out and still classify atomically.
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

    /// Debug-time coherence tripwire for the multi-field
    /// classification coordinators (this `materialize_anchor` and the
    /// engine's `discard_anchor_state`).
    ///
    /// The snapshot-shape (`kind ⇔ current` variant) and
    /// baseline/witness-exclusion invariants are *structural* — no
    /// representable `AnchorClassification` violates them, so there
    /// is nothing to check there. What remains is the pair of
    /// one-directional cross-axis invariants the type system does not
    /// cover, asserted here so a future coordinator that leaves a
    /// `Pending` Profile classified (or holding the anchor claim)
    /// trips at the write site rather than latently at the next
    /// dispatch / reap:
    /// - `Pending ⇒ Unclassified` — during descent the anchor is not
    ///   probed; the descent prefix, not the anchor, carries the watch.
    /// - `Pending ⇒ ¬AnchorClaim::Held` — the descent prefix carries
    ///   the STRUCTURE watch; the anchor claim is installed only at
    ///   materialisation. (`reap_profile`'s trichotomy depends on this.)
    pub fn debug_assert_anchor_coherent(&self) {
        if matches!(self.state, ProfileState::Pending(_)) {
            debug_assert!(
                matches!(self.anchor, AnchorClassification::Unclassified { .. }),
                "anchor coherence: a Pending Profile must be Unclassified",
            );
            debug_assert!(
                matches!(self.contributions.anchor_claim, AnchorClaim::None),
                "anchor coherence: a Pending Profile must not hold the anchor claim",
            );
        }
    }

    /// Sole legitimate post-construction writer of `state`. Returns the
    /// prior state via `mem::replace` so the typed-move callers
    /// (`transition_to_awaiting` → [`PreFireBurst::into_post_fire`];
    /// `restart_burst_from_fire_tail_residual` →
    /// [`PostFireBurst::into_pre_fire_residual`]; `finish_burst_to_idle`)
    /// can consume the prior burst by value without holding a `&mut
    /// state` borrow across the move. Shape-agnostic:
    /// transition preconditions are owned by the engine boundary
    /// (`require_idle` / `require_active_pre_fire`), not duplicated here.
    /// Not `#[must_use]` — whole-value-replace callers discard the return;
    /// only the typed-move callers bind it.
    pub const fn transition_state(&mut self, new: ProfileState) -> ProfileState {
        std::mem::replace(&mut self.state, new)
    }

    /// Install the anchor claim. Idempotent against `Held`. Production
    /// caller: `Engine::bootstrap_immediate`. (The descent-materialised
    /// claim rides [`Self::materialize_anchor`]'s bundled write instead.)
    pub const fn install_anchor_claim_held(&mut self) {
        self.contributions.anchor_claim = AnchorClaim::Held;
    }

    /// Release the anchor claim. Idempotent against `None`. Production
    /// caller: `Engine::release_anchor_claim`, which wraps this with the
    /// Tree-side `sub_watch`.
    pub const fn release_anchor_claim_now(&mut self) {
        self.contributions.anchor_claim = AnchorClaim::None;
    }

    /// The cached watch-root parent Resource, if this Profile owes a
    /// `ContribKey::ProfileParent` STRUCTURE contribution there. `None`
    /// for a root anchor. Read seam over the release-ledger field;
    /// `Engine::set_watch_root_parent` uses it for the cache-coherence
    /// and idempotence checks, `classify_event_carriers` for the
    /// anchor-recovery channel.
    #[must_use]
    pub const fn watch_root_parent(&self) -> Option<ResourceId> {
        self.contributions.watch_root_parent
    }

    /// Cache the watch-root parent id. The single write seam, wrapped
    /// by `Engine::set_watch_root_parent` (which also installs the
    /// Tree-side `add_watch` and the cache-coherence `debug_assert!`).
    /// Plain set — idempotence and coherence are the engine wrapper's
    /// concern, not duplicated here.
    pub const fn set_watch_root_parent(&mut self, parent: ResourceId) {
        self.contributions.watch_root_parent = Some(parent);
    }

    /// Take the cached watch-root parent, clearing it — the symmetric
    /// deferred-release primitive (`Engine::release_watch_root_parent_claim`
    /// keys the `sub_watch` removal off the returned id). Idempotent: a
    /// second call returns `None`.
    pub const fn take_watch_root_parent(&mut self) -> Option<ResourceId> {
        self.contributions.watch_root_parent.take()
    }

    /// Borrow the pre-fire burst payload iff
    /// `state == Active(PreFire(_), _)` — the `&self` mirror of
    /// [`Self::pre_fire_burst_mut`]. A read of the state's structural
    /// shape, never a transition; the engine's pre-fire dispatch reads
    /// (`probe_target`, the Seed first-fire witness `dirty`) route
    /// through this instead of re-matching `state()` inline.
    #[must_use]
    pub const fn pre_fire_burst(&self) -> Option<&PreFireBurst> {
        match &self.state {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(pre),
            _ => None,
        }
    }

    /// Borrow the pre-fire burst payload iff
    /// `state == Active(PreFire(_), _)` — a read of the state's
    /// structural shape, *not* a variant transition (the variant-level
    /// move still routes through [`Self::transition_state`]).
    pub const fn pre_fire_burst_mut(&mut self) -> Option<&mut PreFireBurst> {
        match &mut self.state {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(pre),
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

    /// Borrow the state machine. The universal read path — every `&self`
    /// [`ProfileState`] projection (`discriminant`, `burst_finish`,
    /// `detach_lifecycle`, `timer_token`, `is_draining`, `descent_state`)
    /// routes through this.
    #[must_use]
    pub const fn state(&self) -> &ProfileState {
        &self.state
    }

    #[must_use]
    pub const fn anchor_claim(&self) -> AnchorClaim {
        self.contributions.anchor_claim
    }

    /// Anchor kind discriminant — the sum's variant projected back to
    /// the engine's `Option<ResourceKind>` shape. `Unclassified ⇒
    /// None`; `File ⇒ Some(File)`; `Dir ⇒ Some(Dir)`.
    #[must_use]
    pub const fn kind(&self) -> Option<ResourceKind> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { .. } => Some(ResourceKind::File),
            AnchorClassification::Dir { .. } => Some(ResourceKind::Dir),
        }
    }

    /// The Profile's user-declared event-class mask. Invariant for the
    /// Profile's lifetime (folds into `config_hash`). Stable read seam
    /// over the frozen identity.
    #[must_use]
    pub const fn events(&self) -> ClassSet {
        self.cfg.identity.events
    }

    /// The frozen [`ScanConfig`] half of the Profile identity. Borrow
    /// for the `pattern` / `max_depth` / `recursive` / `exclude`
    /// coverage reads and the probe-request config clone.
    #[must_use]
    pub const fn config(&self) -> &ScanConfig {
        &self.cfg.identity.config
    }

    /// The lifetime-stable canonical config hash — the config axis of
    /// the `(resource, config_hash)` partition key. Bit-identical to
    /// `ProfileIdentity::config_hash()` of the identity passed to
    /// [`Self::new`] (it *is* that value, cached once).
    #[must_use]
    pub const fn config_hash(&self) -> u64 {
        self.cfg.config_hash
    }

    /// The settle-deadline ceiling — the identity half of the burst
    /// timings (folds into `config_hash`; invariant for the Profile's
    /// lifetime, in deliberate contrast to the mutable `settle`).
    #[must_use]
    pub const fn max_settle(&self) -> Duration {
        self.cfg.identity.max_settle
    }

    /// True iff covered Leaves need their own FDs (the event mask
    /// carries CONTENT or METADATA). Invariant for the Profile's
    /// lifetime — the reconciler reads it to decide per-file watch
    /// installation.
    #[must_use]
    pub const fn has_per_file_fds(&self) -> bool {
        self.cfg.has_per_file_fds
    }

    /// The substitution-side projection of `ScanConfig.exclude` (source
    /// strings, builder-canonical order). Returned by reference so the
    /// effect emitter `Arc::clone`s it rather than rebuilding.
    #[must_use]
    pub const fn exclude_strings(&self) -> &Arc<[CompactString]> {
        &self.cfg.exclude_strings
    }

    /// The settled baseline as an owned [`TreeSnapshot`] — `Some` only
    /// in active mode (a settled `Snapshot`). The sum stores the inner
    /// payload, not a `TreeSnapshot`, so this mints the wrapper (Arc
    /// bump for Dir, copy for File). `Unclassified`, a not-yet-settled
    /// anchor, and the loss-window witness all yield `None`.
    #[must_use]
    pub fn baseline(&self) -> Option<TreeSnapshot> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { settled, .. } => settled.snapshot(),
            AnchorClassification::Dir { settled, .. } => settled.snapshot(),
        }
    }

    /// The live `current` snapshot as an owned [`TreeSnapshot`].
    /// Minted on demand (Arc bump for Dir, copy for File) — the sum
    /// cannot lend a `&TreeSnapshot` it does not store in that shape.
    /// Hot Dir readers that only need the inner `Arc` should prefer
    /// [`Self::current_dir`] (no re-wrap); presence-only readers
    /// [`Self::current_is_some`].
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

    /// Borrow the live Dir `current` snapshot's `Arc` directly — the
    /// reconcile / probe hot path that wants `Arc::clone`, not an
    /// owned `TreeSnapshot` re-wrap. `None` for File-kinded,
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

    /// Borrow the settled Dir baseline's `Arc` directly — symmetric
    /// with [`Self::current_dir`] for the settled `Snapshot`. `None`
    /// unless the anchor is `Dir` in active mode.
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

    /// The settled anchor-rooted hash the post-recovery drift verdict
    /// compares `current` against — one total function over the sum:
    /// active-mode `Snapshot` digests its payload, the loss-window
    /// `Witness` passes its retained hash through, the
    /// `Unclassified` arm yields its carried witness, and a
    /// not-yet-settled anchor yields `None`. Replaces the separate
    /// witness accessor plus the ad-hoc baseline-hash branch at the
    /// drift reader.
    #[must_use]
    pub fn settled_hash(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { witness } => *witness,
            AnchorClassification::File { settled, .. } => settled.to_hash(),
            AnchorClassification::Dir { settled, .. } => settled.to_hash(),
        }
    }

    /// The loss-window survival witness: `Some(h)` iff the settled
    /// reference is *currently* a not-yet-consumed `Witness` (the
    /// pre-loss anchor-rooted hash retained across an anchor-loss
    /// window), not an active baseline `Snapshot` and not `Unset`.
    ///
    /// **Deliberately narrower than [`Self::settled_hash`]; the two
    /// must not be unified.** `settled_hash` is the *total* drift
    /// oracle — "what hash does the post-recovery verdict diff
    /// `current` against" — and so folds `Snapshot`, `Witness`, and
    /// the pre-classification `Unclassified { witness }` into one
    /// value. This accessor answers the strictly narrower question
    /// "is the anchor *right now* sitting on a live loss→recovery
    /// witness", true solely between the witness lift
    /// ([`Self::materialize_anchor`] / the `install_*_current`
    /// `Unclassified { witness } ⇒ classified { Witness }` arms) and
    /// its consumption ([`Self::rebase_baseline`], `Witness ⇒
    /// Snapshot`). `settled_hash`'s `Snapshot` arm (an active
    /// baseline is not a survival witness) and its `Unclassified`
    /// arm (recovery has not completed) would each mis-answer it.
    ///
    /// `Unclassified ⇒ None` is correct on both counts above and, at
    /// the sole consumer — a Seed-Ok past `apply_snapshot`, which has
    /// classified the anchor — unreachable.
    #[must_use]
    pub const fn survival_witness(&self) -> Option<u128> {
        match &self.anchor {
            AnchorClassification::Unclassified { .. } => None,
            AnchorClassification::File { settled, .. } => settled.witness_hash(),
            AnchorClassification::Dir { settled, .. } => settled.witness_hash(),
        }
    }

    /// Whether a live `current` snapshot is present, without minting
    /// (or `Arc`-bumping) one. The zero-cost presence check for
    /// readers that branch on "has the anchor been grafted yet?"
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

    /// Whether a settled baseline `Snapshot` is present, without
    /// minting (or `Arc`-bumping) one — the zero-cost presence
    /// complement of [`Self::baseline`], exactly as
    /// [`Self::current_is_some`] is of [`Self::current`]. [`Self::baseline`]
    /// yields `Some` only for a settled `Snapshot`, so this matches that
    /// arm directly.
    ///
    /// A loss-window `Witness` and a not-yet-settled anchor both yield
    /// `false`: neither is a *trustworthy settled baseline*. This is the
    /// load-bearing distinction for the burst-fork question "do I have a
    /// settled baseline to debounce against, or must I re-Seed?" —
    /// `current_is_some` answered it only because a settled baseline and
    /// a live `current` were once installed atomically; once they
    /// decouple (a Seed grafting `current` while deferring the pin) the
    /// fork must read presence of the *baseline*, not of `current`.
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

    /// Mutable descent payload — thin delegator to
    /// [`ProfileState::descent_state_mut`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        self.state.descent_state_mut()
    }

    /// Disarm this Profile's in-flight probe and return its prior
    /// correlation — thin delegator to [`ProfileState::take_probe`],
    /// joining the in-place state-mutator family beside
    /// [`Self::descent_state_mut`] / [`Self::mark_active_for_reap`].
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        self.state.take_probe()
    }

    /// Flip an Active burst's directive to `Reap`. `true` iff the flip
    /// landed (Active). Delegates to [`ProfileState::mark_active_for_reap`].
    #[must_use]
    pub const fn mark_active_for_reap(&mut self) -> bool {
        self.state.mark_active_for_reap()
    }

    /// Revive a zombie burst (`Reap → ReturnToIdle`). `true` iff a zombie
    /// was revived. Delegates to [`ProfileState::clear_active_reap`].
    #[must_use]
    pub const fn clear_active_reap(&mut self) -> bool {
        self.state.clear_active_reap()
    }

    /// The single in-life mutator of [`PostFirePhase::Awaiting`]'s
    /// `outstanding` — pure delegation through the state machine, the
    /// no-public-setter seam shared with [`Self::clear_active_reap`].
    /// The floor is enforced by the owner,
    /// [`PostFireBurst::note_effect_completion`].
    #[must_use]
    pub fn note_effect_completion(&mut self) -> AwaitVerdict {
        self.state.note_effect_completion()
    }

    /// The single in-life mutator of the active burst's `certified`
    /// proof — pure delegation through the state machine, the
    /// no-public-setter seam shared with [`Self::note_effect_completion`].
    /// The certified-sample floor is enforced by the owner,
    /// [`CertifiedPrior::advance`]; an `Active` burst folds through
    /// whichever side (pre-fire verify or post-fire rebase) it is on.
    /// `None` ⇒ the Profile was `Idle` / `Pending` (no Active burst)
    /// when a verdict choke folded a response — a structural regression
    /// the caller surfaces loudly (a choke runs only with its carrier
    /// proven Active).
    #[must_use]
    pub fn advance_quiescence(
        &mut self,
        authority: ProofAuthority,
        response: u128,
    ) -> Option<QuiescenceVerdict> {
        self.state.advance_quiescence(authority, response)
    }

    /// Latch the post-fire rebase-loop ceiling terminal — pure
    /// delegation through the state machine, the no-public-setter seam
    /// shared with [`Self::note_effect_completion`]. The `Armed →
    /// Reached` floor is enforced by the owner,
    /// [`PostFireBurst::mark_rebase_ceiling_reached`]; `true` iff the
    /// latch landed (the ceiling was `Armed`). Any non-`Active(PostFire)`
    /// state yields `false` — the loud-regression signal the timer arm
    /// debug-asserts against (the ceiling fires only from `Armed`,
    /// which `timer_token` filters to `Active(PostFire)`).
    #[must_use]
    pub const fn mark_rebase_ceiling_reached(&mut self) -> bool {
        self.state.mark_rebase_ceiling_reached()
    }

    /// Take the live `current` snapshot, leaving the arm's `current`
    /// `None` and `settled` untouched — the covered-descendant
    /// claim-release primitive. The returned `Dir` snapshot's entries
    /// *are* the descendant membership set the caller
    /// (`Engine::release_descendant_claim`) walks via wholesale
    /// deletion. Idempotent (a second call finds `None`); `File` has
    /// no descendants and `Unclassified` has no snapshot, both
    /// short-circuit to `None`. Not subsumed by
    /// [`Self::clear_anchor_classification`]: it runs first and is also
    /// called standalone from the `dispatch_*_vanished/failed` +
    /// `reap_profile` sites.
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
}

impl ProfileMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up an existing Profile by `(resource, config_hash)`. Returns
    /// `None` if no Profile at this resource matches the hash.
    #[must_use]
    pub fn find(&self, resource: ResourceId, config_hash: u64) -> Option<ProfileId> {
        self.by_resource
            .get(resource)?
            .iter()
            .find(|(h, _)| *h == config_hash)
            .map(|(_, id)| *id)
    }

    /// Insert a fresh Profile and write back-references on both the Tree
    /// (`Resource.profiles`) and the `ProfileMap` (`by_resource`). Caller
    /// has verified `find` returns `None` for `(profile.resource,
    /// profile.config_hash)`; a debug-build assertion guards against repeat.
    ///
    /// Panics if `profile.resource` is stale (no live Tree slot). The Engine
    /// must construct the Resource before attaching a Profile to it.
    pub fn attach(&mut self, tree: &mut Tree, profile: Profile) -> ProfileId {
        let resource = profile.resource;
        let hash = profile.config_hash();
        debug_assert!(
            self.find(resource, hash).is_none(),
            "ProfileMap::attach called twice for the same (resource, config_hash) — caller must `find` first",
        );
        let id = self.profiles.insert(profile);
        // SecondaryMap::entry returns None only if the key has been removed
        // from a primary-tracked SlotMap with a generation that no longer
        // matches. For a freshly-minted ResourceId, we expect `Some`.
        self.by_resource
            .entry(resource)
            .expect("ProfileMap::attach: resource is stale (slot was reaped)")
            .or_default()
            .push((hash, id));
        tree.get_mut(resource)
            .expect("ProfileMap::attach: resource has no live Tree slot")
            .profiles
            .push((hash, id));
        id
    }

    /// Remove a Profile and clear back-references on both indices. The
    /// caller is responsible for any subsequent `tree.try_reap(resource)`
    /// once it confirms no other anchors remain.
    pub fn detach(&mut self, tree: &mut Tree, id: ProfileId) -> Option<Profile> {
        let p = self.profiles.remove(id)?;
        if let Some(v) = self.by_resource.get_mut(p.resource) {
            v.retain(|(h, pid)| !(*pid == id && *h == p.config_hash()));
        }
        if let Some(r) = tree.get_mut(p.resource) {
            r.profiles
                .retain(|(h, pid)| !(*pid == id && *h == p.config_hash()));
        }
        Some(p)
    }

    #[must_use]
    pub fn get(&self, id: ProfileId) -> Option<&Profile> {
        self.profiles.get(id)
    }

    pub fn get_mut(&mut self, id: ProfileId) -> Option<&mut Profile> {
        self.profiles.get_mut(id)
    }

    /// Iterator over the Profiles attached at `resource`, in
    /// `Resource.profiles` insertion order.
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
}

#[cfg(test)]
mod tests {
    use super::{
        AnchorClassification, ClassSet, Profile, ProfileIdentity, ProfileMap, ProfileState,
        ScanConfig, SettledState,
    };
    use crate::fs_id::FsIdentity;
    use crate::ids::ResourceId;
    use crate::output::StepOutput;
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

    /// Test constructor preserving the pre-decomposition 6-arg call
    /// shape: folds `(config, max_settle, events)` into the
    /// [`ProfileIdentity`] the real [`Profile::new`] now takes by
    /// value, so every fixture's exact parameters survive the
    /// decomposition unchanged.
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
            ProfileIdentity {
                config,
                max_settle,
                events,
            },
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
    }

    /// `Profile::new` debug-asserts `settle <= max_settle`. The burst
    /// lifecycle needs the settle (quiet-window) timer to expire before
    /// the burst deadline; otherwise every burst force-fires without
    /// ever reaching a stable verdict. The config layer's
    /// `validate_settle` is the real boundary (it enforces
    /// `max_settle >= 4 × settle`), so reaching construction with
    /// `settle > max_settle` means a caller bypassed config validation —
    /// the constructor trips loudly in debug rather than silently
    /// shipping a Profile that forces every burst.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "must not exceed max_settle")]
    fn profile_new_panics_when_settle_exceeds_max_settle() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        // max_settle = 5s, settle = 10s — the rejected-but-representable
        // combination the config layer should have caught upstream.
        let _ = mk_profile(
            r,
            cfg(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            NO_EVENTS,
            None,
        );
    }

    /// `has_per_file_fds` defaults to false when `events` excludes both
    /// CONTENT and METADATA. The flag is invariant for the Profile's
    /// lifetime — set once at construction from the events mask.
    #[test]
    fn new_profile_initialises_has_per_file_fds_false_for_empty_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!p.has_per_file_fds());
        assert_eq!(p.events(), ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (closes
    /// E2E #3 by default for `subtree-root`).
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        assert!(p.has_per_file_fds());
        assert_eq!(p.events(), ClassSet::CONTENT);
    }

    /// `has_per_file_fds` is also true when METADATA is in the mask (a
    /// metadata-only watch needs per-file FDs for chmod / nlink signals).
    #[test]
    fn new_profile_has_per_file_fds_when_metadata_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert!(p.has_per_file_fds());
    }

    /// STRUCTURE-only watch does not flip `has_per_file_fds` — directory
    /// entries are observed at the parent dir's FD, not at per-file FDs.
    #[test]
    fn new_profile_has_per_file_fds_false_for_structure_only() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE, None);
        assert!(!p.has_per_file_fds());
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

    /// Different `events` mask produces different `config_hash`
    /// (partition-by-mask).
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
        // Recovery shape: a classified Dir carrying a live current and a
        // survival witness (baseline cleared at the prior loss).
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

    /// Re-grafting a Dir current on a Dir-classified Profile keeps the
    /// discriminant and leaves `settled` untouched (a within-epoch
    /// re-graft, fresh or mid-recovery).
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

    /// Grafting onto an `Unclassified` anchor that carries a survival
    /// witness (the post-loss recovery shape) classifies it *and*
    /// carries the witness forward into `settled`, so the
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

    /// Cross-arm misuse: grafting a `Dir` onto a `File`-classified
    /// Profile panics in debug builds. Production paths never reach
    /// this branch — `Engine::kind_agrees_or_finalize` catches the
    /// routing breach at the dispatcher boundary first.
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
        // Boundary-bypass: a future caller skips
        // `kind_agrees_or_finalize`; the graft's debug_assert fires.
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

    /// `Profile.exclude_strings` mirrors `ScanConfig.exclude` in source-string
    /// form, sorted lexicographically. The builder sorts at `build()`, so the
    /// projection inherits the canonical order regardless of insertion order.
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

    /// `Profile.exclude_strings` is empty (zero-length slice) when the
    /// `ScanConfig` has no excludes — pin so consumers can rely on the
    /// projection always being populated.
    #[test]
    fn profile_new_exclude_strings_empty_for_no_excludes() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(p.exclude_strings().is_empty());
    }

    /// The Arc on `Profile.exclude_strings` is the substitution-side handle
    /// shared across every Sub joined to this Profile. Two clones of the
    /// field point at the same allocation; the `bytes-per-Arc` cost is
    /// constant regardless of Sub fanout.
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
        ActiveBurst, AwaitVerdict, BurstFinish, BurstIntent, CertifiedPrior, DescentRemaining,
        DescentState, DirtyProvenance, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase,
        QuiescenceVerdict, TimerKind,
    };
    use crate::ids::{ProbeCorrelation, TimerId};
    use crate::op::ProofAuthority;
    use std::collections::BTreeSet;
    use std::path::Path;

    fn tid(n: u64) -> TimerId {
        TimerId::from(n)
    }

    /// Inline twin of `testkit::dirty_provenance` — the `testkit`
    /// feature is off for `cargo nextest run -p specter-core`, so the
    /// canonical fixture is unreachable from this module's build. Mirrors
    /// the production ingest contract exactly: each pair is one
    /// [`DirtyProvenance::note`] in slice order (a repeated `ResourceId`
    /// is last-writer-wins), paths must be absolute (the component-LCA
    /// relies on every value sharing the root) — a relative path is a
    /// fixture bug, caught loudly in dev/CI and inert in release.
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

    /// The captured-path set a [`DirtyProvenance`] built from `entries`
    /// projects to (`DirtyProvenance::chains`) — the observable the
    /// migrated residual / mutation tests assert against now that the
    /// provenance has no `PartialEq` and no field peek.
    fn expected_chains(entries: &[&str]) -> BTreeSet<Arc<Path>> {
        entries.iter().map(|p| Arc::from(Path::new(*p))).collect()
    }

    /// `n` distinct `ResourceId`s from a throwaway slotmap — core has no
    /// `Tree`, and these tests only need the keys to differ.
    fn rids(n: usize) -> Vec<ResourceId> {
        let mut sm = slotmap::SlotMap::<ResourceId, ()>::with_key();
        (0..n).map(|_| sm.insert(())).collect()
    }

    // `DirtyProvenance`'s own contract. The engine's `pre_fire_target`
    // / emission tests exercise it transitively; these pin the type
    // directly — above all the component-wise LCA the path-LCA scope
    // rests on.

    #[test]
    fn dirty_provenance_lca_path_is_component_wise_not_byte_prefix() {
        // `/w/a` is NOT a prefix of `/w/ab`: a byte-prefix LCA would
        // wrongly root the probe at `/w/a` and clip `/w/ab`.
        // Component-wise, the only shared ancestor is `/w`.
        let r = rids(2);
        let dp = dirty_prov(&[(r[0], "/w/a"), (r[1], "/w/ab")]);
        assert_eq!(dp.lca_path().as_deref(), Some(Path::new("/w")));

        // A genuinely divergent pair reduces to its real ancestor.
        let dp = dirty_prov(&[(r[0], "/w/x/a"), (r[1], "/w/y/b")]);
        assert_eq!(dp.lca_path().as_deref(), Some(Path::new("/w")));
    }

    #[test]
    fn dirty_provenance_lca_path_single_entry_is_identity_empty_is_none() {
        // The dominant single-file-edit case: a lone captured path is
        // its own LCA, returned without reallocating (the
        // `Arc::clone(first)` fast path) — pinned via pointer identity.
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
        // `note` is keyed by slot: a repeat event for one slot
        // overwrites (last-writer-wins, one chain); distinct slots each
        // contribute one chain; `chains()` is exactly the captured set.
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

    fn batching_burst(settle: TimerId, deadline: TimerId, anchor: ResourceId) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: deadline,
            phase: PreFirePhase::Batching {
                settle_timer: settle,
            },
            intent: BurstIntent::Standard,
            forced: false,
            dirty: DirtyProvenance::new(),
            certified: CertifiedPrior::new(),
            probe_target: anchor,
            last_event_time: None,
        }
    }

    fn unit_pre(phase: PreFirePhase, deadline: TimerId, anchor: ResourceId) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: deadline,
            phase,
            intent: BurstIntent::Standard,
            forced: false,
            dirty: DirtyProvenance::new(),
            certified: CertifiedPrior::new(),
            probe_target: anchor,
            last_event_time: None,
        }
    }

    /// Settle on Batching returns the carried token.
    #[test]
    fn timer_token_settle_on_batching_returns_settle_timer() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pre = batching_burst(tid(7), tid(99), r);
        assert_eq!(pre.timer_token(TimerKind::Settle), Some(tid(7)));
    }

    /// BurstDeadline on any pre-fire phase returns the burst's deadline,
    /// non-Optional by construction.
    #[test]
    fn timer_token_burst_deadline_lives_on_every_prefire_phase() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        for phase in [
            PreFirePhase::Batching {
                settle_timer: tid(1),
            },
            PreFirePhase::Verifying(ProbeSlot::empty()),
            PreFirePhase::Draining,
        ] {
            let pre = unit_pre(phase, tid(42), r);
            assert_eq!(pre.timer_token(TimerKind::BurstDeadline), Some(tid(42)));
        }
    }

    /// Settle on non-Batching pre-fire phases returns None — the field
    /// is structurally absent.
    #[test]
    fn timer_token_settle_is_none_on_verifying_or_draining() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        for phase in [
            PreFirePhase::Verifying(ProbeSlot::empty()),
            PreFirePhase::Draining,
        ] {
            let pre = unit_pre(phase, tid(42), r);
            assert!(pre.timer_token(TimerKind::Settle).is_none());
        }
    }

    /// AwaitGateDeadline is type-impossible on pre-fire — returns None.
    #[test]
    fn timer_token_await_gate_is_none_on_prefire() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pre = batching_burst(tid(1), tid(2), r);
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

    /// AwaitGateDeadline on Rebasing returns None — the field doesn't
    /// exist on that variant.
    #[test]
    fn timer_token_await_gate_is_none_on_rebasing() {
        let post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );
        assert!(post.timer_token(TimerKind::AwaitGateDeadline).is_none());
    }

    /// Settle / BurstDeadline are type-impossible on post-fire — None
    /// for both phases.
    #[test]
    fn timer_token_settle_and_burst_deadline_are_none_on_postfire() {
        for phase in [
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(99),
            },
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            PostFirePhase::RebaseSettling {
                spacing_timer: tid(77),
            },
        ] {
            let post = PostFireBurst::new(BurstIntent::Standard, phase, DirtyProvenance::new());
            assert!(post.timer_token(TimerKind::Settle).is_none());
            assert!(post.timer_token(TimerKind::BurstDeadline).is_none());
        }
    }

    /// `RebaseSettle` is the post-fire `Settle`: it lives only on
    /// `RebaseSettling`'s `spacing_timer`, `None` on the other phases.
    #[test]
    fn timer_token_rebase_settle_lives_only_on_rebase_settling() {
        let settling = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::RebaseSettling {
                spacing_timer: tid(31),
            },
            DirtyProvenance::new(),
        );
        assert_eq!(settling.timer_token(TimerKind::RebaseSettle), Some(tid(31)));

        for phase in [
            PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(1),
            },
            PostFirePhase::Rebasing(ProbeSlot::empty()),
        ] {
            let post = PostFireBurst::new(BurstIntent::Standard, phase, DirtyProvenance::new());
            assert!(post.timer_token(TimerKind::RebaseSettle).is_none());
        }
    }

    /// The ceiling edge-methods are the no-bypass seam: `RebaseCeiling`
    /// projects `Some` only while `Armed`, and the two typed edges
    /// (`arm` once, `mark_reached`) are the sole way to move the
    /// lifecycle — exactly the [`CertifiedPrior`] / `note_effect_completion`
    /// discipline applied to the ceiling.
    #[test]
    fn rebase_ceiling_lifecycle_is_arm_once_no_bypass() {
        let mut post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );

        // Born NotStarted: no timer, not reached.
        assert!(post.timer_token(TimerKind::RebaseCeiling).is_none());
        assert!(!post.rebase_ceiling_reached());

        // NotStarted → Armed: the edge lands, the timer projects.
        assert!(post.arm_rebase_ceiling(tid(64)));
        assert_eq!(post.timer_token(TimerKind::RebaseCeiling), Some(tid(64)));
        assert!(!post.rebase_ceiling_reached());

        // Arm-once: a second arm is a `false` no-op, the original
        // timer is preserved (the `RebaseSettling → Rebasing` re-entry
        // must not mint a redundant heap entry).
        assert!(!post.arm_rebase_ceiling(tid(99)));
        assert_eq!(post.timer_token(TimerKind::RebaseCeiling), Some(tid(64)));

        // Armed → Reached: the terminal latches, the stale timer
        // stops projecting (it lazy-drops), `reached` flips true.
        assert!(post.mark_rebase_ceiling_reached());
        assert!(post.rebase_ceiling_reached());
        assert!(post.timer_token(TimerKind::RebaseCeiling).is_none());

        // Idempotent terminal: re-marking is a `false` no-op; arming a
        // Reached ceiling is rejected (no Reached → Armed regression).
        assert!(!post.mark_rebase_ceiling_reached());
        assert!(!post.arm_rebase_ceiling(tid(7)));
        assert!(post.rebase_ceiling_reached());

        // `mark_reached` from `NotStarted` is the loud-regression
        // signal (`false`) — the ceiling only latches from `Armed`.
        let mut fresh = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );
        assert!(!fresh.mark_rebase_ceiling_reached());
        assert!(!fresh.rebase_ceiling_reached());
    }

    /// An `Active(PostFire)` burst folds its own `certified` N=2 proof
    /// (`Some`, never `None`) — the post-fire rebase loop is a
    /// legitimate fold site, not a regression. `None ⇔ Idle | Pending`
    /// is produced only one layer up, at `ProfileState`.
    #[test]
    fn post_fire_advance_quiescence_folds_its_own_certified() {
        let h = 0xabc1_u128;
        let mut burst = ActiveBurst::PostFire(PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        ));

        // First Authoritative sample: no prior ⇒ Unstable (`Some`, not
        // a `None` structural-regression).
        assert_eq!(
            burst.advance_quiescence(ProofAuthority::Authoritative, h),
            Some(QuiescenceVerdict::Unstable),
        );
        // Second equal Authoritative sample: N=2 settled ⇒ Stable.
        assert_eq!(
            burst.advance_quiescence(ProofAuthority::Authoritative, h),
            Some(QuiescenceVerdict::Stable),
        );

        // The fold lives on the post-fire burst's own `certified`
        // (a fresh PostFireBurst opens a fresh N=2 sequence).
        let mut fresh = ActiveBurst::PostFire(PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        ));
        assert_eq!(
            fresh.advance_quiescence(ProofAuthority::Authoritative, h),
            Some(QuiescenceVerdict::Unstable),
            "a fresh post-fire burst does not inherit the prior carrier's sample",
        );

        // The only `None` is `Idle | Pending` at the ProfileState
        // layer; `Active(PostFire)` always folds `Some`.
        let mut idle = ProfileState::Idle;
        assert!(
            idle.advance_quiescence(ProofAuthority::Authoritative, h)
                .is_none()
        );
        let mut active = ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst::new(
                BurstIntent::Standard,
                PostFirePhase::Rebasing(ProbeSlot::empty()),
                DirtyProvenance::new(),
            )),
            BurstFinish::ReturnToIdle,
        );
        assert_eq!(
            active.advance_quiescence(ProofAuthority::Authoritative, h),
            Some(QuiescenceVerdict::Unstable),
        );
    }

    /// `CertifiedPrior::advance` exhaustively — the N=2 quiescence
    /// floor shared by both carriers. Covers `authority × prior ×
    /// response`: the `None`-prior bootstrap, the equal/changed
    /// `Authoritative` verdicts, and the load-bearing `Undischarged`
    /// **withhold** (the prior is never advanced by an unread sample,
    /// so the next sample is never compared against an unread hash).
    #[test]
    fn certified_prior_advance_is_the_n2_floor() {
        let h1 = 0x1111_u128;
        let h2 = 0x2222_u128;
        let unread: std::sync::Arc<std::path::Path> =
            std::sync::Arc::from(std::path::Path::new("first/unread"));
        let undischarged = || ProofAuthority::Undischarged {
            first_unread: std::sync::Arc::clone(&unread),
        };

        // Fresh: no public setter; `== new()` ⟺ no Authoritative
        // sample recorded yet.
        let mut c = CertifiedPrior::new();
        assert_eq!(c, CertifiedPrior::default());

        // Bootstrap: prior `None` ⇒ first Authoritative sample is
        // `Unstable` by construction, and records the response.
        assert_eq!(
            c.advance(ProofAuthority::Authoritative, h1),
            QuiescenceVerdict::Unstable,
        );
        assert_ne!(
            c,
            CertifiedPrior::new(),
            "an Authoritative sample advanced the prior"
        );
        // Equal next sample ⇒ Stable (the N=2 close).
        assert_eq!(
            c.advance(ProofAuthority::Authoritative, h1),
            QuiescenceVerdict::Stable,
        );
        // Changed sample ⇒ Unstable, and the prior re-bases to it.
        assert_eq!(
            c.advance(ProofAuthority::Authoritative, h2),
            QuiescenceVerdict::Unstable,
        );
        assert_eq!(
            c.advance(ProofAuthority::Authoritative, h2),
            QuiescenceVerdict::Stable,
        );

        // Undischarged **withhold**: returns the verdict carrying
        // `first_unread` verbatim, and does NOT advance the prior — the
        // following equal-to-the-pre-withhold-prior sample is still
        // `Stable`, proving the unread `h1` never became the prior.
        let v = c.advance(undischarged(), h1);
        assert!(
            matches!(&v, QuiescenceVerdict::Undischarged { first_unread }
                if first_unread.as_ref() == std::path::Path::new("first/unread")),
            "Undischarged carries first_unread verbatim; got {v:?}",
        );
        assert_eq!(
            c.advance(ProofAuthority::Authoritative, h2),
            QuiescenceVerdict::Stable,
            "the Undischarged sample did not advance the prior off h2",
        );

        // Undischarged on a fresh prior leaves it `None` (still
        // `== new()`): the next Authoritative sample is `Unstable`, not
        // compared against the unread hash.
        let mut fresh = CertifiedPrior::new();
        let _ = fresh.advance(undischarged(), h1);
        assert_eq!(
            fresh,
            CertifiedPrior::new(),
            "Undischarged on a None prior is a pure withhold — prior unchanged",
        );
        assert_eq!(
            fresh.advance(ProofAuthority::Authoritative, h1),
            QuiescenceVerdict::Unstable,
            "post-withhold the prior is still None ⇒ bootstrap Unstable, not a false Stable",
        );
    }

    /// ActiveBurst delegates to the held inner type.
    #[test]
    fn active_burst_timer_token_dispatches_by_lifecycle() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pre = ActiveBurst::PreFire(batching_burst(tid(3), tid(4), r));
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
            TimerKind::RebaseSettle,
            TimerKind::RebaseCeiling,
        ] {
            assert!(s.timer_token(k).is_none());
        }
    }

    /// ProfileState::Pending owns no timers (descent uses the probe
    /// channel for correlation, not a heap timer).
    #[test]
    fn profile_state_timer_token_pending_returns_none_for_every_kind() {
        let s = ProfileState::Pending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
        ));
        for k in [
            TimerKind::Settle,
            TimerKind::BurstDeadline,
            TimerKind::AwaitGateDeadline,
            TimerKind::RebaseSettle,
            TimerKind::RebaseCeiling,
        ] {
            assert!(s.timer_token(k).is_none());
        }
    }

    /// ProfileState::Active delegates to the held ActiveBurst.
    #[test]
    fn profile_state_timer_token_active_delegates_to_burst() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let state = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(11), tid(12), r)),
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
            ActiveBurst::PreFire(unit_pre(PreFirePhase::Draining, tid(1), r)),
            BurstFinish::ReturnToIdle,
        );
        assert!(draining.is_draining());

        // BurstFinish doesn't influence the predicate.
        let draining_reap = ProfileState::Active(
            ActiveBurst::PreFire(unit_pre(PreFirePhase::Draining, tid(1), r)),
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
            )),
            ProfileState::Active(
                ActiveBurst::PreFire(unit_pre(
                    PreFirePhase::Verifying(ProbeSlot::empty()),
                    tid(1),
                    r,
                )),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PreFire(batching_burst(tid(1), tid(2), r)),
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
                    PostFirePhase::RebaseSettling {
                        spacing_timer: tid(4),
                    },
                    DirtyProvenance::new(),
                )),
                BurstFinish::ReturnToIdle,
            ),
        ] {
            assert!(!state.is_draining(), "expected !is_draining for {state:?}");
        }
    }

    /// `descent_state` borrows the inner state in `Pending`, returns
    /// `None` for every other variant.
    #[test]
    fn descent_state_returns_some_only_on_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let descent = DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
            ProbeSlot::empty(),
        );
        let pending = ProfileState::Pending(descent);
        assert!(pending.descent_state().is_some());

        assert!(ProfileState::Idle.descent_state().is_none());
        let active = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2), r)),
            BurstFinish::ReturnToIdle,
        );
        assert!(active.descent_state().is_none());
    }

    /// `descent_state_mut` lets a caller advance the descent in place
    /// when the state is `Pending`.
    #[test]
    fn descent_state_mut_lets_caller_advance_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut state = ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a"), CompactString::from("b")])
                .expect("non-empty"),
            ProbeSlot::empty(),
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

    /// `probe_correlation` projects the Pending descent slot;
    /// `take_probe` consumes it once and idles it. Both are total over
    /// the state space — Idle and Active carry no descent slot.
    #[test]
    fn probe_correlation_and_take_probe_track_pending_slot() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let c = ProbeCorrelation::from(42);

        let armed = || {
            ProfileState::Pending(DescentState::new(
                r,
                DescentRemaining::from_vec(vec![CompactString::from("a")]).expect("non-empty"),
                ProbeSlot::armed(c, ()),
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
        ));
        assert_eq!(idle_pending.probe_correlation(), None);
        assert_eq!(idle_pending.take_probe(), None);

        // Idle / Active hold no descent slot — total projection ⇒ None.
        assert_eq!(ProfileState::Idle.probe_correlation(), None);
        assert_eq!(ProfileState::Idle.take_probe(), None);
        let mut active = ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2), r)),
            BurstFinish::ReturnToIdle,
        );
        assert_eq!(active.probe_correlation(), None);
        assert_eq!(active.take_probe(), None);
    }

    // -----------------------------------------------------------------------
    // State-machine setter / accessor API (clear_anchor_classification,
    // materialize_anchor, transition_state, anchor_claim setters,
    // burst projections, read accessors, delegators, take_current)
    // -----------------------------------------------------------------------

    use super::AnchorClaim;

    fn pending(r: ResourceId) -> ProfileState {
        ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("seg")]).expect("non-empty"),
            ProbeSlot::empty(),
        ))
    }

    fn active_prefire(r: ResourceId) -> ProfileState {
        ProfileState::Active(
            ActiveBurst::PreFire(batching_burst(tid(1), tid(2), r)),
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

    /// The zero-edge: 3 → 2 → 1 → 0 reports `Decremented` until the last
    /// completion, then `LastReached`.
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

    /// Rebasing carries no counter — a late completion the post-fire
    /// counter no longer tracks.
    #[test]
    fn note_effect_completion_on_rebasing_is_not_awaiting() {
        let mut post = PostFireBurst::new(
            BurstIntent::Standard,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            DirtyProvenance::new(),
        );
        assert_eq!(post.note_effect_completion(), AwaitVerdict::NotAwaiting);
    }

    /// Over-completion (more `EffectComplete`s than emitted Effects) is
    /// an invariant breach — the dev/CI floor backstop.
    #[test]
    #[should_panic(expected = "outstanding underflow")]
    fn note_effect_completion_underflow_trips_assert() {
        let _ = awaiting_post(0).note_effect_completion();
    }

    /// `Profile` delegates through the state machine: `NotAwaiting` for
    /// every non-Awaiting state, the live verdict on `Active(Awaiting)`.
    #[test]
    fn note_effect_completion_delegates_through_profile() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert_eq!(p.note_effect_completion(), AwaitVerdict::NotAwaiting);
        p.transition_state(pending(r));
        assert_eq!(p.note_effect_completion(), AwaitVerdict::NotAwaiting);
        p.transition_state(active_prefire(r));
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

    /// The residual provenance seeds the restart: the typed move re-arms
    /// a fresh `Batching` Standard burst with the engine-minted timers
    /// and the anchor placeholder, carries the captured paths over
    /// whole (so the restarted burst's first verify obligates over
    /// them), and opens a fresh quiescence sequence (`certified` resets
    /// to a fresh [`CertifiedPrior`]).
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
            anchor,
            now,
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
        assert_eq!(pre.probe_target, anchor);
        assert_eq!(pre.last_event_time, Some(now));
        assert_eq!(
            pre.certified,
            CertifiedPrior::new(),
            "a restarted burst opens a fresh N=2 quiescence sequence",
        );
    }

    /// A Seed-origin residual restarts just as a Standard one does: the
    /// move is origin-agnostic and *sets* `intent: Standard` (a
    /// restarted debounce burst is Standard by definition). This is the
    /// closed Seed-residual event-loss — a Seed drift → fire → rebase
    /// with absorbed events rejoins the Standard debounce lifecycle
    /// instead of being dropped. No origin gate, no panic; the
    /// reconfirm is a fresh query, so there is no per-origin balance to
    /// keep.
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
            anchor,
            std::time::Instant::now(),
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
        assert_eq!(
            pre.certified,
            CertifiedPrior::new(),
            "a restarted burst opens a fresh N=2 quiescence sequence",
        );
    }

    /// An empty residual is a misuse — the restart would have no seed and
    /// would mask a caller that failed to gate on a non-empty fire-tail.
    #[test]
    #[should_panic(expected = "empty residual")]
    fn into_pre_fire_residual_empty_residual_trips_assert() {
        let mut tree = Tree::new();
        let anchor = tree.ensure_root("anchor", ResourceRole::User);
        let _ = rebasing_post(BurstIntent::Standard, DirtyProvenance::new())
            .into_pre_fire_residual(tid(1), tid(2), anchor, std::time::Instant::now());
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
        // Already-lost shape: classified but baseline cleared at the
        // prior loss, only the survival witness remains.
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

    /// Recovery path: descent re-materialises an anchor that lost its
    /// baseline. The survival witness held on the `Unclassified`
    /// anchor must survive classification so the post-recovery Seed-Ok
    /// drift verdict still has a reference (states B → C).
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

        p.transition_state(active_prefire(r));
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
        p.transition_state(active_prefire(r));
        assert!(
            p.post_fire_burst_mut().is_none(),
            "PreFire has no post-fire"
        );

        p.transition_state(active_postfire());
        let post = p
            .post_fire_burst_mut()
            .expect("PostFire carries the payload");
        post.dirty.note(r, Arc::from(Path::new("/w/anchor")));
        assert!(
            p.post_fire_burst_mut()
                .expect("still PostFire")
                .dirty
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

        // take_probe delegates to ProfileState::take_probe: arming the
        // Pending descent slot through the typed mint edge, then taking
        // it idles the machine state (the take is the linear consume,
        // so the slot is never dropped armed).
        p.descent_state_mut()
            .expect("still Pending")
            .arm_probe(ProbeCorrelation::from(7));
        assert_eq!(p.take_probe(), Some(ProbeCorrelation::from(7)));
        assert_eq!(p.take_probe(), None, "delegate idled the slot");

        // mark/clear_active_for_reap delegate the bool semantics.
        let mut q = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!q.mark_active_for_reap(), "Idle cannot be marked");
        assert!(!q.clear_active_reap(), "Idle has nothing to clear");
        q.transition_state(active_prefire(r));
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

    /// Guarded random walk over the public anchor mutators, asserting
    /// after every op that the projection surface stays consistent
    /// with the underlying sum and that every reachable shape is one
    /// of the documented states. The snapshot-shape and
    /// baseline/witness-exclusion invariants are *structural* (no
    /// representable `AnchorClassification` violates them) — these
    /// assertions are the defense-in-depth tripwire that would catch a
    /// future flat-field regression or a projection bug. Guards
    /// respect each mutator's documented precondition so a step trips
    /// the consistency check, never a precondition `debug_assert!`.
    /// Deterministic xorshift64 PRNG, seed pinned in the fn name; 16
    /// fresh Profiles so the one-shot `materialize_anchor` is exercised.
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

        // Every public projection must agree with the underlying sum,
        // and the shape must be one of the eight reachable rows.
        fn assert_invariants(p: &Profile, op: &str) {
            let current = p.current();
            let baseline = p.baseline();

            // Snapshot-shape: when both present they share a variant;
            // kind tracks the current variant.
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
                // `baseline` is exposed iff `settled` is an active
                // `Snapshot`; a `Witness` is a survival hash, not a
                // live baseline. `Snapshot` xor `Witness` is
                // structural, so this can never observe both. The
                // File / Dir arms differ only in the `settled`
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
                // Ops 0..=5 are the six anchor-classification mutators
                // (install_dir / install_file / clear / materialize /
                // rebase / take_current); ops 6..=9 drive the orthogonal
                // ProfileState axis so the materialize precondition
                // (`Pending`) is reachable mid-walk.
                match rng.below(10) {
                    0 => {
                        // Precondition: not File-classified (cross-arm
                        // graft is a dispatcher-boundary breach).
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
                        // Total across the sum, no precondition:
                        // Unclassified ⇒ no-op `None`; File/Dir ⇒ takes
                        // `current`, leaving `settled` untouched (states
                        // E→F / D→C). The only route to state F
                        // (`current` None ∧ `settled` Snapshot) in the
                        // interleaved walk — the standalone unit test
                        // can't catch a *sequenced* coherence regression.
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
                        p.transition_state(active_prefire(r));
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

    /// `Profile::new`'s `kind` → sum projection is total: `None` ⇒
    /// `Unclassified` (state A), `Some(Dir)` / `Some(File)` ⇒ a
    /// classified anchor with no snapshot or baseline (state C′).
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

    /// `Some(ResourceKind::Unknown)` is defensively dead — the sole
    /// production caller threads `Resource::kind()` which maps
    /// `Unknown → None`. Release builds degrade to `Unclassified`
    /// (same shape as `None`) rather than constructing an illegal
    /// state; debug builds trip the `debug_assert!`.
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

    /// `settled_hash` is the one total drift reference across the sum:
    /// not-yet-settled ⇒ `None`; active baseline ⇒ its digest;
    /// loss-window witness ⇒ the retained hash; carried after a clear.
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

    /// `debug_assert_anchor_coherent` enforces the residual
    /// cross-axis invariant `Pending ⇒ Unclassified ∧ ¬Held`. The
    /// happy path (every shape outside `Pending`, or `Pending` while
    /// `Unclassified`) is silent; a classified `Pending` trips.
    #[test]
    fn anchor_coherent_is_silent_on_reachable_shapes() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = mk_profile(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.debug_assert_anchor_coherent(); // Idle + Unclassified
        p.transition_state(pending(r));
        p.debug_assert_anchor_coherent(); // Pending + Unclassified ✓
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
