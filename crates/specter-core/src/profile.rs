//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is computed at construction from
//! `(config, max_settle)` and is the lifetime-stable identity of the Profile.
//! `ProfileMap` keeps `(resource, config_hash) → ProfileId` and updates
//! `Resource.profiles` in lockstep — `attach`/`detach` are the only mutators
//! of either index.

use crate::effect::DedupKey;
use crate::ids::{ProfileId, ResourceId, TimerId};
use crate::op::ProbeCorrelation;
use crate::resource::ResourceKind;
use crate::scan_config::{ScanConfig, compute_config_hash};
use crate::snapshot::tree::TreeSnapshot;
use crate::sub::ClassSet;
use crate::tree::Tree;
use compact_str::CompactString;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tinyvec::TinyVec;

/// One fire cycle.
///
/// A `Burst` lives `Idle → Active(Burst) → Idle`; its `phase` walks
/// `Batching → Verifying [→ Draining → Verifying] → Awaiting → Rebasing`
/// and its `intent` (`Standard | Seed`) decides the terminal action.
///
/// Burst-level state (`intent`, `forced`, `burst_deadline`) survives every
/// phase transition; `phase` carries the **correlation token of the input
/// the burst is currently waiting on**:
/// - `Batching { settle_timer }` — armed debounce timer; the burst is
///   waiting for a quiet gap (or a fresh `FsEvent` to extend it).
/// - `Verifying` — probe in flight. The probe correlation lives on
///   [`Profile::pending_probe`] (the per-Profile probe-channel slot, the
///   single source of truth for "what probe?"); this variant carries no
///   payload of its own.
/// - `Draining` — self-stable, descendant Profiles still resolving;
///   correlated externally by `Profile.dirty_descendants`.
/// - `Awaiting { outstanding, gate_deadline }` — Effects emitted; the
///   engine is waiting for `outstanding` `EffectComplete` arrivals from
///   the actuator. `gate_deadline` is a recovery timer for actuator
///   hangs. Reaching `outstanding == 0` transitions to `Rebasing`.
/// - `Rebasing` — post-fire probe in flight at the anchor. The probe's
///   response captures the post-command tree as the new baseline; the
///   correlation slot is the same `Profile::pending_probe` reused
///   (Verifying and Rebasing are time-disjoint within one burst).
///
/// `dirty_resources` and `force_walk_resources` are accumulators consumed
/// at every `transition_to_verifying`; `probe_target` survives Verifying
/// → Draining → Verifying so the reconfirm probe reuses the original LCA.
///
/// `last_event_time` is the source of truth for the settle deadline: the
/// settle timer is scheduled once on Batching entry and reschedules on
/// expiry only when `last_event_time` has advanced since. Event arrivals
/// while already in Batching update this field but do **not** re-insert
/// a fresh heap entry — heap inserts are bounded to one per
/// `last_event_time + settle` boundary, regardless of event density.
#[derive(Debug)]
pub struct Burst {
    pub burst_deadline: TimerId,
    pub phase: BurstPhase,
    pub intent: BurstIntent,
    pub forced: bool,
    /// Resources whose `FsEvent` drove (or is driving) this burst.
    /// Populated by `start_standard_burst` (`{ event_resource }` seed)
    /// and `event_drives_batching` (each FsEvent during `Active`'s
    /// pre-fire phases — `Batching | Verifying | Draining`). Cleared
    /// when the `Burst` is dropped (`finish_burst_to_idle`). Used to
    /// compute the LCA target at every `transition_to_verifying`. Not
    /// extended on the post-fire absorb path: the rebase probe targets
    /// the anchor unconditionally, with no LCA to compute.
    pub dirty_resources: BTreeSet<ResourceId>,
    /// Set of resources whose snapshots the next probe must visit
    /// fresh, defeating the walker's coarse-mtime skip. Two
    /// accumulation sources, each consumed at the next probe issuance:
    /// • Pre-fire: `start_standard_burst` and `event_drives_batching`
    ///   (FsEvents during `Batching | Verifying | Draining`) seed the
    ///   set; `transition_to_verifying` consumes and clears.
    /// • Post-fire: `drive_burst`'s absorb arm (FsEvents during
    ///   `Awaiting | Rebasing`) seeds the set; `transition_to_rebasing`
    ///   consumes and clears.
    /// Events absorbed during `Rebasing` after the rebase probe is in
    /// flight have no consumer — they accumulate into the cleared
    /// field and drop at `finish_burst_to_idle`. The bounded residual
    /// window (≈ probe round-trip latency) is the v1 carve-out.
    pub force_walk_resources: BTreeSet<ResourceId>,
    /// `target_resource` of the most recently emitted probe in this burst.
    /// Mirrors the latest `ProbeRequest.target_resource`. Read by the
    /// Draining→Verifying reconfirm path (`dirty_resources` is empty
    /// there, so LCA would degenerate to the anchor — reuse the prior
    /// target instead) and by `dispatch_standard_ok` to know which
    /// subtree of `Profile.current` to compare against `response_subtree`
    /// for the stability verdict. `None` until the first probe emits.
    pub probe_target: Option<ResourceId>,
    /// Non-anchor resources whose `suppress_count` was bumped 0→1 by
    /// `event_drives_batching` during this burst's pre-fire phases.
    /// Taken (via `mem::take`) at `transition_to_verifying` to drive
    /// the symmetric `sub_suppress` drain, and defensively at
    /// `finish_burst_to_idle` for abnormal-end paths
    /// (`finalize_anchor_lost`, reap mid-burst). The take leaves the
    /// field empty for the next pre-fire cycle without a follow-up
    /// `clear()`.
    ///
    /// **Anchor explicitly excluded.** The anchor's suppress is the
    /// existing `start_*_burst → finish_burst_to_idle` lifecycle and is
    /// unrelated to this set. The exclusion is currently expressed as
    /// `event_resource != anchor` in `event_drives_batching`; a future
    /// change that adds parent-dir or other identity-floor resources to
    /// the Profile should widen the exclusion to "any resource in the
    /// Profile's identity-floor set" rather than continue to spell
    /// `event_resource != anchor` literally.
    ///
    /// Empty after every `transition_to_verifying`. Re-armed by the next
    /// `event_drives_batching` call after an unstable verify routes the
    /// burst back to Batching. Empty for `Seed` bursts (no Batching
    /// phase); the field exists for struct uniformity.
    ///
    /// `BTreeSet` (not `Vec`) so iteration order is deterministic — the
    /// `sub_suppress` drain emits `Unsuppress` ops in `ResourceId`
    /// ascending order, matching `StepOutput.watch_ops`'s sort
    /// discipline. Size is typically 0 (W_ssh — anchor-only event
    /// source) to N (W_build — distinct per-file resources receiving
    /// events during one Batching window). No allocation pressure
    /// relative to the existing `dirty_resources` /
    /// `force_walk_resources`.
    pub suppressed_resources: BTreeSet<ResourceId>,
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
    /// - `None` from `start_seed_burst` — Seed bursts transition Idle →
    ///   Active(Verifying) directly, with no Batching phase at start.
    ///   If a fresh `FsEvent` later arrives during the Seed verify
    ///   (`event_drives_batching` from the `Verifying → Batching` arm),
    ///   the field is repopulated.
    /// - Updated by `event_drives_batching` on every event, regardless
    ///   of which pre-fire phase (`Batching | Verifying | Draining`)
    ///   preceded.
    /// - **Preserved** across `unstable_response_drives_batching` — the
    ///   verify just responded; the next event repopulates. The
    ///   freshly-scheduled settle timer fires at `now + settle`; the
    ///   on-expiry handler then sees `now − last_event_time ≥ settle`
    ///   (because `now ≥ unstable_response_at + settle ≥
    ///   last_event_time + settle`) and transitions cleanly.
    /// - **Preserved** across `transition_to_verifying` (the reconfirm
    ///   path) and `transition_to_draining` — phase swaps without
    ///   semantic resets.
    /// - Cleared implicitly when the `Burst` is dropped at
    ///   `finish_burst_to_idle`.
    ///
    /// **Distinct from the watcher's `last_event_at`.** The watcher's
    /// field is per-watcher, scoped to drain-cadence recency. This field
    /// is per-burst, scoped to settle-deadline reschedule. Different
    /// consumers, different cadences.
    pub last_event_time: Option<Instant>,
}

/// What the burst is waiting on, as a discriminator.
///
/// `Batching` carries its own correlation token (`settle_timer: TimerId`)
/// because timer correlation is per-Burst and has no peer slot to live on.
/// `Verifying` is unit: the probe correlation lives on
/// [`Profile::pending_probe`] — the per-Profile probe-channel slot — so the
/// burst phase only encodes "probe in flight" as state-machine identity.
/// `Draining` is correlated externally by `Profile.dirty_descendants` and
/// carries no token of its own.
///
/// `Awaiting { outstanding, gate_deadline }` is the post-fire phase: the
/// engine has emitted Effects to the actuator and is waiting for their
/// completions to drive `outstanding → 0`. `gate_deadline` is the
/// safety-net `AwaitGateDeadline` timer — a hung child is recovered by
/// force-transitioning to `Rebasing` once the timer expires. `Rebasing`
/// is unit (post-fire probe in flight; correlation lives on
/// [`Profile::pending_probe`], same slot Verifying used — they are
/// time-disjoint within one burst by construction).
#[derive(Debug)]
pub enum BurstPhase {
    /// Activity-gap detection. `settle_timer` is the armed debounce
    /// timer; an `FsEvent` reschedules it (`event_drives_batching`),
    /// timer expiry advances to `Verifying` (`transition_to_verifying`).
    Batching { settle_timer: TimerId },
    /// Probe in flight. The matching `ProbeCorrelation` lives on
    /// [`Profile::pending_probe`]; this variant is unit because the
    /// Profile-side slot is the single source of truth (encoding the
    /// correlation twice would invite drift).
    Verifying,
    /// Self-stable; descendants pending. The stable snapshot lives on
    /// `Profile.current` — `dispatch_standard_ok` updates `current` to
    /// the stable response immediately before transitioning here, so the
    /// reconfirm probe (Draining → Verifying on `dirty_descendants → 0`)
    /// compares against `Profile.current`. Holding a duplicate
    /// `TreeSnapshot` on the variant would only invite drift between the
    /// two references.
    Draining,
    /// Effects emitted; awaiting completion(s) from the actuator.
    /// `outstanding` decrements on each `EffectComplete` for this
    /// Profile's `DedupKey`s; reaching zero transitions to `Rebasing`
    /// (or, when `Profile.reap_pending` is set, finishes the burst
    /// directly without re-probing). `gate_deadline` is the recovery
    /// timer for an actuator that never reports completion — its
    /// expiry forces the burst into `Rebasing` so the engine can
    /// re-establish a baseline against disk reality.
    Awaiting {
        outstanding: u32,
        gate_deadline: TimerId,
    },
    /// Post-fire probe in flight. Correlation lives on
    /// [`Profile::pending_probe`] (same slot Verifying uses — Verifying
    /// and Rebasing are time-disjoint within one burst). The probe's
    /// `Ok` response captures the post-command tree; `dispatch_rebase_ok`
    /// then sets `baseline := current` and finishes the burst to Idle.
    Rebasing,
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
/// - `Active(Burst)`: anchor is materialized; a stability burst is in
///   flight.
///
/// I5 (at most one outstanding probe per Profile) is enforced as a
/// **field discipline** on [`Profile::pending_probe`]: that slot holds the
/// correlation of the in-flight probe, regardless of which lifecycle state
/// drives it. Pending and Active remain mutually exclusive at the type
/// level, so the dispatch site routes a live response on state identity
/// alone (see `Engine::on_probe_response` in `specter-engine`).
#[derive(Debug, Default)]
pub enum ProfileState {
    #[default]
    Idle,
    /// Pending-path descent in flight. The anchor (`Profile.resource`) is
    /// `DescentScaffold`-roled and carries no `watch_demand` from this
    /// Profile; `DescentState.current_prefix` does. When the anchor
    /// materializes (descent's last component arrives) the engine
    /// transitions Pending → Idle (releasing the prefix's contribution and
    /// bumping the anchor's), then immediately Idle → Active(Seed) via
    /// `start_seed_burst`.
    Pending(DescentState),
    Active(Burst),
}

/// State for a Profile undergoing pending-path descent.
///
/// Lives inline on `ProfileState::Pending` for the duration of descent.
///
/// Invariants:
/// - `current_prefix` carries a `+1` `watch_demand` contribution from this
///   Profile (added at descent registration / advancement; dropped at
///   descent end or rewind).
/// - `remaining_components` is non-empty (the anchor itself is the last
///   component). Empty `remaining_components` is a state-machine bug; the
///   defensive check in the descent dispatch transitions the Profile back
///   to `Idle`.
///
/// I5 ("at most one outstanding probe per Profile") for the Pending
/// lifecycle is enforced by the per-Profile probe channel slot
/// ([`Profile::pending_probe`]) — the same slot used for Active bursts.
/// The descent's variant payload holds no probe-correlation data of its
/// own.
#[derive(Clone, Debug)]
pub struct DescentState {
    /// Deepest existing ancestor currently Watched. The Profile
    /// contributes `+1` to this Resource's `watch_demand`.
    pub current_prefix: ResourceId,
    /// Path components from `current_prefix` (exclusive) down to the
    /// anchor (inclusive). Single-component segments (no `/`).
    /// `CompactString` keeps typical-length names (≤24 bytes) inline,
    /// so advance / rewind clones avoid the heap.
    pub remaining_components: Vec<CompactString>,
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
/// `Settle` — debounce timer armed during [`BurstPhase::Batching`]. Expiry
/// drives Batching → Verifying.
/// `BurstDeadline` — Burst-level max-settle timer armed at Burst start.
/// Expiry sets `Burst.forced = true` and dispatches by current phase. The
/// timer is structurally relevant only in pre-fire phases (`Batching`,
/// `Verifying`, `Draining`); once the burst transitions to `Awaiting` the
/// fire has already happened, the deadline is moot, and a stale fire is
/// dropped silently by the validation in `Engine::is_timer_referenced`
/// (in `specter-engine`).
/// `AwaitGateDeadline` — recovery timer armed at
/// [`BurstPhase::Awaiting`] entry. Expiry indicates the actuator is
/// taking longer than expected (likely a hung child); the engine
/// force-transitions to `Rebasing` to re-establish a baseline against
/// disk reality.
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
}

/// Lifecycle of a Profile's anchor `watch_demand` contribution.
///
/// Two-state machine:
/// - [`Self::None`] — Profile holds no anchor contribution. Reachable
///   when the Profile is `Pending` (descent prefix carries the
///   STRUCTURE watch instead), `Purged` (`Input::WatchOpRejected`
///   clamped the slot), or freshly constructed pre-attach.
/// - [`Self::Held`] — Profile contributes `+1 events_union` to its
///   anchor's `watch_demand`. Set on the path that bumped the counter
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

#[derive(Debug)]
pub struct Profile {
    pub resource: ResourceId,
    pub config: ScanConfig,
    pub exclude_strings: Arc<[CompactString]>,
    pub config_hash: u64,
    /// Cached classification of the anchor — the on-disk shape Specter
    /// observed at the resource. Set on the path that first learned the
    /// kind (resource-based attach with a classified slot, descent
    /// materialization, or first Seed-Ok response) and invariant for the
    /// rest of the Profile's lifetime: anchor kind changes on disk surface
    /// as `Vanished` at probe time, never as a mid-life mutation here.
    ///
    /// **Lifecycle.**
    /// - `None` while the Profile's `state` is `Pending` (descent has not
    ///   yet materialised the anchor) **or** while a fresh resource-based
    ///   attach hasn't yet seen its first probe response (the Resource
    ///   slot was `Unknown` at attach-time and no probe has classified it).
    /// - `None` after anchor loss (`Vanished` / `Failed` /
    ///   `WatchOpRejected` on the anchor) — `Engine::discard_anchor_state`
    ///   clears the cache so the next Seed burst routes through the
    ///   kind-agnostic Subtree probe and avoids a wasted round-trip
    ///   against a recreated anchor of a different on-disk shape.
    /// - `Some(kind)` from the materialisation moment (descent → Idle, or
    ///   first Seed-Ok dispatch, or resource-based attach against a
    ///   classified slot) until the next anchor loss.
    ///
    /// The "first observation wins" invariant applies **within a single
    /// materialised epoch**. Across recovery cycles the cache is
    /// deliberately invalidated so the kind-shape dispatch doesn't
    /// misroute against a recreated anchor of a different shape.
    ///
    /// **Why a Profile field, not a Tree lookup.** Engine-side dispatch
    /// sites need the anchor's kind on the hot path: the burst-launch
    /// helpers (`start_seed_burst`, `transition_to_verifying`,
    /// `transition_to_rebasing`) read it to choose between
    /// `emit_anchor_probe` and `emit_subtree_probe`, and `emit_effects`
    /// reads it via `compute_cwd`. Each previously did
    /// `tree.get(profile.resource).and_then(Resource::kind)` with a
    /// hand-rolled fallback for the unprobed case. Caching once on the
    /// Profile removes the per-dispatch lookup, lets the call sites read
    /// the invariant directly, and centralises the fallback rationale on
    /// this field's documentation rather than repeating it at each reader.
    ///
    /// **Reader convention.** Probe-shape dispatch matches the variant
    /// directly (`Some(File) ⇒ AnchorFile`,
    /// `Some(Dir | Unknown) | None ⇒ Subtree`); `Vanished` from a
    /// kind-mismatched `Subtree` probe routes to descent recovery. The
    /// `compute_cwd` reader uses `kind.unwrap_or(Dir)` since a Dir cwd at
    /// the path itself is recoverable via `EffectOutcome::Failed` if the
    /// path turns out to be a File.
    ///
    /// **Coherence with `Resource.kind`.** The Tree slot's `kind` field
    /// (`Resource::kind`) is a parallel cache of the same observation,
    /// updated by reconcile and explicit `Tree::set_kind` calls. The
    /// engine reads `Profile.kind` for anchor-kind decisions; it does
    /// not consult `Tree.kind` for the anchor in any post-attach path.
    /// The Tree-side cache may stay stale across an
    /// anchor-loss-recover cycle for shared anchors (the slot survives
    /// because other Profiles anchor it). No production reader sees
    /// the stale value because no engine path consults
    /// `tree.get(anchor).kind` for the anchor's own kind in the
    /// post-loss window — the invariant is "engine reads
    /// `Profile.kind`, never `Tree.kind` for the anchor's kind."
    /// Future write sites that introduce such a reader must
    /// invalidate the Tree-side cache at the appropriate sites.
    ///
    /// **Snapshot-shape invariant.** When `current.is_some()`, the
    /// `TreeSnapshot` variant must agree with `kind`:
    /// `current = Some(TreeSnapshot::File(_)) ⇒ kind == Some(File)`;
    /// `current = Some(TreeSnapshot::Dir(_)) ⇒ kind == Some(Dir)`. The
    /// engine's typed [`crate::ProbeRequest`] / [`crate::ProbeOutcome`]
    /// dispatch chain enforces this at runtime — not at compile time —
    /// so the invariant is narrative; a sum-typed `current` would
    /// type-enforce it but at the cost of every kind-agnostic reader of
    /// `current` and `baseline` paying a per-variant dispatch tax. Any
    /// future write site that mutates `current` and `kind` independently
    /// must preserve the agreement; `Engine::discard_anchor_state`
    /// clears both atomically inside one `Engine::step`.
    pub kind: Option<ResourceKind>,
    pub state: ProfileState,
    /// Engine-side slot for the **probe channel** — the per-Profile
    /// communication primitive between the engine and the Prober pool.
    /// Holds the correlation token of an outstanding `ProbeRequest`, or
    /// `None` if no probe is in flight.
    ///
    /// **Discipline.** Open via `Engine::mint_probe_correlation`; close
    /// via the response-dispatch path (top of `Engine::on_probe_response`)
    /// or via `Engine::cancel_pending_probe`. Open for at most one
    /// outstanding request, regardless of which lifecycle state
    /// (`Pending` or `Active`) drives the emission.
    ///
    /// **Sibling channels.** Distinct from the *watch channel*
    /// (per-Resource, refcounted via `watch_demand`) and the *effect
    /// channel* (per-(Sub, DedupKey), coalesced in the Actuator).
    pub pending_probe: Option<ProbeCorrelation>,
    pub baseline: Option<TreeSnapshot>,
    pub current: Option<TreeSnapshot>,
    /// Cached nearest covering ancestor Profile — the parent edge
    /// `propagate` walks at burst-start (`+1`) and burst-end (`-1`).
    /// `None` for root Profiles whose ancestor chain holds no
    /// covering Profile. Re-resolved engine-side at fresh-Profile
    /// attach, interpose-attach, and parent reap; the cache keeps
    /// `propagate`'s hot path at O(depth) chain reads (recomputing
    /// from `covers(P, R)` per step would be O(depth² ×
    /// profiles_per_resource) with a PathBuf allocation per call).
    ///
    /// **Discipline.** Engine writes converge on the
    /// `stability::write_parent_edge` helper, the single source of
    /// the self-parent `debug_assert_ne!`. Direct field assignment
    /// is reserved for testkit / unit-test setup — same convention
    /// as `pending_probe`.
    pub parent_profile: Option<ProfileId>,
    pub dirty_descendants: u32,
    pub sub_refcount: u32,
    pub max_settle: Duration,
    /// Settle interval driving `start_standard_burst` and the backoff base.
    /// Cached on construction from the first attached Sub; the engine
    /// recomputes this as `min(remaining_subs.settles)` on `attach_sub`
    /// (existing Profile) and `detach_sub`.
    pub settle: Duration,
    /// True iff the last Sub on this Profile was detached while a burst was
    /// in flight. The active burst runs to completion; `finish_burst_to_idle`
    /// checks this flag, suppresses Effect emission, and reaps the Profile.
    pub reap_pending: bool,
    /// Cached parent Resource that this Profile contributes a watch to.
    /// `attach_sub` sets it; `detach_sub` releases the contribution via the
    /// cached id without re-deriving the parent. `None` if the anchor is
    /// itself a root (no parent in the Tree) — root rename detection is then
    /// unavailable.
    pub watch_root_parent: Option<ResourceId>,
    /// Tracks whether this Profile currently holds a `+1` contribution on
    /// `resource.watch_demand` — [`AnchorClaim::Held`] on the path that
    /// called `add_watch_demand(anchor)` (immediate-Seed in
    /// `attach_sub_inner` or descent's anchor materialization), cleared
    /// to [`AnchorClaim::None`] on the matching `sub_watch_demand(anchor)`
    /// (anchor terminal event, reap, clamp purge).
    ///
    /// The claim distinguishes three reap-time lifecycle states that
    /// otherwise look identical in the Profile/descent registry:
    /// **materialized** (`Held` ⇒ release anchor), **pending**
    /// (descent in flight ⇒ release descent prefix instead), and
    /// **purged** (`None`, descent already removed by
    /// `Input::WatchOpRejected` ⇒ no contribution to release; the clamp
    /// already did it).
    ///
    /// Without this field a heuristic like `baseline.is_some() ||
    /// current.is_some()` undercounts `dispatch_seed_vanished` paths
    /// (which clear the snapshots while leaving the anchor's contribution
    /// intact) and a heuristic like `tree.get(anchor).watch_demand > 0`
    /// overcounts in multi-Profile sharing (would steal another
    /// Profile's contribution).
    pub anchor_claim: AnchorClaim,
    /// Set of `DedupKey`s for which this Profile has emitted at least one
    /// Effect that has not been cleared by a `Failed` outcome,
    /// `detach_sub`, or covered-leaf reap. Pure existence — no value
    /// payload. Drives drift recovery's "should we conservative-fire?"
    /// question by gating the `SeedDrift` filter; B1 dedup derives
    /// directly from `baseline.hash() == current.hash()` and does not
    /// consult this field.
    ///
    /// **Lifecycle.** Inserted at successful emit (`emit_effects` Subtree
    /// and PerFile arms). Removed on `EffectComplete::Failed`,
    /// `detach_sub_inner`, and `purge_per_file_fired_subs_for_reaped_slots`.
    /// Preserved across anchor loss by `discard_anchor_state` — the fire
    /// history is the answer to "which Subs should re-fire on recovery if
    /// drift is detected?"
    pub fired_subs: BTreeSet<DedupKey>,
    /// Anchor-rooted snapshot hash of `baseline` at the moment of
    /// `discard_anchor_state` — the survival witness used by
    /// `seed_drift_observed` to detect post-recovery drift after
    /// `baseline` has been cleared. `None` whenever `baseline.is_some()`.
    ///
    /// **Lifecycle.** Set by [`Profile::capture_witness_at_loss`] (called
    /// from `discard_anchor_state`, only when `baseline` was `Some` at the
    /// time of loss). Cleared by [`Profile::rebase_baseline`] (called
    /// from `dispatch_seed_ok` — both branches — and `dispatch_rebase_ok`).
    ///
    /// **Cross-field invariant.** `baseline.is_some() ⇒
    /// last_settled_hash_at_loss.is_none()`. The witness exists *only*
    /// during the survival window between anchor loss and recovery.
    /// Active-mode drift detection consults `baseline` directly; the
    /// witness substitutes for `baseline.hash()` once `baseline` is
    /// cleared.
    pub last_settled_hash_at_loss: Option<u128>,
    /// User-declared event-class mask for this Profile. Every Sub on a
    /// Profile shares the same `events` by construction (mask folds into
    /// `config_hash`), so this field is the Sub's mask — the "union"
    /// naming is structural: per-Sub contributions OR onto the
    /// Profile's mask, even though the OR is a no-op here. The
    /// per-Resource `events_union` aggregated across covering Profiles
    /// reads this as the per-Profile contribution.
    pub events_union: ClassSet,
    /// True iff covered Leaves need their own FDs. Derived at construction
    /// from `events.intersects(CONTENT | METADATA)` and invariant for the
    /// Profile's lifetime (events are part of `config_hash`, so a mask
    /// change forks a new Profile rather than flipping this flag).
    ///
    /// The walker-side reconciler reads this to decide whether covered
    /// Leaf children get `add_watch_demand` (per-file FDs for in-place
    /// edit detection — closes E2E #3 by default for `subtree-root` Subs
    /// whose default mask includes CONTENT).
    pub has_per_file_fds: bool,
}

impl Profile {
    /// Construct a fresh Profile: state `Idle`, no baseline/current,
    /// refcounts at zero, no reap pending, no watch-root parent recorded.
    /// `config_hash` is computed from `(config, max_settle, events)` and
    /// is stable for the Profile's lifetime — there is no path to a
    /// Profile with an unset or stale hash.
    ///
    /// `events` becomes the Profile's `events_union` and drives
    /// `has_per_file_fds` (true iff CONTENT or METADATA is in the mask).
    /// Every Sub on a Profile shares the same `events`, so
    /// `events_union` is invariant for the Profile's lifetime.
    ///
    /// `exclude_strings` is projected once here from `config.exclude` —
    /// the [`ScanConfig`] builder has already sorted the vector by source,
    /// so the projection is canonical without re-sorting.
    #[must_use]
    pub fn new(
        resource: ResourceId,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        events: ClassSet,
    ) -> Self {
        let config_hash = compute_config_hash(&config, max_settle, events);
        let has_per_file_fds = events.intersects(ClassSet::CONTENT | ClassSet::METADATA);
        let exclude_strings: Arc<[CompactString]> = config
            .exclude
            .iter()
            .map(|g| CompactString::from(g.source()))
            .collect();
        Self {
            resource,
            config,
            exclude_strings,
            config_hash,
            kind: None,
            state: ProfileState::Idle,
            pending_probe: None,
            baseline: None,
            current: None,
            parent_profile: None,
            dirty_descendants: 0,
            sub_refcount: 0,
            max_settle,
            settle,
            reap_pending: false,
            watch_root_parent: None,
            anchor_claim: AnchorClaim::None,
            fired_subs: BTreeSet::new(),
            last_settled_hash_at_loss: None,
            events_union: events,
            has_per_file_fds,
        }
    }

    /// Reassert active mode after a rebase: lift `current` into `baseline`
    /// (Arc bump on `Dir`, copy on `File`) and clear the survival witness.
    /// Called from `dispatch_rebase_ok` and from both branches of
    /// `dispatch_seed_ok` after a successful graft.
    ///
    /// **Post-condition.** Cross-field invariant
    /// `baseline.is_some() ⇒ last_settled_hash_at_loss.is_none()` holds at
    /// exit (assuming `current.is_some()` at entry, which holds at every
    /// post-graft call site).
    pub fn rebase_baseline(&mut self) {
        self.baseline = self.current.clone();
        self.last_settled_hash_at_loss = None;
    }

    /// Capture the survival witness from `baseline` at anchor loss. Called
    /// from `discard_anchor_state` immediately before the helper clears
    /// `baseline = None`. Idempotent against `baseline.is_none()` —
    /// leaves any previously-captured witness in place rather than
    /// overwriting with `None`.
    ///
    /// **Post-condition (when `baseline.is_some()` at entry).**
    /// `last_settled_hash_at_loss == Some(baseline.hash())` at exit; the
    /// caller then clears `baseline` to honour the cross-field invariant.
    pub fn capture_witness_at_loss(&mut self) {
        if let Some(b) = self.baseline.as_ref() {
            self.last_settled_hash_at_loss = Some(b.hash());
        }
    }
}

#[derive(Debug, Default)]
pub struct ProfileMap {
    profiles: SlotMap<ProfileId, Profile>,
    by_resource: SecondaryMap<ResourceId, TinyVec<[(u64, ProfileId); 1]>>,
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
        let hash = profile.config_hash;
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
            v.retain(|(h, pid)| !(*pid == id && *h == p.config_hash));
        }
        if let Some(r) = tree.get_mut(p.resource) {
            r.profiles
                .retain(|(h, pid)| !(*pid == id && *h == p.config_hash));
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
    use super::{ClassSet, Profile, ProfileMap, ProfileState, ScanConfig, compute_config_hash};
    use crate::ids::ResourceId;
    use crate::output::StepOutput;
    use crate::resource::ResourceRole;
    use crate::scan_config::GlobPattern;
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

    #[test]
    fn new_profile_starts_idle_with_zero_refcounts() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(matches!(p.state, ProfileState::Idle));
        assert!(p.baseline.is_none());
        assert!(p.current.is_none());
        assert!(p.parent_profile.is_none());
        assert_eq!(p.dirty_descendants, 0);
        assert_eq!(p.sub_refcount, 0);
        assert_eq!(p.max_settle, MAX_SETTLE);
        assert_eq!(p.settle, SETTLE);
    }

    /// `fired_subs` defaults to an empty map; engine fills it on
    /// first successful Effect emission.
    #[test]
    fn new_profile_initialises_fired_subs_empty() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(p.fired_subs.is_empty());
    }

    /// `has_per_file_fds` defaults to false when `events` excludes both
    /// CONTENT and METADATA. The flag is invariant for the Profile's
    /// lifetime — set once at construction from the events mask.
    #[test]
    fn new_profile_initialises_has_per_file_fds_false_for_empty_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(!p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (closes
    /// E2E #3 by default for `subtree-root`).
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT);
        assert!(p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::CONTENT);
    }

    /// `has_per_file_fds` is also true when METADATA is in the mask (a
    /// metadata-only watch needs per-file FDs for chmod / nlink signals).
    #[test]
    fn new_profile_has_per_file_fds_when_metadata_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA);
        assert!(p.has_per_file_fds);
    }

    /// STRUCTURE-only watch does not flip `has_per_file_fds` — directory
    /// entries are observed at the parent dir's FD, not at per-file FDs.
    #[test]
    fn new_profile_has_per_file_fds_false_for_structure_only() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE);
        assert!(!p.has_per_file_fds);
    }

    #[test]
    fn config_hash_matches_compute_config_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let c = cfg();
        let expected = compute_config_hash(&c, MAX_SETTLE, NO_EVENTS);
        let p = Profile::new(r, c, MAX_SETTLE, SETTLE, NO_EVENTS);
        assert_eq!(p.config_hash, expected);
    }

    /// Different `events` mask produces different `config_hash`
    /// (partition-by-mask).
    #[test]
    fn config_hash_partitions_by_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p_content = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT);
        let p_meta = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA);
        assert_ne!(p_content.config_hash, p_meta.config_hash);
    }

    #[test]
    fn attach_writes_both_indices() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let h = p.config_hash;
        let pid = profiles.attach(&mut tree, p);

        assert_eq!(profiles.find(r, h), Some(pid));
        assert_eq!(tree.get(r).unwrap().profiles(), &[(h, pid)]);
    }

    #[test]
    fn attach_anchors_resource_against_reap() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        tree.vacate(r, &mut StepOutput::default());
        assert!(!tree.try_reap(r), "Profile-anchored resource must not reap");
    }

    #[test]
    fn detach_clears_back_references() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let h = p.config_hash;
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
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        profiles.detach(&mut tree, pid);
        tree.vacate(r, &mut StepOutput::default());
        assert!(tree.try_reap(r));
        assert!(tree.get(r).is_none());
    }

    #[test]
    fn at_iterates_profiles_attached_at_resource() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);

        let pid_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS),
        );
        // Different max_settle ⇒ different config_hash ⇒ distinct Profile.
        let pid_b = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS),
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
        let r1 = tree.ensure(None, "a", ResourceRole::User);
        let r2 = tree.ensure(None, "b", ResourceRole::User);

        let p1 = profiles.attach(
            &mut tree,
            Profile::new(r1, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p2 = profiles.attach(
            &mut tree,
            Profile::new(r2, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
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
        let r = tree.ensure(None, "x", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        // Caller failed to `find` first; second attach hits debug_assert.
        let _pid2 = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
    }

    // -----------------------------------------------------------------------
    // rebase_baseline / capture_witness_at_loss
    // -----------------------------------------------------------------------

    fn empty_dir_snapshot(resource: ResourceId) -> Arc<DirSnapshot> {
        Arc::new(DirSnapshot::new(
            resource,
            DirMeta {
                mtime: UNIX_EPOCH,
                inode: 0,
                device: 0,
            },
            0,
            BTreeMap::new(),
        ))
    }

    fn empty_leaf_entry() -> LeafEntry {
        LeafEntry::new(EntryKind::File, 0, UNIX_EPOCH, 0, 0)
    }

    #[test]
    fn rebase_baseline_clones_current_into_baseline() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        p.current = Some(TreeSnapshot::Dir(empty_dir_snapshot(r)));
        assert!(p.baseline.is_none());

        p.rebase_baseline();

        assert!(p.baseline.is_some());
        assert_eq!(
            p.baseline.as_ref().unwrap().hash(),
            p.current.as_ref().unwrap().hash(),
            "baseline matches current",
        );
    }

    #[test]
    fn rebase_baseline_clears_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        p.current = Some(TreeSnapshot::Dir(empty_dir_snapshot(r)));
        p.last_settled_hash_at_loss = Some(0xdead_beef);

        p.rebase_baseline();

        assert!(
            p.last_settled_hash_at_loss.is_none(),
            "rebase clears the witness",
        );
    }

    #[test]
    fn capture_witness_at_loss_sets_witness_from_baseline_dir_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let snap = TreeSnapshot::Dir(empty_dir_snapshot(r));
        let expected = snap.hash();
        p.baseline = Some(snap);

        p.capture_witness_at_loss();

        assert_eq!(p.last_settled_hash_at_loss, Some(expected));
    }

    #[test]
    fn capture_witness_at_loss_sets_witness_from_baseline_leaf_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "file", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let snap = TreeSnapshot::File(empty_leaf_entry());
        let expected = snap.hash();
        p.baseline = Some(snap);

        p.capture_witness_at_loss();

        assert_eq!(p.last_settled_hash_at_loss, Some(expected));
    }

    #[test]
    fn capture_witness_at_loss_no_op_when_baseline_none() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        // Pre-populate witness; helper must not overwrite with None.
        p.last_settled_hash_at_loss = Some(0x00c0_ffee);

        p.capture_witness_at_loss();

        assert_eq!(
            p.last_settled_hash_at_loss,
            Some(0x00c0_ffee),
            "no-op preserves prior witness",
        );
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
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let cfg = ScanConfig::builder()
            .exclude(glob("z"))
            .exclude(glob("a"))
            .exclude(glob("m"))
            .build();

        let p = Profile::new(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS);

        let actual: Vec<&str> = p
            .exclude_strings
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
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(p.exclude_strings.is_empty());
    }

    /// The Arc on `Profile.exclude_strings` is the substitution-side handle
    /// shared across every Sub joined to this Profile. Two clones of the
    /// field point at the same allocation; the `bytes-per-Arc` cost is
    /// constant regardless of Sub fanout.
    #[test]
    fn profile_exclude_strings_arc_shared_across_siblings() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let cfg = ScanConfig::builder()
            .exclude(glob("*.tmp"))
            .exclude(glob("*.bak"))
            .build();

        let p = Profile::new(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS);

        let initial = Arc::strong_count(&p.exclude_strings);
        let sibling_a = Arc::clone(&p.exclude_strings);
        let sibling_b = Arc::clone(&p.exclude_strings);

        assert!(
            Arc::ptr_eq(&sibling_a, &sibling_b),
            "siblings reading exclude_strings observe one allocation",
        );
        assert_eq!(
            Arc::strong_count(&p.exclude_strings),
            initial + 2,
            "each sibling clone bumps the strong count",
        );
    }
}
