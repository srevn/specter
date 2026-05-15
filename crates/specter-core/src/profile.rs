//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is computed at construction from
//! `(config, max_settle)` and is the lifetime-stable identity of the Profile.
//! `ProfileMap` keeps `(resource, config_hash) → ProfileId` and updates
//! `Resource.profiles` in lockstep — `attach`/`detach` are the only mutators
//! of either index.

use crate::effect::DedupKey;
use crate::ids::{ProfileId, ResourceId, TimerId};
use crate::resource::ResourceKind;
use crate::scan_config::{ScanConfig, compute_config_hash};
use crate::snapshot::tree::TreeSnapshot;
use crate::sub::ClassSet;
use crate::tree::Tree;
use compact_str::CompactString;
use slotmap::{SecondaryMap, SlotMap};
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One fire cycle, split by the fire-transition boundary.
///
/// A burst lives `Idle → Active(ActiveBurst) → Idle`. The fire transition
/// (`Verifying → Awaiting`) is a typed state-machine move from
/// [`PreFireBurst`] to [`PostFireBurst`]: the two sides have disjoint
/// valid mutators, valid timers, valid probe responses, and accumulator
/// semantics. Encoding the split at the type level means a field that
/// has no post-fire consumer (e.g. `forced`, `burst_deadline`,
/// `dirty_resources`) cannot leak across the boundary by construction.
///
/// **Pre-fire** (`Batching | Verifying | Draining`): event-driven
/// debounce window, in-flight verify or self-stable / descendants-pending
/// idle. Carries the event-accumulators (`dirty_resources`,
/// `force_walk_resources`, `suppressed_resources`) and the source of
/// truth for the settle deadline (`last_event_time`).
///
/// **Post-fire** (`Awaiting | Rebasing`): effect emitted, gate-timer
/// armed, then post-fire probe re-establishing the baseline. The
/// pre-fire accumulators are *gone* — they were consumed at the
/// `transition_to_verifying` immediately preceding the fire — and the
/// `BurstDeadline` timer becomes structurally irrelevant
/// ([`PostFireBurst::timer_token`] folds it to `None` for post-fire
/// phases, so the engine's stale-drain lazily collects the heap
/// entry). The only fresh accumulator is `force_walk_resources`,
/// which the post-fire absorb arm of `drive_burst` (now:
/// `absorb_event_into_fire_tail`) feeds for the rebase probe's
/// force-walk hint.
#[derive(Debug)]
pub enum ActiveBurst {
    PreFire(PreFireBurst),
    PostFire(PostFireBurst),
}

/// Pre-fire lifecycle — every phase before the fire transition.
///
/// Fields are split across two roles:
/// - **Burst-scoped invariants** (`intent`, `forced`, `burst_deadline`,
///   `probe_target`): survive every pre-fire phase transition.
/// - **Pre-fire accumulators** (`dirty_resources`,
///   `force_walk_resources`, `suppressed_resources`,
///   `last_event_time`): populated by `event_drives_batching`, consumed
///   at the next `transition_to_verifying`.
///
/// `force_walk_resources` carries the events the next probe must visit
/// fresh (defeats the walker's coarse-mtime skip on per-event-dirty
/// paths). Single accumulator across `Batching | Verifying | Draining` —
/// `transition_to_verifying` consumes and clears.
///
/// `dirty_resources` is preserved across the burst's pre-fire lifetime
/// because the LCA target is recomputed from it at every reconfirm
/// (`Draining → Verifying`) — the *target* mutates, the *basis* doesn't.
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
/// `None` only at fresh Seed start (no first event); `event_drives_batching`
/// repopulates on any subsequent FsEvent.
#[derive(Debug)]
pub struct PreFireBurst {
    pub burst_deadline: TimerId,
    pub phase: PreFirePhase,
    pub intent: BurstIntent,
    pub forced: bool,
    /// Resources whose `FsEvent` drove (or is driving) this burst.
    /// Populated by `start_standard_burst` (`{ event_resource }` seed)
    /// and `event_drives_batching` (each FsEvent during the pre-fire
    /// phases — `Batching | Verifying | Draining`). Used to compute the
    /// LCA target at every `transition_to_verifying`.
    pub dirty_resources: BTreeSet<ResourceId>,
    /// Resources whose snapshots the next probe must visit fresh,
    /// defeating the walker's coarse-mtime skip. Seeded by
    /// `start_standard_burst` and `event_drives_batching`;
    /// `transition_to_verifying` consumes and clears via `mem::take`.
    pub force_walk_resources: BTreeSet<ResourceId>,
    /// Latest probe target. Initialised to the Profile's anchor at
    /// burst start. Overwritten by `transition_to_verifying` to the
    /// `pre_fire_target` result (File anchor → anchor; Seed → anchor;
    /// Standard → LCA of `dirty_resources`). `transition_to_rebasing`
    /// targets the anchor unconditionally but does not write this
    /// field (the post-fire phases live on `PostFireBurst`, which has
    /// no `probe_target` — Rebasing's target is structurally fixed).
    ///
    /// **Draining → Verifying reconfirm.** Recomputed via the same
    /// `pre_fire_target` rule because `dirty_resources` is preserved
    /// across the burst's pre-fire lifetime: production code mutates
    /// it only by `insert`, so the LCA basis is identical at the
    /// reconfirm to what it was at the original Verifying entry.
    /// Slot reaping during Draining only narrows the result —
    /// `lca_target` filters reaped slots and falls back to anchor on
    /// an empty live set.
    pub probe_target: ResourceId,
    /// Non-anchor resources whose `suppress_count` was bumped 0→1 by
    /// `event_drives_batching` during this burst's pre-fire phases.
    /// Taken (via `mem::take`) at `transition_to_verifying` to drive
    /// the symmetric `sub_suppress` drain, and defensively at
    /// `finish_burst_to_idle` for abnormal-end paths
    /// (`finalize_anchor_lost`, reap mid-burst).
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
    /// `BTreeSet` (not `Vec`) so iteration order is deterministic — the
    /// `sub_suppress` drain emits `Unsuppress` ops in `ResourceId`
    /// ascending order, matching `StepOutput.watch_ops`'s sort
    /// discipline.
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
    ///   `Active(PreFire(Verifying))` directly, with no Batching phase
    ///   at start. If a fresh `FsEvent` later arrives during the Seed
    ///   verify (`event_drives_batching` from the `Verifying → Batching`
    ///   arm), the field is repopulated.
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
}

/// Pre-fire phase discriminator.
///
/// `Batching` carries its own correlation token (`settle_timer: TimerId`)
/// because timer correlation is per-Burst and has no peer slot to live on.
/// `Verifying` is unit: the probe correlation lives on the engine's
/// `ProbeChannel` keyed by `ProbeOwner::Profile(_)` with
/// `OpenKind::ProfileVerifying`, so the burst phase only encodes
/// "probe in flight" as state-machine identity. `Draining` is
/// correlated externally by `Profile.dirty_descendants` and carries no
/// token of its own.
#[derive(Debug)]
pub enum PreFirePhase {
    /// Activity-gap detection. `settle_timer` is the armed debounce
    /// timer; an `FsEvent` reschedules it (`event_drives_batching`),
    /// timer expiry advances to `Verifying` (`transition_to_verifying`).
    Batching { settle_timer: TimerId },
    /// Probe in flight. The matching `ProbeCorrelation` lives on the
    /// engine's `ProbeChannel` (keyed by `ProbeOwner::Profile(_)` with
    /// `OpenKind::ProfileVerifying`); this variant is unit because
    /// the channel is the single source of truth (encoding the
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
}

/// Post-fire lifecycle — Awaiting effect completion or Rebasing.
///
/// **Three fields, by construction.**
/// - No `forced`: no fire decision left (the fire already happened).
/// - No `burst_deadline`: the BurstDeadline timer is filtered out by
///   [`PostFireBurst::timer_token`]'s `Settle | BurstDeadline` arm for
///   post-fire phases. Stored on the pre-fire side; lazy-dropped when
///   it expires post-fire.
/// - No `probe_target`: Rebasing always targets the Profile's anchor.
/// - No `dirty_resources` / `suppressed_resources` / `last_event_time`:
///   pre-fire accumulators, drained at `transition_to_verifying`
///   entry (the only path to fire).
///
/// `intent: BurstIntent` survives because `dispatch_rebase_*` reads it
/// for the `ProbeVanished` / `ProbeFailed` diagnostic — Seed-driven
/// drift rebases and Standard-driven post-fire rebases both reach
/// PostFire, and the diagnostic distinguishes them.
///
/// `force_walk_resources` is a fresh accumulator distinct from the
/// pre-fire one. Populated by `absorb_event_into_fire_tail` (FsEvents
/// arriving during the post-fire tail) and consumed at
/// `transition_to_rebasing` for the rebase probe's force-walk hint —
/// closes the POSIX content-edit hole where a descendant content
/// change during fire-tail doesn't bump the anchor's mtime.
#[derive(Debug)]
pub struct PostFireBurst {
    pub intent: BurstIntent,
    pub phase: PostFirePhase,
    /// Events absorbed during the post-fire tail
    /// (`Awaiting | Rebasing`). Seeded by `absorb_event_into_fire_tail`
    /// in `drive_burst`'s post-fire arm; consumed at
    /// `transition_to_rebasing`. Events absorbed during `Rebasing`
    /// after the rebase probe is in flight have no consumer — they
    /// accumulate into the cleared field and drop at
    /// `finish_burst_to_idle`. The bounded residual window (≈ probe
    /// round-trip latency) is the v1 carve-out.
    pub force_walk_resources: BTreeSet<ResourceId>,
}

/// Post-fire phase discriminator.
///
/// `Awaiting { outstanding, gate_deadline }`: effects emitted, counter
/// decrements on each `EffectComplete` for this Profile's `DedupKey`s.
/// Reaching zero advances to `Rebasing` (or, when the burst carries
/// [`BurstFinish::Reap`], finishes the burst directly). `gate_deadline`
/// is the recovery timer for an actuator that never reports completion
/// — its expiry forces the burst into `Rebasing` (or, on a zombie
/// burst, directly into [`crate::ProfileState::Idle`] via reap).
///
/// `Rebasing`: post-fire probe in flight at the anchor. Correlation
/// lives on the engine's `ProbeChannel` (keyed by
/// `ProbeOwner::Profile(_)` with `OpenKind::ProfileRebasing`; Verifying
/// and Rebasing are time-disjoint within one burst so the same channel
/// key is reused). `dispatch_rebase_ok` then sets `baseline := current`
/// and finishes the burst to Idle.
#[derive(Debug)]
pub enum PostFirePhase {
    Awaiting {
        outstanding: u32,
        gate_deadline: TimerId,
    },
    Rebasing,
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
    /// - [`TimerKind::AwaitGateDeadline`] — type-impossible here (the
    ///   field lives on [`PostFireBurst`] only); the arm returns
    ///   `None` to encode the structural absence.
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
                PreFirePhase::Verifying | PreFirePhase::Draining => None,
            },
            TimerKind::BurstDeadline => Some(self.burst_deadline),
            TimerKind::AwaitGateDeadline => None,
        }
    }

    /// Typed move from pre-fire to post-fire.
    ///
    /// Drops, by leaving them out of the constructor:
    /// - `burst_deadline` — lazy-dropped by
    ///   [`PostFireBurst::timer_token`]'s `None` arm once it expires
    ///   post-fire.
    /// - `forced` — no fire decision left in the post-fire lifecycle.
    /// - `probe_target` — Rebasing always targets the anchor.
    /// - `last_event_time` — pre-fire-only accumulator.
    /// - `dirty_resources` — pre-fire-only accumulator.
    /// - `force_walk_resources` — pre-fire-only accumulator. Drained
    ///   to empty at the Verifying-entry that immediately precedes the
    ///   fire (`transition_to_verifying`'s `mem::take`); the
    ///   debug_assert below catches a future regression that omits the
    ///   drain.
    /// - `suppressed_resources` — likewise drained at
    ///   `transition_to_verifying`; debug_asserted here as
    ///   defense-in-depth.
    ///
    /// `intent` is preserved (read by `dispatch_rebase_*` for the
    /// diagnostic).
    #[must_use]
    pub fn into_post_fire(self, outstanding: u32, gate_deadline: TimerId) -> PostFireBurst {
        debug_assert!(
            self.force_walk_resources.is_empty(),
            "PreFireBurst::into_post_fire: force_walk_resources not drained \
             at Verifying entry — drain must happen at transition_to_verifying",
        );
        debug_assert!(
            self.suppressed_resources.is_empty(),
            "PreFireBurst::into_post_fire: suppressed_resources not drained \
             at Verifying entry — drain must happen at transition_to_verifying",
        );
        PostFireBurst {
            intent: self.intent,
            phase: PostFirePhase::Awaiting {
                outstanding,
                gate_deadline,
            },
            force_walk_resources: BTreeSet::new(),
        }
    }
}

impl PostFireBurst {
    /// The `TimerId` armed on this burst for `kind`, or `None` if the
    /// post-fire shape doesn't carry a slot for `kind`.
    ///
    /// Post-fire owns:
    /// - [`TimerKind::AwaitGateDeadline`] — lives on
    ///   [`PostFirePhase::Awaiting`]'s `gate_deadline` field; the arm
    ///   returns `None` once the burst advances to `Rebasing` (the
    ///   field doesn't exist on that variant).
    /// - [`TimerKind::Settle`] / [`TimerKind::BurstDeadline`] —
    ///   type-impossible here (the fields were dropped at
    ///   [`PreFireBurst::into_post_fire`]); the arm returns `None`
    ///   to encode the structural absence.
    #[must_use]
    pub const fn timer_token(&self, kind: TimerKind) -> Option<TimerId> {
        match kind {
            TimerKind::AwaitGateDeadline => match &self.phase {
                PostFirePhase::Awaiting { gate_deadline, .. } => Some(*gate_deadline),
                PostFirePhase::Rebasing => None,
            },
            TimerKind::Settle | TimerKind::BurstDeadline => None,
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
}

/// Burst-finish directive — *what does the Profile do at burst-end?*
///
/// Carried as the second payload of [`ProfileState::Active`]. Default
/// [`Self::ReturnToIdle`]: the burst completes, the Profile returns to
/// [`ProfileState::Idle`], and the next `FsEvent` may start a fresh
/// burst. [`Self::Reap`] flips the directive: the active burst still
/// runs to completion (so the `propagate(-1) / sub_suppress` drain
/// ordering is preserved), but `finish_burst_to_idle` then routes
/// through `reap_profile` instead of returning the Profile to Idle.
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
///   — a burst is in flight whose `propagate(-1) / sub_suppress` drain
///   ordering must run before reap. The caller flips
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
/// I5 (at most one outstanding probe per Profile) is enforced
/// **structurally** by the engine's `ProbeChannel`: at most one map
/// entry per `ProbeOwner`. Open via `ProbeChannel::open` (panics on
/// double-open); close via `close_if` (correlation-matched) or
/// `close` (unconditional). The dispatch site routes on the channel's
/// `OpenKind` discriminant rather than per-state inspection (see
/// `Engine::on_probe_response` in `specter-engine`).
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

    /// True iff the state is `Active(PreFire(Draining))`. The
    /// reconfirm cascade (the `Draining → Verifying` re-probe driven
    /// when `dirty_descendants → 0`) keys off this predicate — only
    /// Draining ancestors care about the descendants-cleared edge.
    /// `Idle` and `Pending` are structurally not-Draining; the
    /// post-fire arm and the other pre-fire phases (Batching,
    /// Verifying) also return `false`.
    #[must_use]
    pub const fn is_draining(&self) -> bool {
        match self {
            Self::Active(ActiveBurst::PreFire(pre), _) => {
                matches!(pre.phase, PreFirePhase::Draining)
            }
            Self::Idle | Self::Pending(_) | Self::Active(ActiveBurst::PostFire(_), _) => false,
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
/// I5 ("at most one outstanding probe per Profile") for the Pending
/// lifecycle is enforced structurally by the engine's `ProbeChannel`
/// (keyed by `ProbeOwner::Profile(_)`; descent probes carry
/// `OpenKind::ProfileDescent`). The descent's variant payload holds no
/// probe-correlation data of its
/// own.
#[derive(Clone, Debug)]
pub struct DescentState {
    /// Deepest existing ancestor currently Watched. The Profile
    /// contributes `+1` to this Resource's `watch_demand`.
    pub(crate) current_prefix: ResourceId,
    /// Path components from `current_prefix` (exclusive) down to the
    /// anchor (inclusive). Non-empty by type construction;
    /// single-component segments (no `/`).
    pub(crate) remaining_components: DescentRemaining,
}

impl DescentState {
    /// Construct a fresh descent payload. Sole producer pattern used
    /// by `materialize_path_or_pending` (Profile pending arm), the
    /// Promoter attach path's pending arm, and the recovery / rewind
    /// flows in `engine::descent` that re-enter `Pending` after an
    /// anchor-terminal event.
    ///
    /// Field-private; callers route through this constructor so the
    /// invariants on `current_prefix` (Watched, refcounted) and
    /// `remaining_components` (non-empty by [`DescentRemaining`]'s
    /// own constructor) are pinned at a single boundary. The engine's
    /// refcount setup runs around this constructor (the contribution
    /// at `current_prefix` is installed by `add_watch` separately) —
    /// the struct itself only carries the bookkeeping needed to
    /// dispatch the next descent step.
    #[must_use]
    pub const fn new(current_prefix: ResourceId, remaining_components: DescentRemaining) -> Self {
        Self {
            current_prefix,
            remaining_components,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DescentRemaining {
    inner: Vec<CompactString>,
}

impl DescentRemaining {
    /// Construct from a `Vec`. Returns `None` iff `v` is empty,
    /// preserving the non-empty invariant. Sole intended producer is
    /// `materialize_path_or_pending`'s Pending branch, where the
    /// `prefix_idx + 1 < components.len()` gate already guarantees
    /// non-empty; the `None` arm is defense-in-depth against future
    /// callers.
    #[must_use]
    pub fn from_vec(v: Vec<CompactString>) -> Option<Self> {
        if v.is_empty() {
            None
        } else {
            Some(Self { inner: v })
        }
    }

    /// First (next-to-consume) segment. Always present by invariant.
    #[must_use]
    pub fn head(&self) -> &CompactString {
        // Indexing rather than `first().unwrap()` to encode the invariant
        // at the access site — a future maintainer can't accidentally
        // weaken `head` to a defensive `Option` without also adjusting
        // the type's construction discipline.
        &self.inner[0]
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
    /// materialization on this edge and never calls [`advance`].
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.inner.len() == 1
    }

    /// Consume the head, shifting the tail forward by one. Preserves
    /// the non-empty invariant by debug-asserting non-terminal at entry;
    /// release builds clamp on terminal (no-op) rather than violating
    /// the invariant.
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
            self.inner.remove(0);
        }
    }

    /// Rewind by one segment: insert `segment` as the new head. Used by
    /// `dispatch_descent_vanished`'s rewind branch, where a `Vanished`
    /// response on `current_prefix` shifts the descent up one level and
    /// the vanished prefix's own segment becomes the next-to-consume
    /// component on the way back down.
    pub fn prepend(&mut self, segment: CompactString) {
        self.inner.insert(0, segment);
    }

    /// Borrow the components for read-only iteration (test assertions,
    /// diagnostics). Production code uses [`head`](Self::head) /
    /// [`len`](Self::len) / [`is_terminal`](Self::is_terminal) — direct
    /// slice access is for fixture / assertion use only.
    #[must_use]
    pub fn as_slice(&self) -> &[CompactString] {
        &self.inner
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
    /// is reserved for testkit / unit-test setup.
    pub parent_profile: Option<ProfileId>,
    pub dirty_descendants: u32,
    pub max_settle: Duration,
    /// Settle interval driving `start_standard_burst` and the backoff base.
    /// Cached on construction from the first attached Sub; the engine
    /// recomputes this as `min(remaining_subs.settles)` on `attach_sub`
    /// (existing Profile) and `detach_sub`.
    pub settle: Duration,
    /// Cached parent Resource that this Profile contributes a watch to.
    /// `attach_sub` sets it; `detach_sub` releases the contribution via the
    /// cached id without re-deriving the parent. `None` if the anchor is
    /// itself a root (no parent in the Tree) — root rename detection is then
    /// unavailable.
    pub watch_root_parent: Option<ResourceId>,
    /// Tracks whether this Profile currently holds the anchor
    /// contribution at `resource` — [`AnchorClaim::Held`] on the path
    /// that called `add_watch(anchor, ContribKey::ProfileAnchor(pid), ...)`
    /// (immediate-Seed in `attach_sub_inner` or descent's anchor
    /// materialization), cleared to [`AnchorClaim::None`] on the
    /// matching `sub_watch(anchor, ContribKey::ProfileAnchor(pid))`
    /// (anchor terminal event, reap, clamp purge).
    ///
    /// The claim distinguishes three reap-time lifecycle states that
    /// otherwise look identical in the Profile/descent registry:
    /// **materialized** (`Held` ⇒ release anchor), **pending**
    /// (descent in flight ⇒ release descent prefix instead), and
    /// **purged** (`None`, descent already removed by
    /// `Input::WatchOpRejected` ⇒ no contribution to release; the clamp
    /// already cleared the contributions map).
    ///
    /// Without this field a heuristic like `baseline.is_some() ||
    /// current.is_some()` undercounts `dispatch_seed_vanished` paths
    /// (which clear the snapshots while leaving the anchor's
    /// contribution intact) and a heuristic like
    /// `tree.get(anchor).is_watched()` overcounts in multi-Profile
    /// sharing (would steal another Profile's contribution).
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
    /// The walker-side reconciler reads this to decide whether
    /// covered Leaf children get an
    /// [`crate::ContribKey::ProfileDescendant`] contribution
    /// installed via `add_watch` — per-file FDs for in-place edit
    /// detection.
    pub has_per_file_fds: bool,
}

impl Profile {
    /// Construct a fresh Profile: state `Idle` (so no burst-finish
    /// directive exists yet), no baseline/current, no watch-root parent
    /// recorded. `config_hash` is computed from
    /// `(config, max_settle, events)` and is stable for the Profile's
    /// lifetime — there is no path to a Profile with an unset or stale
    /// hash.
    ///
    /// `events` becomes the Profile's `events_union` and drives
    /// `has_per_file_fds` (true iff CONTENT or METADATA is in the mask).
    /// Every Sub on a Profile shares the same `events`, so
    /// `events_union` is invariant for the Profile's lifetime.
    ///
    /// `exclude_strings` is projected once here from `config.exclude` —
    /// the [`ScanConfig`] builder has already sorted the vector by source,
    /// so the projection is canonical without re-sorting.
    ///
    /// `kind` is the anchor's classified shape at construction: `None`
    /// for a `DescentScaffold` or freshly-`ensure`d-but-unprobed slot
    /// (descent materialisation classifies it via
    /// [`Self::materialize_anchor`]; the first Seed-Ok via
    /// [`Self::install_dir_current`] / [`Self::install_file_current`]).
    #[must_use]
    pub fn new(
        resource: ResourceId,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        events: ClassSet,
        kind: Option<ResourceKind>,
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
            kind,
            state: ProfileState::Idle,
            baseline: None,
            current: None,
            parent_profile: None,
            dirty_descendants: 0,
            max_settle,
            settle,
            watch_root_parent: None,
            anchor_claim: AnchorClaim::None,
            fired_subs: BTreeSet::new(),
            last_settled_hash_at_loss: None,
            events_union: events,
            has_per_file_fds,
        }
    }

    /// Install a Dir-shaped `current`, atomically setting `kind =
    /// Some(Dir)` alongside. Sole legitimate writer of
    /// `(kind, current)` on the Dir arm — `grep -rnE 'p\.current
    /// = '` on `crates/` should turn up only `Profile::*`
    /// internals and this helper's call sites.
    ///
    /// Atomic write here means: callers that observe `current` as
    /// `Some(Dir)` are guaranteed to observe `kind` as `Some(Dir)` in
    /// the same step. The setter encodes the [`Profile::kind`]
    /// rustdoc's snapshot-shape invariant —
    /// `current = Some(TreeSnapshot::Dir(_)) ⇒ kind == Some(Dir)` —
    /// at the write API, not just in prose.
    ///
    /// **Precondition.** `kind.is_none() || kind == Some(Dir)`. A
    /// `File`-kinded Profile receiving a Dir install is a walker /
    /// dispatcher routing breach; the engine's dispatcher boundary
    /// (`Engine::kind_agrees_or_finalize`) catches this case and
    /// routes through `finalize_anchor_lost`, so the setter's
    /// `debug_assert!` is a defensive backstop against a future caller
    /// bypassing the boundary.
    ///
    /// **Baseline shape.** Production paths preserve `baseline` /
    /// `current` shape agreement automatically: `rebase_baseline`
    /// clones `current` into `baseline` (shape matches by construction);
    /// `Engine::discard_anchor_state` clears both atomically. The
    /// `debug_assert!` on baseline shape catches any direct-write
    /// regression in tests; production paths never trigger it.
    pub fn install_dir_current(&mut self, snapshot: Arc<crate::snapshot::tree::DirSnapshot>) {
        debug_assert!(
            self.kind.is_none() || self.kind == Some(ResourceKind::Dir),
            "install_dir_current: kind mismatch (existing = {:?})",
            self.kind,
        );
        debug_assert!(
            self.baseline
                .as_ref()
                .is_none_or(|b| matches!(b, TreeSnapshot::Dir(_))),
            "install_dir_current: baseline shape disagrees with new current (Dir)",
        );
        self.kind = Some(ResourceKind::Dir);
        self.current = Some(TreeSnapshot::Dir(snapshot));
    }

    /// Install a File-shaped `current`, atomically setting `kind =
    /// Some(File)` alongside. Sole legitimate writer of `(kind,
    /// current)` on the File arm — symmetric with
    /// [`Self::install_dir_current`].
    ///
    /// **Precondition.** `kind.is_none() || kind == Some(File)`. A
    /// `Dir`-kinded Profile receiving a File install is the symmetric
    /// dispatcher routing breach; `Engine::kind_agrees_or_finalize`
    /// catches it and routes through `finalize_anchor_lost`.
    pub fn install_file_current(&mut self, leaf: crate::snapshot::tree::LeafEntry) {
        debug_assert!(
            self.kind.is_none() || self.kind == Some(ResourceKind::File),
            "install_file_current: kind mismatch (existing = {:?})",
            self.kind,
        );
        debug_assert!(
            self.baseline
                .as_ref()
                .is_none_or(|b| matches!(b, TreeSnapshot::File(_))),
            "install_file_current: baseline shape disagrees with new current (File)",
        );
        self.kind = Some(ResourceKind::File);
        self.current = Some(TreeSnapshot::File(leaf));
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

    /// Atomically clear the anchor classification triple
    /// `(kind, baseline, current)`. Captures the survival witness from
    /// `baseline` *first*, so the cross-field invariant
    /// `baseline.is_some() ⇒ last_settled_hash_at_loss.is_none()` holds
    /// at every step boundary. `current` is already `None` at every
    /// production caller (`discard_anchor_state` runs
    /// [`Self::take_current`] first); the write here backstops a future
    /// caller that skips the take. The witness is *not* cleared — it is
    /// consumed later by [`Self::rebase_baseline`]. Inverse of
    /// [`Self::materialize_anchor`].
    pub fn clear_anchor_classification(&mut self) {
        self.capture_witness_at_loss();
        self.baseline = None;
        self.current = None;
        self.kind = None;
    }

    /// Atomically install a descent-materialised anchor: transition
    /// `Pending → Idle`, install the claim, pin the discovered kind. Sole
    /// caller `Engine::materialize_profile_anchor`, which launches the
    /// Seed burst on the next statement — the `Idle` written here is a
    /// structural intermediate, never observed. Inverse of
    /// [`Self::clear_anchor_classification`].
    ///
    /// Debug-asserts the fresh-materialisation preconditions
    /// (`state == Pending`, no claim, unprobed kind); release builds
    /// compile the asserts out and still write the triple atomically (a
    /// breach is the same shape as an `install_*_current` kind mismatch).
    pub fn materialize_anchor(&mut self, kind: ResourceKind) {
        debug_assert!(
            matches!(self.state, ProfileState::Pending(_)),
            "materialize_anchor: state must be Pending (was {:?})",
            self.state.discriminant(),
        );
        debug_assert!(
            matches!(self.anchor_claim, AnchorClaim::None),
            "materialize_anchor: anchor_claim must be None",
        );
        debug_assert!(
            self.kind.is_none(),
            "materialize_anchor: kind must be unprobed",
        );
        self.state = ProfileState::Idle;
        self.anchor_claim = AnchorClaim::Held;
        self.kind = Some(kind);
    }

    /// Sole legitimate post-construction writer of `state`. Returns the
    /// prior state via `mem::replace` so the typed-move callers
    /// (`transition_to_awaiting`, `finish_burst_to_idle`) can consume the
    /// prior burst by value for [`PreFireBurst::into_post_fire`] without
    /// holding a `&mut state` borrow across the move. Shape-agnostic:
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
        self.anchor_claim = AnchorClaim::Held;
    }

    /// Release the anchor claim. Idempotent against `None`. Production
    /// caller: `Engine::release_anchor_claim`, which wraps this with the
    /// Tree-side `sub_watch`.
    pub const fn release_anchor_claim_now(&mut self) {
        self.anchor_claim = AnchorClaim::None;
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
        self.anchor_claim
    }

    #[must_use]
    pub const fn kind(&self) -> Option<ResourceKind> {
        self.kind
    }

    #[must_use]
    pub const fn baseline(&self) -> Option<&TreeSnapshot> {
        self.baseline.as_ref()
    }

    #[must_use]
    pub const fn current(&self) -> Option<&TreeSnapshot> {
        self.current.as_ref()
    }

    #[must_use]
    pub const fn last_settled_hash_at_loss(&self) -> Option<u128> {
        self.last_settled_hash_at_loss
    }

    /// Mutable descent payload — thin delegator to
    /// [`ProfileState::descent_state_mut`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        self.state.descent_state_mut()
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

    /// Take `current`, leaving `None` — the descendant-claim-release
    /// primitive. Sole caller `Engine::release_descendant_claim`;
    /// idempotent (a second call finds `None`). Not subsumed by
    /// [`Self::clear_anchor_classification`]: it runs first and is also
    /// called standalone from the `dispatch_*_vanished/failed` +
    /// `reap_profile` sites.
    pub const fn take_current(&mut self) -> Option<TreeSnapshot> {
        self.current.take()
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
    use crate::fs_id::FsIdentity;
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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(matches!(p.state, ProfileState::Idle));
        assert!(p.baseline.is_none());
        assert!(p.current.is_none());
        assert!(p.parent_profile.is_none());
        assert_eq!(p.dirty_descendants, 0);
        assert_eq!(p.max_settle, MAX_SETTLE);
        assert_eq!(p.settle, SETTLE);
    }

    /// `fired_subs` defaults to an empty map; engine fills it on
    /// first successful Effect emission.
    #[test]
    fn new_profile_initialises_fired_subs_empty() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(p.fired_subs.is_empty());
    }

    /// `has_per_file_fds` defaults to false when `events` excludes both
    /// CONTENT and METADATA. The flag is invariant for the Profile's
    /// lifetime — set once at construction from the events mask.
    #[test]
    fn new_profile_initialises_has_per_file_fds_false_for_empty_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (closes
    /// E2E #3 by default for `subtree-root`).
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        assert!(p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::CONTENT);
    }

    /// `has_per_file_fds` is also true when METADATA is in the mask (a
    /// metadata-only watch needs per-file FDs for chmod / nlink signals).
    #[test]
    fn new_profile_has_per_file_fds_when_metadata_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert!(p.has_per_file_fds);
    }

    /// STRUCTURE-only watch does not flip `has_per_file_fds` — directory
    /// entries are observed at the parent dir's FD, not at per-file FDs.
    #[test]
    fn new_profile_has_per_file_fds_false_for_structure_only() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE, None);
        assert!(!p.has_per_file_fds);
    }

    #[test]
    fn config_hash_matches_compute_config_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let c = cfg();
        let expected = compute_config_hash(&c, MAX_SETTLE, NO_EVENTS);
        let p = Profile::new(r, c, MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(p.config_hash, expected);
    }

    /// Different `events` mask produces different `config_hash`
    /// (partition-by-mask).
    #[test]
    fn config_hash_partitions_by_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p_content = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        let p_meta = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA, None);
        assert_ne!(p_content.config_hash, p_meta.config_hash);
    }

    #[test]
    fn attach_writes_both_indices() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let h = p.config_hash;
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
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
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
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
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
            Profile::new(r, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS, None),
        );
        // Different max_settle ⇒ different config_hash ⇒ distinct Profile.
        let pid_b = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS, None),
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
            Profile::new(r1, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        let p2 = profiles.attach(
            &mut tree,
            Profile::new(r2, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
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
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
        // Caller failed to `find` first; second attach hits debug_assert.
        let _pid2 = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None),
        );
    }

    // -----------------------------------------------------------------------
    // rebase_baseline / capture_witness_at_loss
    // -----------------------------------------------------------------------

    fn empty_dir_snapshot() -> Arc<DirSnapshot> {
        Arc::new(DirSnapshot::new(
            DirMeta {
                mtime: UNIX_EPOCH,
                fs_id: FsIdentity {
                    inode: 0,
                    device: 0,
                },
            },
            0,
            BTreeMap::new(),
        ))
    }

    fn empty_leaf_entry() -> LeafEntry {
        LeafEntry::new(
            EntryKind::File,
            0,
            UNIX_EPOCH,
            FsIdentity {
                inode: 0,
                device: 0,
            },
        )
    }

    #[test]
    fn rebase_baseline_clones_current_into_baseline() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.current = Some(TreeSnapshot::Dir(empty_dir_snapshot()));
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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.current = Some(TreeSnapshot::Dir(empty_dir_snapshot()));
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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let snap = TreeSnapshot::Dir(empty_dir_snapshot());
        let expected = snap.hash();
        p.baseline = Some(snap);

        p.capture_witness_at_loss();

        assert_eq!(p.last_settled_hash_at_loss, Some(expected));
    }

    #[test]
    fn capture_witness_at_loss_sets_witness_from_baseline_leaf_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("file", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let snap = TreeSnapshot::File(empty_leaf_entry());
        let expected = snap.hash();
        p.baseline = Some(snap);

        p.capture_witness_at_loss();

        assert_eq!(p.last_settled_hash_at_loss, Some(expected));
    }

    // -----------------------------------------------------------------------
    // install_dir_current / install_file_current — atomic kind+current write
    // -----------------------------------------------------------------------

    #[test]
    fn install_dir_current_sets_kind_and_current_atomically() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(p.kind.is_none(), "fresh Profile has unprobed kind");
        assert!(p.current.is_none());

        p.install_dir_current(empty_dir_snapshot());

        assert_eq!(
            p.kind,
            Some(crate::resource::ResourceKind::Dir),
            "kind set atomically with current",
        );
        assert!(matches!(p.current, Some(TreeSnapshot::Dir(_))));
    }

    #[test]
    fn install_file_current_sets_kind_and_current_atomically() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("file", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.install_file_current(empty_leaf_entry());

        assert_eq!(p.kind, Some(crate::resource::ResourceKind::File));
        assert!(matches!(p.current, Some(TreeSnapshot::File(_))));
    }

    /// Setter is idempotent on `kind`: re-installing a Dir current on a
    /// Dir-kinded Profile keeps `kind = Some(Dir)` and updates `current`.
    #[test]
    fn install_dir_current_idempotent_on_dir_kind() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());

        // Second install with a fresh snapshot.
        p.install_dir_current(empty_dir_snapshot());

        assert_eq!(p.kind, Some(crate::resource::ResourceKind::Dir));
    }

    /// Cross-arm misuse: installing a `Dir` on a `File`-kinded Profile
    /// panics in debug builds. Production paths never reach this branch
    /// — `Engine::kind_agrees_or_finalize` catches the routing breach
    /// at the dispatcher boundary before any caller invokes the setter.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "install_dir_current: kind mismatch")]
    fn install_dir_current_panics_on_file_kinded_profile_in_debug() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_file_current(empty_leaf_entry());
        // Boundary-bypass: a future caller skips
        // `kind_agrees_or_finalize`; the setter's debug_assert fires.
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.install_dir_current(empty_dir_snapshot());
        p.install_file_current(empty_leaf_entry());
    }

    #[test]
    fn capture_witness_at_loss_no_op_when_baseline_none() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let cfg = ScanConfig::builder()
            .exclude(glob("z"))
            .exclude(glob("a"))
            .exclude(glob("m"))
            .build();

        let p = Profile::new(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS, None);

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
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(p.exclude_strings.is_empty());
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

        let p = Profile::new(r, cfg, MAX_SETTLE, SETTLE, NO_EVENTS, None);

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

    // -----------------------------------------------------------------------
    // ProfileState projections: timer_token / is_draining / descent_state
    // -----------------------------------------------------------------------

    use super::{
        ActiveBurst, BurstFinish, BurstIntent, DescentRemaining, DescentState, PostFireBurst,
        PostFirePhase, PreFireBurst, PreFirePhase, TimerKind,
    };
    use crate::ids::TimerId;
    use std::collections::BTreeSet;

    fn tid(n: u64) -> TimerId {
        TimerId::from(n)
    }

    fn batching_burst(settle: TimerId, deadline: TimerId, anchor: ResourceId) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: deadline,
            phase: PreFirePhase::Batching {
                settle_timer: settle,
            },
            intent: BurstIntent::Standard,
            forced: false,
            dirty_resources: BTreeSet::new(),
            force_walk_resources: BTreeSet::new(),
            probe_target: anchor,
            suppressed_resources: BTreeSet::new(),
            last_event_time: None,
        }
    }

    fn unit_pre(phase: PreFirePhase, deadline: TimerId, anchor: ResourceId) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: deadline,
            phase,
            intent: BurstIntent::Standard,
            forced: false,
            dirty_resources: BTreeSet::new(),
            force_walk_resources: BTreeSet::new(),
            probe_target: anchor,
            suppressed_resources: BTreeSet::new(),
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
            PreFirePhase::Verifying,
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
        for phase in [PreFirePhase::Verifying, PreFirePhase::Draining] {
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
        let post = PostFireBurst {
            intent: BurstIntent::Standard,
            phase: PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(55),
            },
            force_walk_resources: BTreeSet::new(),
        };
        assert_eq!(
            post.timer_token(TimerKind::AwaitGateDeadline),
            Some(tid(55)),
        );
    }

    /// AwaitGateDeadline on Rebasing returns None — the field doesn't
    /// exist on that variant.
    #[test]
    fn timer_token_await_gate_is_none_on_rebasing() {
        let post = PostFireBurst {
            intent: BurstIntent::Standard,
            phase: PostFirePhase::Rebasing,
            force_walk_resources: BTreeSet::new(),
        };
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
            PostFirePhase::Rebasing,
        ] {
            let post = PostFireBurst {
                intent: BurstIntent::Standard,
                phase,
                force_walk_resources: BTreeSet::new(),
            };
            assert!(post.timer_token(TimerKind::Settle).is_none());
            assert!(post.timer_token(TimerKind::BurstDeadline).is_none());
        }
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

        let post = ActiveBurst::PostFire(PostFireBurst {
            intent: BurstIntent::Standard,
            phase: PostFirePhase::Awaiting {
                outstanding: 1,
                gate_deadline: tid(9),
            },
            force_walk_resources: BTreeSet::new(),
        });
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
        ));
        for k in [
            TimerKind::Settle,
            TimerKind::BurstDeadline,
            TimerKind::AwaitGateDeadline,
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
            )),
            ProfileState::Active(
                ActiveBurst::PreFire(unit_pre(PreFirePhase::Verifying, tid(1), r)),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PreFire(batching_burst(tid(1), tid(2), r)),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    intent: BurstIntent::Standard,
                    phase: PostFirePhase::Awaiting {
                        outstanding: 1,
                        gate_deadline: tid(3),
                    },
                    force_walk_resources: BTreeSet::new(),
                }),
                BurstFinish::ReturnToIdle,
            ),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    intent: BurstIntent::Standard,
                    phase: PostFirePhase::Rebasing,
                    force_walk_resources: BTreeSet::new(),
                }),
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
        ));

        {
            let d = state.descent_state_mut().expect("Pending carries descent");
            d.remaining_components_mut().advance();
        }

        let d = state.descent_state().expect("still Pending");
        assert_eq!(
            d.remaining_components().as_slice(),
            &[CompactString::from("b")]
        );

        // Mutator returns None on non-Pending states.
        let mut idle = ProfileState::Idle;
        assert!(idle.descent_state_mut().is_none());
    }

    // -----------------------------------------------------------------------
    // State-machine setter / accessor API (clear_anchor_classification,
    // materialize_anchor, transition_state, anchor_claim setters,
    // burst projections, read accessors, delegators, take_current)
    // -----------------------------------------------------------------------

    use super::AnchorClaim;
    use crate::resource::ResourceKind;

    fn pending(r: ResourceId) -> ProfileState {
        ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("seg")]).expect("non-empty"),
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
            ActiveBurst::PostFire(PostFireBurst {
                intent: BurstIntent::Standard,
                phase: PostFirePhase::Rebasing,
                force_walk_resources: BTreeSet::new(),
            }),
            BurstFinish::ReturnToIdle,
        )
    }

    #[test]
    fn profile_new_threads_kind_param() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let classified = Profile::new(
            r,
            cfg(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
            Some(ResourceKind::Dir),
        );
        assert_eq!(classified.kind(), Some(ResourceKind::Dir));
        let unprobed = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert_eq!(unprobed.kind(), None);
    }

    #[test]
    fn read_accessors_mirror_fields() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert!(matches!(p.state(), ProfileState::Idle));
        assert_eq!(p.anchor_claim(), AnchorClaim::None);
        assert_eq!(p.kind(), None);
        assert!(p.baseline().is_none());
        assert!(p.current().is_none());
        assert_eq!(p.last_settled_hash_at_loss(), None);

        let snap = TreeSnapshot::Dir(empty_dir_snapshot());
        let h = snap.hash();
        p.baseline = Some(snap.clone());
        p.current = Some(snap);
        p.kind = Some(ResourceKind::Dir);
        p.anchor_claim = AnchorClaim::Held;
        p.last_settled_hash_at_loss = Some(h);

        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        assert_eq!(p.kind(), Some(ResourceKind::Dir));
        assert!(matches!(p.baseline(), Some(TreeSnapshot::Dir(_))));
        assert!(matches!(p.current(), Some(TreeSnapshot::Dir(_))));
        assert_eq!(p.last_settled_hash_at_loss(), Some(h));
    }

    #[test]
    fn transition_state_replaces_and_returns_prior() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        let prior = p.transition_state(pending(r));
        assert!(matches!(prior, ProfileState::Idle));
        assert!(matches!(p.state(), ProfileState::Pending(_)));

        let prior = p.transition_state(ProfileState::Idle);
        assert!(matches!(prior, ProfileState::Pending(_)));
        assert!(matches!(p.state(), ProfileState::Idle));
    }

    #[test]
    fn clear_anchor_classification_clears_triple_and_captures_witness() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        let snap = TreeSnapshot::Dir(empty_dir_snapshot());
        let expected = snap.hash();
        p.kind = Some(ResourceKind::Dir);
        p.baseline = Some(snap.clone());
        p.current = Some(snap);

        p.clear_anchor_classification();

        assert_eq!(p.kind(), None);
        assert!(p.baseline().is_none());
        assert!(p.current().is_none());
        assert_eq!(
            p.last_settled_hash_at_loss(),
            Some(expected),
            "witness captured from baseline before the clear",
        );
    }

    #[test]
    fn clear_anchor_classification_preserves_prior_witness_when_baseline_none() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        // Already-lost shape: no baseline, witness already held.
        p.last_settled_hash_at_loss = Some(0x00c0_ffee);
        p.kind = Some(ResourceKind::Dir);

        p.clear_anchor_classification();

        assert_eq!(
            p.last_settled_hash_at_loss(),
            Some(0x00c0_ffee),
            "capture is a no-op against None baseline — prior witness survives",
        );
        assert_eq!(p.kind(), None);
    }

    #[test]
    fn materialize_anchor_installs_triple_from_pending() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);

        p.materialize_anchor(ResourceKind::Dir);

        assert!(matches!(p.state(), ProfileState::Idle));
        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        assert_eq!(p.kind(), Some(ResourceKind::Dir));
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);
        p.anchor_claim = AnchorClaim::Held;
        p.materialize_anchor(ResourceKind::Dir);
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "kind must be unprobed")]
    fn materialize_anchor_panics_when_kind_probed() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);
        p.kind = Some(ResourceKind::Dir);
        p.materialize_anchor(ResourceKind::Dir);
    }

    #[test]
    fn anchor_claim_setters_are_idempotent() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert!(p.pre_fire_burst_mut().is_none(), "Idle has no pre-fire");
        p.state = pending(r);
        assert!(p.pre_fire_burst_mut().is_none(), "Pending has no pre-fire");
        p.state = active_postfire();
        assert!(p.pre_fire_burst_mut().is_none(), "PostFire has no pre-fire");

        p.state = active_prefire(r);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        assert!(p.post_fire_burst_mut().is_none(), "Idle has no post-fire");
        p.state = active_prefire(r);
        assert!(
            p.post_fire_burst_mut().is_none(),
            "PreFire has no post-fire"
        );

        p.state = active_postfire();
        let post = p
            .post_fire_burst_mut()
            .expect("PostFire carries the payload");
        post.force_walk_resources.insert(r);
        assert!(
            p.post_fire_burst_mut()
                .expect("still PostFire")
                .force_walk_resources
                .contains(&r),
            "mutation through the projection persists",
        );
    }

    #[test]
    fn delegators_route_to_profile_state() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        // descent_state_mut: Some only on Pending; advancing persists.
        assert!(p.descent_state_mut().is_none());
        p.state = ProfileState::Pending(DescentState::new(
            r,
            DescentRemaining::from_vec(vec![CompactString::from("a"), CompactString::from("b")])
                .expect("non-empty"),
        ));
        p.descent_state_mut()
            .expect("Pending carries descent")
            .remaining_components_mut()
            .advance();
        assert_eq!(
            p.descent_state_mut()
                .expect("still Pending")
                .remaining_components()
                .as_slice(),
            &[CompactString::from("b")],
        );

        // mark/clear_active_for_reap delegate the bool semantics.
        let mut q = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(!q.mark_active_for_reap(), "Idle cannot be marked");
        assert!(!q.clear_active_reap(), "Idle has nothing to clear");
        q.state = active_prefire(r);
        assert!(q.mark_active_for_reap(), "Active flips to Reap");
        assert!(q.mark_active_for_reap(), "already-Reap is idempotent true");
        assert!(q.clear_active_reap(), "zombie revived");
        assert!(!q.clear_active_reap(), "nothing left to clear");
    }

    #[test]
    fn take_current_takes_and_is_idempotent() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.current = Some(TreeSnapshot::Dir(empty_dir_snapshot()));

        let taken = p.take_current();
        assert!(matches!(taken, Some(TreeSnapshot::Dir(_))));
        assert!(p.current().is_none(), "take leaves None");
        assert!(p.take_current().is_none(), "second take is idempotent");
    }

    /// Guarded random walk over the public state-machine mutators,
    /// asserting after every op: `baseline`/`current` share a
    /// `TreeSnapshot` variant when both set; `kind` agrees with
    /// `current`'s variant; a present `baseline` implies no survival
    /// witness. Guards read the live accessors so a step never trips a
    /// precondition instead of the invariant under test. Deterministic
    /// xorshift64 PRNG, seed pinned in the fn name; 16 fresh Profiles so
    /// the one-shot `materialize_anchor` is exercised.
    #[test]
    fn cross_field_invariants_hold_under_random_api_walk_seed_0x5eed_f00d() {
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

        fn assert_invariants(p: &Profile, op: &str) {
            if let (Some(b), Some(c)) = (p.baseline(), p.current()) {
                assert_eq!(
                    is_dir(b),
                    is_dir(c),
                    "inv1 (baseline/current share a variant) violated after {op}",
                );
            }
            if let (Some(k), Some(c)) = (p.kind(), p.current()) {
                assert_eq!(
                    matches!(k, ResourceKind::Dir),
                    is_dir(c),
                    "inv2 (kind matches current variant) violated after {op}",
                );
            }
            if p.baseline().is_some() {
                assert!(
                    p.last_settled_hash_at_loss().is_none(),
                    "inv3 (baseline ⇒ no survival witness) violated after {op}",
                );
            }
        }

        let mut master = XorShift64(0x5EED_F00D_D1CE_C0DE);
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        for _ in 0..16 {
            let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
            let mut rng = XorShift64(master.next_u64() | 1);
            assert_invariants(&p, "construction");

            for _ in 0..512 {
                match rng.below(9) {
                    0 => {
                        let kind_ok = matches!(p.kind(), None | Some(ResourceKind::Dir));
                        let base_ok = p
                            .baseline()
                            .is_none_or(|b| matches!(b, TreeSnapshot::Dir(_)));
                        if kind_ok && base_ok {
                            p.install_dir_current(empty_dir_snapshot());
                            assert_invariants(&p, "install_dir_current");
                        }
                    }
                    1 => {
                        let kind_ok = matches!(p.kind(), None | Some(ResourceKind::File));
                        let base_ok = p
                            .baseline()
                            .is_none_or(|b| matches!(b, TreeSnapshot::File(_)));
                        if kind_ok && base_ok {
                            p.install_file_current(empty_leaf_entry());
                            assert_invariants(&p, "install_file_current");
                        }
                    }
                    2 => {
                        p.clear_anchor_classification();
                        assert_invariants(&p, "clear_anchor_classification");
                    }
                    3 => {
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
                        p.rebase_baseline();
                        assert_invariants(&p, "rebase_baseline");
                    }
                    5 => {
                        p.transition_state(ProfileState::Idle);
                        assert_invariants(&p, "transition_state(Idle)");
                    }
                    6 => {
                        p.transition_state(pending(r));
                        assert_invariants(&p, "transition_state(Pending)");
                    }
                    7 => {
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
}
