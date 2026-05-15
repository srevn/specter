//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is computed at construction from
//! `(config, max_settle)` and is the lifetime-stable identity of the Profile.
//! `ProfileMap` keeps `(resource, config_hash) → ProfileId` and updates
//! `Resource.profiles` in lockstep — `attach`/`detach` are the only mutators
//! of either index.

use crate::effect::DedupKey;
use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
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
#[derive(Clone, Eq, PartialEq)]
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
    /// materialization on this edge and never calls [`advance`].
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

/// Identity of a past Effect emission *within one Profile's fire
/// history* ([`Profile::fired_subs`]).
///
/// Profile-free by construction: the owning `Profile` is the
/// `BTreeSet`'s container, so a `profile` discriminator would be
/// invariant across every entry — pure redundancy in both storage and
/// `Ord`. Contrast [`DedupKey`], which *does* carry `profile`: it is
/// the actuator's coalescing identity and credits the per-Profile
/// `Awaiting` counter on every `EffectComplete` in O(1) across the
/// actuator boundary, where no container implies the Profile. The two
/// identities are distinct concerns; conversion is one-way
/// ([`From<&DedupKey>`](FiredKey::from)) — the fire-history never needs
/// to reconstruct a routing key.
///
/// `Ord` drives the `BTreeSet`. `Hash` is intentionally not derived,
/// mirroring [`DedupKey`]: `core` bans `hashbrown`, so the only
/// container is a `BTreeSet`, never a `HashSet`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum FiredKey {
    Subtree(SubId),
    PerFile { sub: SubId, resource: ResourceId },
}

impl From<&DedupKey> for FiredKey {
    /// Project the actuator's coalescing identity onto the Profile's
    /// fire-history identity by dropping the (container-implied)
    /// `profile`. `DedupKey` is `Copy`, so `*dk` is a cheap read.
    fn from(dk: &DedupKey) -> Self {
        match *dk {
            DedupKey::Subtree { sub, .. } => Self::Subtree(sub),
            DedupKey::PerFile { sub, resource, .. } => Self::PerFile { sub, resource },
        }
    }
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// One stability state machine per `(Resource, ScanConfig)`.
///
/// The state-machine fields (`anchor`, `state`, `anchor_claim`) are
/// module-private — their invariants are enforced by the
/// setter/accessor API below, which is the only cross-crate write
/// surface (`specter-engine` cannot assign them directly). `anchor`
/// folds the anchor's kind, settled baseline, live snapshot, and
/// survival witness into one [`AnchorClassification`] sum so the
/// snapshot-shape and baseline/witness-exclusion invariants are
/// structural rather than enforced by convention at clear sites. The
/// remaining fields stay `pub` for engine read/write.
#[derive(Debug)]
pub struct Profile {
    pub resource: ResourceId,
    pub config: ScanConfig,
    pub exclude_strings: Arc<[CompactString]>,
    pub config_hash: u64,
    /// The anchor's classification and settled reference, as one sum
    /// (kind ⊕ live snapshot ⊕ settled baseline ⊕ survival witness).
    ///
    /// The discriminant *is* the anchor kind: there is no parallel
    /// `kind` field to drift from the snapshot variant, and the
    /// "no snapshot while unclassified" / "no baseline while a
    /// survival witness is held" rules hold by construction. Reads go
    /// through [`Self::kind`] / [`Self::current`] / [`Self::baseline`]
    /// / [`Self::current_dir`] / [`Self::baseline_dir`] /
    /// [`Self::settled_hash`] / [`Self::current_is_some`]; writes
    /// through [`Self::install_dir_current`] /
    /// [`Self::install_file_current`] (graft), [`Self::rebase_baseline`]
    /// (settle the current snapshot as the baseline),
    /// [`Self::take_current`] (release the covered-descendant claim),
    /// [`Self::clear_anchor_classification`] (anchor loss — captures
    /// the survival witness), and [`Self::materialize_anchor`]
    /// (descent materialisation — carries the witness forward).
    ///
    /// **Lifecycle.** `Unclassified` while the anchor's kind is
    /// unknown (descent in flight, fresh resource-attach against an
    /// `Unknown` slot, or after anchor loss). `File` / `Dir` from the
    /// materialisation moment until the next loss. The kind is fixed
    /// for a materialised epoch — an on-disk kind change surfaces as a
    /// probe `Vanished`, recovers through descent, and re-materialises
    /// the sum, never mutates the discriminant in place.
    ///
    /// **Coherence with `Resource.kind`.** The Tree slot's `kind` is a
    /// parallel cache updated by reconcile / `Tree::set_kind`; the
    /// engine reads the anchor's kind here, never `Tree.kind` for the
    /// anchor's own kind in any post-attach path, so a slot left stale
    /// across a loss/recover cycle for a shared anchor is never
    /// observed.
    anchor: AnchorClassification,
    /// Sole post-construction writer: [`Self::transition_state`]; read
    /// via [`Self::state`].
    state: ProfileState,
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
    anchor_claim: AnchorClaim,
    /// Set of [`FiredKey`]s for which this Profile has emitted at least
    /// one Effect that has not been cleared by a `Failed` outcome,
    /// `detach_sub`, or covered-leaf reap. Pure existence — no value
    /// payload. Drives drift recovery's "should we conservative-fire?"
    /// question by gating the `SeedDrift` filter; B1 dedup derives
    /// directly from `baseline.hash() == current.hash()` and does not
    /// consult this field. The `profile` axis of [`DedupKey`] is dropped
    /// on the way in ([`FiredKey::from`]) — this set's owner *is* the
    /// Profile, so carrying it per entry would be redundant.
    ///
    /// **Lifecycle.** Inserted at successful emit (`emit_effects` Subtree
    /// and PerFile arms). Removed on `EffectComplete::Failed`,
    /// `detach_sub_inner`, and `purge_per_file_fired_subs_for_reaped_slots`.
    /// Preserved across anchor loss by `discard_anchor_state` — the fire
    /// history is the answer to "which Subs should re-fire on recovery if
    /// drift is detected?"
    pub fired_subs: BTreeSet<FiredKey>,
    /// User-declared event-class mask for this Profile. Every Sub on a
    /// Profile shares this by construction (the mask folds into
    /// `config_hash`), so it is invariant for the Profile's lifetime.
    /// Module-private — [`Self::events`] is the stable read seam; the
    /// per-Resource union aggregates this across covering Profiles.
    events: ClassSet,
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
    /// `events` becomes the Profile's event-class mask and drives
    /// `has_per_file_fds` (true iff CONTENT or METADATA is in the mask).
    /// Every Sub on a Profile shares the same `events`, so it is
    /// invariant for the Profile's lifetime.
    ///
    /// `exclude_strings` is projected once here from `config.exclude` —
    /// the [`ScanConfig`] builder has already sorted the vector by source,
    /// so the projection is canonical without re-sorting.
    ///
    /// `kind` is the anchor's classified shape at construction,
    /// projected into the [`AnchorClassification`] sum: `None` ⇒
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
            config,
            exclude_strings,
            config_hash,
            anchor,
            state: ProfileState::Idle,
            parent_profile: None,
            dirty_descendants: 0,
            max_settle,
            settle,
            watch_root_parent: None,
            anchor_claim: AnchorClaim::None,
            fired_subs: BTreeSet::new(),
            events,
            has_per_file_fds,
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
    /// Called from `dispatch_rebase_ok` and from both branches of
    /// `dispatch_seed_ok` after a successful graft, where
    /// `current.is_some()` holds.
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
            matches!(self.anchor_claim, AnchorClaim::None),
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
        self.anchor_claim = AnchorClaim::Held;
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
    /// representable [`AnchorClassification`] violates them, so there
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
                matches!(self.anchor_claim, AnchorClaim::None),
                "anchor coherence: a Pending Profile must not hold the anchor claim",
            );
        }
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
    /// over the module-private field.
    #[must_use]
    pub const fn events(&self) -> ClassSet {
        self.events
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
    use super::{
        AnchorClassification, ClassSet, Profile, ProfileMap, ProfileState, ScanConfig,
        SettledState, compute_config_hash,
    };
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
        assert!(p.baseline().is_none());
        assert!(!p.current_is_some());
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
        assert_eq!(p.events(), ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (closes
    /// E2E #3 by default for `subtree-root`).
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT, None);
        assert!(p.has_per_file_fds);
        assert_eq!(p.events(), ClassSet::CONTENT);
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
    fn rebase_baseline_settles_current_as_baseline() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
            d.remaining_components().iter().cloned().collect::<Vec<_>>(),
            vec![CompactString::from("b")]
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
    fn read_accessors_project_the_sum() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);

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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);
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
    #[should_panic(expected = "anchor must be Unclassified")]
    fn materialize_anchor_panics_when_already_classified() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);
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
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("b")],
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
    fn take_current_takes_leaves_settled_and_is_idempotent() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut q = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
            let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
            let mut rng = XorShift64(master.next_u64() | 1);
            assert_invariants(&p, "construction");

            for _ in 0..512 {
                match rng.below(9) {
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

    /// `Profile::new`'s `kind` → sum projection is total: `None` ⇒
    /// `Unclassified` (state A), `Some(Dir)` / `Some(File)` ⇒ a
    /// classified anchor with no snapshot or baseline (state C′).
    #[test]
    fn profile_new_projects_kind_to_initial_state() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("anchor", ResourceRole::User);

        let a = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        assert!(matches!(
            a.anchor,
            AnchorClassification::Unclassified { witness: None }
        ));
        assert_eq!(a.kind(), None);
        assert_eq!(a.settled_hash(), None);

        let c_dir = Profile::new(
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

        let c_file = Profile::new(
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
        let _ = Profile::new(
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);

        p.debug_assert_anchor_coherent(); // Idle + Unclassified
        p.state = pending(r);
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
        let mut p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS, None);
        p.state = pending(r);
        p.anchor = AnchorClassification::Dir {
            current: None,
            settled: SettledState::Unset,
        };
        p.debug_assert_anchor_coherent();
    }
}
