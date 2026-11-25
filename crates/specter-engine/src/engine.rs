//! `Engine` — pure, deterministic, total.
//!
//! The engine owns the data model (`Tree`, `ProfileMap`, `SubRegistry`), the timer wheel, and the
//! stability index; `step` consumes one [`Input`] at a time and emits a sorted [`StepOutput`].
//! State-machine bodies live in sibling modules:
//! - `burst.rs` — Idle ↔ Active phase transitions.
//! - `transitions.rs` — per-input handlers (`on_fs_event`, etc.).
//! - `reconcile.rs` — newly-discovered descendants.
//! - `refcounts.rs` — `watch_demand` (contributions-map) edges.
//!
//! `step` is the single dispatch point; each `Input` variant routes to the corresponding `on_*`
//! handler — Sub attachment and detachment enter as `Input::AttachSub` / `Input::DetachSub` through
//! `step`, not as standalone methods (`attach_sub_inner` / `detach_sub_inner` are `pub(crate)`
//! internals).

use crate::counter::MonotonicCounter;
use crate::path::empty_path;
use crate::timer::{TimerEntry, TimerHeap};
use crate::transitions::ParkNarration;
use compact_str::format_compact;
// Only the inline test module names `CompactString`, reaching it via `use super::*`; gating keeps
// the lib build warning-free.
#[cfg(test)]
use compact_str::CompactString;
// Identity.
use specter_core::{ProfileId, ResourceId, SubId, TimerId};
// Tree + path validation.
use specter_core::Tree;
// Profile state machine.
use specter_core::{
    AnchorClaim, BurstFinish, DescentRemaining, DescentState, DescentWitness, DetachLifecycle,
    DetachReason, Profile, ProfileMap, ProfileState, ReapTrigger, TimerKind,
};
// Registries.
use specter_core::{
    MintEntry, ProfileIdentity, Sub, SubAttachAnchor, SubAttachRequest, SubParams, SubRegistry,
};
// Per-Resource bookkeeping.
use specter_core::ResourceKind;
// Probe + effect correlation.
use specter_core::{CorrelationId, ProbeCorrelation};
// Engine step I/O.
use specter_core::{Diagnostic, FsEvent, Input, StepOutput};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-call stale-drain bound for [`Engine::pop_expired`].
///
/// Realistic worst case per call is single-digit (each Active Profile carries ~2 timer slots and
/// orphans at most one per burst transition; the bin's tick loop polls frequently enough that
/// orphans collect incrementally rather than piling at the heap top). The bound is loose enough to
/// absorb multi-Profile burst-end cleanup yet tight enough to surface an "engine transition leaks
/// timer references" regression in dev/CI.
const STALE_DRAIN_BOUND: u32 = 32;

/// `pub(crate)` field visibility lets sibling modules read/write engine state directly. External
/// consumers go through the public methods.
///
/// Per-Profile descent state lives inline on the Profile's state enum
/// (`ProfileState::Pending(DescentState)`). Read through `Engine::descent_state` /
/// `Engine::descent_state_mut` (both `pub(crate)`); per-event fan-out lives next to its sole
/// consumer (`Engine::classify_event_carriers` in `transitions.rs`).
#[derive(Debug, Default)]
pub struct Engine {
    pub(crate) tree: Tree,
    pub(crate) profiles: ProfileMap,
    pub(crate) subs: SubRegistry,
    pub(crate) timers: TimerHeap,
    /// The engine-wide [`ProbeCorrelation`] monotone floor — the one irreducible engine-resident
    /// probe datum now that every probe-bearing fact homes on its owner's state slot. Driven solely
    /// by [`Engine::mint_probe_correlation`]; see [`crate::probe`] for the mint contract and the
    /// state-derived projection surface beside it. Phantom-typed distinct from
    /// `effect_correlations`: the two id spaces stay structurally separate at the type level.
    pub(crate) correlations: MonotonicCounter<ProbeCorrelation>,
    /// Monotonic counter for [`CorrelationId`] minting. Bumped at every `Effect` push in
    /// `transitions.rs::emit_effects` to stamp each emission with a process-unique narration id —
    /// completion routing is `DedupKey`-keyed and never reads it; see
    /// [`specter_core::Effect::correlation`] for what the actuator actually consumes it for.
    /// Phantom-typed distinct from the probe-side counter: the two id spaces stay structurally
    /// separate at the type level.
    pub(crate) effect_correlations: MonotonicCounter<CorrelationId>,
    /// Reusable relative-path buffer for [`crate::coverage::covers`], owned at engine scope so its
    /// capacity survives across `step` calls — under a keeps-up storm the per-event covering walk
    /// and the per-fire reconcile walk reuse one allocation rather than minting two `PathBuf`s per
    /// covered-descendant test. Logically per-`covers`-call (cleared at the start of each build),
    /// so its cross-call residue is never observable state; the engine stays a
    /// pure `Input -> StepOutput` machine. Not thread-local, not
    /// interior-mutable: an explicit `&mut` threaded only into the two hot paths; the cold
    /// pure-derivation queries keep a local.
    pub(crate) coverage_scratch: PathBuf,
    /// Debug-only consume-once tripwire — the cross-step witness that no [`ProbeCorrelation`]
    /// reaches a `dispatch_*` arm twice. The structural laws (core slot arm-once,
    /// [`Engine::take_owner_probe`] disarm-once) make the violation unconstructable; this records
    /// the per-owner high-water dispatched correlation so a property test or fuzzer surfaces a
    /// regression as a panic. Absent in release: zero cost, zero footprint.
    #[cfg(debug_assertions)]
    pub(crate) dispatch_ledger: crate::probe::DispatchLedger,
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Descent-state accessor. Returns the `DescentState` payload of the Profile's
    /// `ProfileState::Pending` variant; `None` for Profiles not currently descending, stale ids, or
    /// any other state.
    ///
    /// Sole reader API for the descent-state payload outside the routing match sites in
    /// `on_*_probe_response`; the state-type projection ([`ProfileState::descent_state`]) owns the
    /// in-variant payload match, so the dispatcher here stays a thin route.
    #[must_use]
    pub(crate) fn descent_state(&self, owner: ProfileId) -> Option<&DescentState> {
        self.profiles.get(owner)?.state().descent_state()
    }

    /// Mutable counterpart to [`Engine::descent_state`].
    pub(crate) fn descent_state_mut(&mut self, owner: ProfileId) -> Option<&mut DescentState> {
        self.profiles.get_mut(owner)?.descent_state_mut()
    }

    /// Pure, deterministic, total. Consumes one [`Input`], emits a sorted [`StepOutput`]. Each
    /// variant routes to the corresponding `on_*` handler (`transitions.rs`) or registration entry
    /// (`attach_sub_inner` / `detach_sub_inner`). Exhaustive — adding a variant to [`Input`] is a
    /// compile error here until a handler lands.
    ///
    /// Lifecycle inputs ([`Input::AttachSub`], [`Input::DetachSub`]) take effect on the engine's
    /// registries and narrate the outcome via `out.diagnostics` ([`Diagnostic::SubAttached`]), not
    /// via a synchronous return. Identity (`name → SubId`) is resolved engine-side through the
    /// registry's `by_name` index, so the dispatcher's uniform shape (one input, one
    /// [`StepOutput`]) holds across every variant.
    ///
    /// **MUST NOT be wrapped in `catch_unwind`.** [`specter_core::ProbeSlot`]'s in-unwind silence
    /// depends on a mid-`step` panic being fatal — *and* on the engine thread being the only thread
    /// that mutates `ProbeSlot`s, so an unwind on any other thread never reaches an armed slot. The
    /// driver's `tick` / `run` carry the matching contract; this is the engine-side mirror, the seam
    /// any library or test consumer of the engine crate that bypasses the driver first encounters.
    pub fn step(&mut self, input: Input, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        match input {
            Input::FsEvent { resource, event } => {
                self.on_fs_event(resource, event, now, &mut out);
            }
            Input::ProbeResponse(resp) => {
                self.on_probe_response(resp, now, &mut out);
            }
            Input::TimerExpired { profile, kind, id } => {
                self.on_timer_expired(profile, kind, id, now, &mut out);
            }
            Input::EffectComplete(completion) => {
                self.on_effect_complete(
                    completion.sub,
                    &completion.key,
                    &completion.outcome,
                    now,
                    &mut out,
                );
            }
            Input::WatchOpRejected { resource, failure } => {
                self.on_watch_op_rejected(resource, failure, &mut out);
            }
            Input::ConfigDiff(diff) => {
                self.on_config_diff(diff, now, &mut out);
            }
            Input::SensorOverflow { scope } => {
                self.on_sensor_overflow(scope, now, &mut out);
            }
            Input::AttachSub(req) => self.attach_sub_inner(req, now, &mut out),
            Input::DetachSub(sub) => {
                // [`Input::DetachSub`] is the engine's public detach surface. Its sole external
                // producer is the bin's IPC `disable` handler. Hardcode the canonical reason here
                // so the engine's public input shape stays a single-arg `SubId` rather than
                // widening with a caller-supplied reason that has exactly one valid value in v1.
                // Internal call sites (`on_config_diff`, the discovery cascade) call
                // `detach_sub_inner` directly with their own reason.
                self.detach_sub_inner(sub, DetachReason::IpcDisabled, &mut out);
            }
            Input::ArmAbsorb { profile, duration } => {
                self.on_arm_absorb(profile, duration, now, &mut out);
            }
        }
        out.sort_for_emission();
        out
    }

    /// Attach a Sub at its [`SubAttachAnchor`] — a caller-supplied `Resource` slot or a `Path` the
    /// engine materialises. Reuses an existing Profile when `(anchor, config_hash)` matches
    /// ([`Self::join_profile`]); otherwise creates a fresh Profile and seeds it at its anchor
    /// ([`Self::seed_at_anchor`]: the anchor's and its parent's claims, then a Seed burst whose
    /// `Stable` verdict pins the baseline) — at once when the anchor is on disk, or at the terminus
    /// of the pending descent it enters when it is not. Nothing but intent stands behind a request,
    /// so its fresh anchor seeds cold and its descent enters unwitnessed.
    ///
    /// Sole public entry is [`Input::AttachSub`] via [`Self::step`]; `pub(crate)` so
    /// [`Self::on_config_diff`] can compose detaches and attaches into one [`StepOutput`]. A Sub a
    /// template mints enters through [`Self::mint_sub`] instead.
    ///
    /// Total: an unresolvable anchor attaches nothing and narrates a typed diagnostic —
    /// [`Diagnostic::AttachPathInvalid`] for a malformed `Path`, [`Diagnostic::AttachResourceStale`]
    /// for a `Resource` with no live slot. External callers reconcile their `name → SubId` index from
    /// the [`Diagnostic::SubAttached`] / `AttachPathInvalid` / `AttachResourceStale` stream.
    ///
    /// # Production invariants (`Path` anchor)
    ///
    /// 1. **Absolute paths only.** A [`SubAttachAnchor::Path`] must be absolute and UTF-8.
    ///    [`Tree::parse_attach_path`] is the canonical gate; it rejects non-absolute paths,
    ///    non-UTF-8 segments, `.` / `..` components, Windows path prefixes, and empty segments. The
    ///    config layer's `canonicalize_lenient` already enforces absolute paths for TOML-loaded
    ///    configs, but hot-reload `ConfigDiff::added` constructs `SubAttachRequest` from a
    ///    different path; the gate keeps the engine's contract independent of every caller.
    /// 2. **Single FS-root.** Every validated [`specter_core::TreePath`] starts with
    ///    [`specter_core::FS_ROOT_SEGMENT`]; `materialize_path_or_pending` lazily bootstraps a
    ///    synthetic `/` slot (role `ResourceRole::DescentScaffold`) before the pre-existence walk
    ///    so every Profile's rewind chain terminates at this shared slot. The FS-root invariant is
    ///    documented here rather than enforced at the Tree type level — unit tests for lower-level
    ///    Tree functions (`coverage`, `refcounts`) still construct multi-root trees outside of the
    ///    attach pipeline.
    /// 3. **`Tree::path_of` reconstructs absolute paths.** `PathBuf::push("/")` resets the buffer
    ///    to absolute, so the Sensor's `WatchOp::Watch { path }` always carries an absolute path
    ///    for any Profile registered through this pipeline.
    ///
    /// # Pipeline
    ///
    /// 1. **Identity resolution.** `resolve_attach_anchor` resolves the request's `anchor` —
    ///    parsing and materialising a `Path`, or liveness-checking a `Resource` — and yields a
    ///    typed [`AnchorResolution`] indicating whether the anchor is materialised (`Immediate`) or
    ///    scaffolded (`Pending`). `find_or_create_profile` then classifies the `(anchor,
    ///    config_hash)` lookup into a [`ProfileOrigin`].
    /// 2. **Sub registration** ([`Self::register_sub`]).
    /// 3. **Per-origin bookkeeping.** A Profile that already stood takes the join; a fresh one
    ///    seeds at its anchor ([`Self::seed_at_anchor`]) when it is live, else only enters the
    ///    descent ([`Self::enter_pending_descent`]), whose terminus runs the same tail when the
    ///    anchor appears.
    pub(crate) fn attach_sub_inner(
        &mut self,
        req: SubAttachRequest,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // A discovery template and the `MatchChain` shape are coupled iff: a template on a
        // non-chain Profile could never reconcile, and a plain Sub on a chain Profile could never
        // react (its Profile mints attachments, not Effects).
        debug_assert_eq!(
            req.params.is_template(),
            req.identity.config().match_chain().is_some(),
            "attach_sub_inner: ReactionSpec::Mint ⟺ ScanConfig::MatchChain \
             (a template mints; a chain Profile reconciles — neither exists without the other)",
        );

        let SubAttachRequest {
            anchor,
            identity,
            params,
        } = req;
        let Some(resolved) = self.resolve_attach_anchor(&anchor, out) else {
            return;
        };
        let (profile_id, origin) =
            self.find_or_create_profile(resolved.anchor(), identity, params.settle);
        self.register_sub(Sub::from_request(profile_id, params), out);
        match origin {
            ProfileOrigin::Fresh => match resolved {
                AnchorResolution::Immediate { .. } => {
                    self.seed_at_anchor(profile_id, SeedWitness::Cold, now, out);
                }
                AnchorResolution::Pending {
                    prefix, remaining, ..
                } => {
                    // A terminus whose segments were all present on first observation pins silently
                    // (attach-over-existing must not fire), while an appearance after the attach is
                    // witnessed by the probes themselves — an absence, then a presence.
                    self.enter_pending_descent(
                        profile_id,
                        prefix,
                        remaining,
                        DescentWitness::Unobserved,
                        out,
                    );
                }
            },
            ProfileOrigin::Existing(join) => self.join_profile(profile_id, join, now, out),
        }
    }

    /// Seat the Sub `template` mints at `terminus`, entering its match set as `entry` — the one way
    /// a minted Sub comes to be. `terminus` is a live slot the pass has resolved, its snapshot
    /// lists it as `kind`, and it holds no Sub of `template`'s (the pass's dedup); nothing is
    /// resolved or parsed here.
    ///
    /// The Profile is found or created at `terminus` under the spec's minted identity. A fresh one
    /// seeds at its anchor with the entry's witness: an enumeration pins cold, an appearance owes
    /// its first fire. One that already stood takes the join, and an appearance is noted into
    /// whatever burst it runs — a STRUCTURE event at the anchor, which is what an appearance is —
    /// so its verdict fires instead of pinning cold; a Pending or Parked Profile drops the note,
    /// its descent keeping its own witness.
    ///
    /// Narrates [`Diagnostic::DiscoveryMinted`] last, after the Sub's own `SubAttached`.
    pub(crate) fn mint_sub(
        &mut self,
        template: SubId,
        terminus: ResourceId,
        kind: ResourceKind,
        entry: MintEntry,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let path = self.anchor_path(terminus);
        // `<template>@<path>`: config validation reserves `@`, so no operator name collides with a
        // minted one in `by_name`, and the first `@` splits prefix from path — no two mints
        // synthesise one name.
        let Some((name, spec)) = self.subs.get(template).and_then(|t| {
            let spec = Arc::clone(t.mint_spec()?);
            Some((format_compact!("{}@{}", t.name, path.display()), spec))
        }) else {
            // A missed mint costs one pass: the pass is idempotent, and the next one mints it.
            debug_assert!(
                false,
                "mint_sub: the pass hands over a live template (template = {template:?})",
            );
            return;
        };
        // A minted Sub reacts, so its Profile must fire: on a chain Profile it would never react.
        debug_assert!(
            spec.identity.config().match_chain().is_none(),
            "mint_sub: a minted identity is never a MatchChain (template = {template:?})",
        );
        // A Profile takes its anchor's kind from the slot when it is founded, and the pass meets
        // slots the graft never kinded: one a watch rejection reset, one a reap took and the pass
        // made anew.
        self.tree.set_kind(terminus, kind);
        let (profile_id, origin) =
            self.find_or_create_profile(terminus, spec.identity.clone(), spec.settle);
        self.register_sub(Sub::minted(profile_id, name, template, &spec), out);
        match origin {
            ProfileOrigin::Fresh => self.seed_at_anchor(profile_id, entry.into(), now, out),
            ProfileOrigin::Existing(join) => {
                self.join_profile(profile_id, join, now, out);
                if entry == MintEntry::Appeared {
                    self.drive_burst(
                        profile_id,
                        terminus,
                        &path,
                        FsEvent::StructureChanged,
                        now,
                        out,
                    );
                }
            }
        }
        out.diagnostics.push(Diagnostic::DiscoveryMinted {
            source: template,
            path,
            kind,
            entry,
        });
    }

    /// Phase 1 of `attach_sub_inner` — resolve the request's anchor to a Tree slot, classifying the
    /// outcome as `Immediate` (anchor is materialised on disk) or `Pending` (anchor is a scaffold;
    /// the engine must descend before the burst can start).
    ///
    /// A [`SubAttachAnchor::Path`] routes through [`Tree::parse_attach_path`] (which rejects
    /// non-absolute, non-UTF-8, `.` / `..`, Windows-prefix, and empty-segment paths) and then
    /// [`Self::materialize_path_or_pending`]; a malformed path emits
    /// [`Diagnostic::AttachPathInvalid`] and returns `None`.
    ///
    /// A [`SubAttachAnchor::Resource`] re-gates the caller's claimed slot with an O(1) liveness
    /// check: a stale id emits [`Diagnostic::AttachResourceStale`] and returns `None`, otherwise it
    /// resolves [`AnchorResolution::Immediate`] (a `Resource` anchor never descends).
    fn resolve_attach_anchor(
        &mut self,
        anchor: &SubAttachAnchor,
        out: &mut StepOutput,
    ) -> Option<AnchorResolution> {
        match anchor {
            SubAttachAnchor::Path(path) => {
                let parsed = match Tree::parse_attach_path(path) {
                    Ok(p) => p,
                    Err(err) => {
                        out.diagnostics.push(Diagnostic::AttachPathInvalid {
                            path: std::sync::Arc::from(path.as_path()),
                            hint: err.hint(),
                        });
                        return None;
                    }
                };
                Some(match self.materialize_path_or_pending(&parsed) {
                    crate::descent::MaterializeResult::Materialized(anchor) => {
                        AnchorResolution::Immediate { anchor }
                    }
                    crate::descent::MaterializeResult::Pending {
                        anchor,
                        prefix,
                        remaining,
                    } => AnchorResolution::Pending {
                        anchor,
                        prefix,
                        remaining,
                    },
                })
            }
            SubAttachAnchor::Resource(r) => {
                // The caller claims `r` is already live. Re-gate it at the boundary — the request
                // crossed a channel and cannot hold a Tree borrow, so liveness is verified here,
                // symmetric with the Path arm's `parse_attach_path` re-check. A stale slot is
                // dropped gracefully rather than panicking downstream.
                if self.tree.get(*r).is_none() {
                    out.diagnostics
                        .push(Diagnostic::AttachResourceStale { resource: *r });
                    return None;
                }
                Some(AnchorResolution::Immediate { anchor: *r })
            }
        }
    }

    /// Pure pre-check that returns `true` iff a subsequent [`Self::attach_sub_inner`] call with
    /// `req` would clear its only fallible boundary — `Tree::parse_attach_path` for a `Path`
    /// anchor, or slot liveness for a `Resource` anchor. Mutates nothing; never installs a
    /// scaffold, never promotes a role.
    ///
    /// Sole consumer is [`Self::on_config_diff`]'s `modified_identity` arm: it runs the validate
    /// *before* detaching the old Sub so a malformed path doesn't tear down a live attachment for
    /// nothing. The detach + attach pair is then total — validate said yes, so the attach won't
    /// surface a parse error.
    ///
    /// Emits the same diagnostic [`Self::attach_sub_inner`] would on the failure path
    /// (`AttachPathInvalid` / `AttachResourceStale`), so the validate-then-act site never re-emits
    /// on its own. The re-parse at attach time (one `Tree::parse_attach_path` call, O(path length),
    /// pure) is the cost of total composition; the attach has no engine-state precondition the
    /// validate could lock in.
    pub(crate) fn validate_sub_attach(&self, req: &SubAttachRequest, out: &mut StepOutput) -> bool {
        match &req.anchor {
            SubAttachAnchor::Path(p) => match Tree::parse_attach_path(p) {
                Ok(_) => true,
                Err(err) => {
                    out.diagnostics.push(Diagnostic::AttachPathInvalid {
                        path: std::sync::Arc::from(p.as_path()),
                        hint: err.hint(),
                    });
                    false
                }
            },
            SubAttachAnchor::Resource(r) => {
                if self.tree.get(*r).is_some() {
                    true
                } else {
                    out.diagnostics
                        .push(Diagnostic::AttachResourceStale { resource: *r });
                    false
                }
            }
        }
    }

    /// File `sub` and narrate [`Diagnostic::SubAttached`] — the one point a `SubId` enters the
    /// registry, for either entry, and the line's sole emitter. The line is operator narration:
    /// identity is resolved engine-side through the registry's `by_name` index, not from it.
    fn register_sub(&mut self, sub: Sub, out: &mut StepOutput) {
        let name = sub.name.clone();
        let minted_by = sub.minted_by();
        let sub = self.subs.insert(sub);
        out.diagnostics.push(Diagnostic::SubAttached {
            sub,
            name,
            minted_by,
        });
    }

    /// A Sub joins a Profile that already stood ([`Join`]): it may lower the settle floor; a
    /// zombie's deferred reap is cancelled, its in-flight burst running on under the new Sub set; a
    /// parked Profile re-arms its recovery descent — a join is the in-band equivalent of the
    /// operator restart a park waits for, and a re-rejected watch re-parks, bounded by the attach
    /// rate. No watch moves: the events mask folds into the identity hash, so the joiner shares the
    /// Profile's mask.
    ///
    /// **An Idle discovery Profile reseeds**, cold. Only a template joins a chain Profile, and it
    /// arrives with no tree event: without one fresh certified walk it would starve until external
    /// churn. The walk's pass mints for the joiner cold, its enumeration still open — a join over
    /// an existing tree is attach-over-existing. Every other state already converges: a descent's
    /// terminus seeds, an Active pass reads its templates off the live registry, and a zombie's
    /// revived burst runs its pass. The Idle test is this arm's own: `start_seed_burst`'s guard
    /// narrates a contract violation.
    fn join_profile(
        &mut self,
        profile_id: ProfileId,
        join: Join,
        now: Instant,
        out: &mut StepOutput,
    ) {
        self.recompute_profile_settle(profile_id);
        match join {
            Join::Zombie => self.revive_zombie(profile_id, out),
            Join::Parked => self.start_pending_recovery(profile_id, out),
            Join::Live => {
                if self.profiles.get(profile_id).is_some_and(|p| {
                    p.config().match_chain().is_some() && matches!(p.state(), ProfileState::Idle)
                }) {
                    self.start_seed_burst(profile_id, None, now, out);
                }
            }
        }
    }

    /// The zombie arm of [`Self::join_profile`]: the deferred-reap detach flipped the Active
    /// burst's finish directive to [`BurstFinish::Reap`], and the join flips it back
    /// ([`ProfileState::clear_active_reap`] — `true` by construction of [`Join::Zombie`], which the
    /// `debug_assert!` pins against a routing breach), narrating
    /// [`Diagnostic::ReapPendingCancelled`]. A second join in the same step finds it live.
    fn revive_zombie(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let cleared = self
            .profiles
            .get_mut(profile_id)
            .is_some_and(specter_core::Profile::clear_active_reap);
        debug_assert!(
            cleared,
            "revive_zombie: Join::Zombie must clear an active Reap directive \
             (profile = {profile_id:?})",
        );
        out.diagnostics.push(Diagnostic::ReapPendingCancelled {
            profile: profile_id,
        });
    }

    /// The anchor is live: install its claim and its parent's, then start the Seed — the one tail
    /// of every anchor that goes live, whether an attach or a mint founded its Profile on it or a
    /// descent's terminus just found it (`materialize_profile_anchor`).
    ///
    /// Sequence:
    /// 1. Promote the anchor slot to `User` — a no-op where a real role stands, so a co-resident
    ///    peer's `WatchRootParent` / `User` is never clobbered (the tag is metadata).
    /// 2. Install the anchor claim ([`Self::install_anchor_claim`]) and the watch-root parent claim
    ///    ([`Self::install_watch_root_parent_claim`] — a recovery descent's outlived the loss, so
    ///    that install adds nothing).
    /// 3. Start the Seed burst, selected by `witness`. `Cold` opens the cold arm (`Verifying` at
    ///    construction). `Appeared` is drained into its two readers, one per kind of reaction:
    ///    every template on the Profile closes its enumeration ([`Self::prefix_appeared`]), and the
    ///    anchor threads as the trigger, so the Seed opens `Batching`-first and a firing Sub's
    ///    stable verdict owes the first fire. The trigger's only job is landing in the Seed's
    ///    `dirty`, where `seed_owes_first_fire` finds the witness — every Seed probe targets the
    ///    anchor regardless. The settle window debounces only what the anchor's own watch delivers
    ///    — a file's writes, a directory's direct entries; below a directory nothing is watched
    ///    until the verdict's graft installs it.
    ///
    /// Touches no descent state: the descent terminus finishes its own — the flip, its prefix claim
    /// — around this call.
    pub(crate) fn seed_at_anchor(
        &mut self,
        profile_id: ProfileId,
        witness: SeedWitness,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(anchor) = self.profiles.get(profile_id).map(Profile::resource) else {
            return;
        };
        self.tree
            .promote_scaffold(anchor, specter_core::ResourceRole::User);
        self.install_anchor_claim(profile_id, out);
        self.install_watch_root_parent_claim(profile_id, out);
        let trigger = match witness {
            SeedWitness::Cold => None,
            SeedWitness::Appeared => {
                self.prefix_appeared(profile_id);
                Some((anchor, self.anchor_path(anchor)))
            }
        };
        self.start_seed_burst(profile_id, trigger, now, out);
    }

    /// The path of an anchor slot — a live Profile's, or the terminus a mint is about to anchor:
    /// the event path an observed appearance carries into a Seed's trigger or an existing burst's
    /// `dirty`, a minted Sub's name, and the line a reaped one leaves. The slot is held — by the
    /// Profile's back-reference, or by the pass that just resolved the terminus — so the path
    /// resolves; a miss is a state-machine breach, loud in debug, and degrades to the empty path in
    /// release rather than dropping the appearance (an empty path in `dirty` still counts as
    /// activity).
    pub(crate) fn anchor_path(&self, anchor: ResourceId) -> Arc<Path> {
        self.tree.path_of(anchor).unwrap_or_else(|| {
            debug_assert!(
                false,
                "a held anchor slot resolves a path (resource = {anchor:?})",
            );
            empty_path()
        })
    }

    /// Find the Profile at `(anchor, identity.config_hash())`, classifying how a Sub joins it, or
    /// create one whose settle floor is the attaching Sub's `settle`.
    ///
    /// `identity` is taken by value: the `Fresh` arm moves it straight into [`Profile::new`]; a
    /// standing Profile drops it. The anchor's classified kind is read from the Tree slot and
    /// threaded through the constructor: `None` for a `DescentScaffold` anchor (descent
    /// materialisation classifies it) or a freshly-`ensure`d-but-unprobed slot (the first Seed-Ok
    /// classifies it). `profiles.attach` registers only Profile-side indices on the slot, never its
    /// `kind`, so the read before construction is the read after.
    fn find_or_create_profile(
        &mut self,
        anchor: ResourceId,
        identity: ProfileIdentity,
        settle: Duration,
    ) -> (ProfileId, ProfileOrigin) {
        if let Some(pid) = self.profiles.find(anchor, identity.config_hash()) {
            let join = match self.profiles.get(pid).map(Profile::state) {
                Some(ProfileState::Active(_, BurstFinish::Reap)) => Join::Zombie,
                Some(ProfileState::Parked) => Join::Parked,
                _ => Join::Live,
            };
            return (pid, ProfileOrigin::Existing(join));
        }
        let anchor_kind = self.tree.get(anchor).and_then(specter_core::Resource::kind);
        let p = Profile::new(anchor, identity, settle, anchor_kind);
        let pid = self.profiles.attach(&mut self.tree, p);
        (pid, ProfileOrigin::Fresh)
    }

    /// Detach a Sub by id. The last Sub to leave a Profile reaps it, at once or at burst end
    /// ([`DetachLifecycle`]); otherwise the Profile's settle floor is recomputed over the Subs that
    /// remain. A stale id narrates [`Diagnostic::DetachUnknownSub`] and changes nothing; `reason`
    /// rides the [`Diagnostic::SubDetached`] of a Sub that left.
    ///
    /// Takes no `now`: nothing here schedules a timer or moves a burst, and a burst on a detached
    /// Profile runs on under its own schedule.
    ///
    /// **A template's minted set follows it**, whatever `reason` removed the template (IPC disable,
    /// config removal, identity change): [`SubRegistry::remove`] hands back what it minted and
    /// [`Self::reap_orphans`] detaches each.
    pub(crate) fn detach_sub_inner(
        &mut self,
        sub: SubId,
        reason: DetachReason,
        out: &mut StepOutput,
    ) {
        let Some(removed) = self.subs.remove(sub) else {
            out.diagnostics.push(Diagnostic::DetachUnknownSub { sub });
            return;
        };
        let profile_id = removed.sub.profile();
        // Narrated before the cascade and the reap below: `sort_for_emission` leaves diagnostics in
        // insertion order, so emitting in causal order is what makes a reaped Profile read as the
        // consequence of its Sub's detach.
        out.diagnostics.push(Diagnostic::SubDetached {
            sub,
            profile: profile_id,
            reason,
        });

        // A template's orphans detach before its own Profile bookkeeping: each minted Sub's detach
        // fully resolves (its Profile reaps) while causally downstream of the template's
        // `SubDetached` above. Minted Subs live on terminus Profiles, never on the discovery
        // Profile (termini sit at depth ≥ 1 and chain configs hash-fork from non-chain), so the
        // cascade cannot perturb the Sub count read below.
        self.reap_orphans(removed.orphans, out);

        // A live Sub's Profile is live by the attach invariant; the guard is defence in depth.
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        if !self.subs.at(profile_id).is_empty() {
            // Every Sub on a Profile shares its `events` mask (it folds into `config_hash`), so
            // only the settle floor can move.
            self.recompute_profile_settle(profile_id);
            return;
        }

        let lifecycle = self
            .profiles
            .get(profile_id)
            .map(|p| p.state().detach_lifecycle());
        match lifecycle {
            Some(DetachLifecycle::ReapNow) => {
                self.reap_profile(profile_id, ReapTrigger::Immediate, out);
            }
            Some(DetachLifecycle::DeferToBurstEnd) => {
                // `recompute_profile_settle` is deliberately skipped — the Profile is about to drop
                // on burst end, so the recompute would be wasted. A revival via fresh
                // `attach_sub_inner` (zombie-revival branch) un-defers the reap and runs the
                // recompute symmetrically.
                let marked = self
                    .profiles
                    .get_mut(profile_id)
                    .is_some_and(specter_core::Profile::mark_active_for_reap);
                debug_assert!(
                    marked,
                    "detach_sub_inner: DetachLifecycle::DeferToBurstEnd requires \
                     ProfileState::Active(_, _) (profile = {profile_id:?})",
                );
            }
            None => {}
        }
    }

    /// In-place per-Sub rebind — the `modified_params` arm of [`Self::on_config_diff`]. Replaces
    /// `sub`'s spawn spec (`program` / `scope` / `log_output`) and `settle` via the engine-internal
    /// edge method [`SubRegistry::rebind`]; preserves `SubId`, `profile`, `name`, `minted_by`, and
    /// the fire history.
    ///
    /// Touches no Profile state, no Tree slot, and no kernel watch: the silent biggest win over the
    /// `modified_identity` detach+attach path is that the anchor's `watch_demand` stays installed
    /// throughout the rebind — there is no brief "no watcher" window.
    ///
    /// Bookkeeping is one tiny ledger update: when `params.settle` differs from the prior settle,
    /// the Profile's `settle` floor (min over its live Subs) is recomputed via
    /// [`Self::recompute_profile_settle`]. The currently-armed settle timer (if any) keeps its
    /// existing deadline; the *next* settle window uses the new floor.
    ///
    /// On a stale [`SubId`] (the rebind invariant says the dispatcher just resolved through
    /// `find_by_name`, so this is structurally unexpected), emits [`Diagnostic::RebindUnknownSub`]
    /// and returns without mutating engine state.
    pub(crate) fn rebind_sub_inner(
        &mut self,
        sub: SubId,
        new_params: SubParams,
        out: &mut StepOutput,
    ) {
        let new_settle = new_params.settle;
        let Some((prior_settle, profile)) = self.subs.rebind(sub, new_params) else {
            out.diagnostics.push(Diagnostic::RebindUnknownSub { sub });
            return;
        };
        if prior_settle != new_settle {
            self.recompute_profile_settle(profile);
        }
        out.diagnostics.push(Diagnostic::SubRebound { sub });
    }

    /// Reap a Profile: release every contribution it holds (anchor watch, watch-root parent watch,
    /// descent prefix watch, per-descendant watches), detach from `ProfileMap`, try-reap the anchor
    /// Resource, and emit a [`Diagnostic::ProfileReaped`] carrying the [`ReapTrigger`] that drove
    /// this reap. The trigger is supplied by the caller (not derived from state) because the two
    /// paths reach `reap_profile` with structurally distinct preconditions: `Immediate` from
    /// `detach_sub_inner` on Idle/Pending (no burst), and `DeferredFromBurst` from
    /// `finish_burst_to_idle` honouring a prior [`BurstFinish::Reap`] (the burst's drain has just
    /// run).
    ///
    /// **Quartet.** A Profile may hold up to four kinds of contribution to per-Resource
    /// `watch_demand`:
    ///
    ///   - **Anchor** (1-to-1): `Profile.resource.watch_demand` carries `+1` from this Profile
    ///     while `Profile.anchor_claim == AnchorClaim::Held`.
    ///   - **Watch-root parent** (1-to-1): `Profile.watch_root_parent`'s resource carries `+1`
    ///     `STRUCTURE` for anchor-reappearance detection.
    ///   - **Descent prefix** (1-to-1): the deepest existing prefix on a Pending Profile's path
    ///     carries `+1` `STRUCTURE`.
    ///   - **Per-descendant** (1-to-N): every covered Tree slot in `Profile.current` carries `+1`
    ///     (Dir always; Leaf under `has_per_file_fds`). The 1-to-N source-of-truth is the snapshot
    ///     itself, not a per-Profile flag.
    ///
    /// **Trichotomy invariant** (preserved from prior shape, now within the quartet). Anchor and
    /// descent-prefix are mutually exclusive at any moment: either the Profile is `Pending`
    /// (descent prefix only) or materialized (anchor + descendants + watch-root parent). The clamp
    /// recovery path (`Input::WatchOpRejected`) leaves the Profile with no contributions; the purge
    /// fan-out cleans up the bookkeeping.
    ///
    /// # Partial order
    ///
    /// `reap_profile`'s steps partition into five strictly-ordered groups where group N must
    /// complete before group N+1 begins. Within groups (1) and (4) members are unordered — any
    /// permutation produces an equivalent [`StepOutput`]:
    ///
    /// 1. **Probe disarm.** [`Engine::cancel_owner_probe`] emits a `ProbeOp::Cancel` and disarms the
    ///    owner's slot for any in-flight probe (a Pending Profile's descent probe; an Active burst
    ///    never reaches this entry with an armed slot — every path that drives a deferred reap from
    ///    Active disarms first: response dispatch via `take_owner_probe`, or the overflow-reseed reap
    ///    branch via `cancel_owner_probe`, both *before* `finish_burst_to_idle`). Must precede (2)
    ///    because [`Engine::release_descent_prefix_claim`]'s `transition_state(Idle)` *drops* the
    ///    prior `Pending(DescentState)`: an armed descent slot reaching that discard trips
    ///    `ProbeSlot`'s Drop tripwire (the cancel-first contract, now structurally enforced; see the
    ///    helper's rustdoc).
    ///
    /// 2. **Release quartet** — `release_descent_prefix_claim`, `release_descendant_claim`,
    ///    `release_anchor_claim`, `release_watch_root_parent_claim`. Each is idempotent,
    ///    counter-aware, and safe on a post-vacate slot. None reads or mutates state another might
    ///    also touch:
    ///    - `release_descendant_claim` `take()`s `Profile.current`; the other three never read it.
    ///    - `release_anchor_claim` removes `ProfileAnchor(pid)` from the anchor's contributions;
    ///      the other three never read the anchor's contribution map.
    ///    - `release_watch_root_parent_claim` calls `try_reap` on the *parent* slot (no-op while
    ///      the anchor is still a child); the anchor's eventual `try_reap` in (5) cascades upward
    ///      and reaps the parent then.
    ///    - `release_descent_prefix_claim` releases the descent-prefix contribution and transitions
    ///      `Pending → Idle`; mutually exclusive with anchor by the trichotomy invariant.
    ///
    ///    Cardinality order chosen for readability: 1-to-1 prefixed claims first, the 1-to-N
    ///    descendant walk, then the remaining 1-to-1 claims. Any of 4! = 24 permutations is correct.
    ///
    /// 3. **`ProfileMap::detach`.** Must follow (2) — the release helpers read `&Profile` (anchor,
    ///    watch_root_parent, snapshot). Must precede (4) — `Tree::try_reap` only reaps the anchor
    ///    slot once this Profile no longer occupies it.
    ///
    /// 4. **Anchor try-reap.** `Tree::try_reap(anchor)` cascades upward through any now-orphaned
    ///    ancestors (the watch-root parent slot is freed in this step if the anchor was its sole
    ///    remaining child). No parent-edge recompute accompanies it: the `Draining → Verifying`
    ///    reconfirm is a fresh `coverage` query, so a reaped Profile leaves no cached ancestor edge
    ///    for dependents to have rewritten — they re-derive on their next query against the
    ///    post-detach topology.
    ///
    /// 5. **`Diagnostic::ProfileReaped` emit.** Last by convention so the diagnostic ordering
    ///    across a step (which is sorted by emission site rather than by the
    ///    [`StepOutput::sort_for_emission`] pass) reads "do the work, then announce it."
    ///
    /// **Note on `discard_anchor_state` overlap.** This helper performs `release_descendant_claim`
    /// and `release_anchor_claim` inline rather than via [`Engine::discard_anchor_state`]. The two
    /// helpers differ in purpose:
    ///
    /// - `discard_anchor_state` exists for the "anchor lost, Profile lives" case — reached solely
    ///   through the `finalize_anchor_lost` coordinator (the funnel for every observed-loss route,
    ///   including the six probe vanished/failed dispatches). Its `kind = None` and `baseline =
    ///   None` writes prepare the Profile for the next Seed burst's probe-shape dispatch, and it
    ///   deliberately preserves `watch_root_parent` (the recovery channel).
    /// - `reap_profile` is "Profile dies entirely." There is no next Seed burst — the Profile
    ///   detaches in group (3) — so the `kind` and `baseline` writes that `discard_anchor_state`
    ///   would perform are wasted on a struct about to drop. Reap also releases
    ///   `watch_root_parent`, which `discard_anchor_state` deliberately preserves.
    ///
    /// The structural overlap (both call `release_descendant_claim + release_anchor_claim`) is
    /// intentional; the field clears and `watch_root_parent` release are deliberately partitioned
    /// across the two helpers.
    ///
    /// Sole call sites: `detach_sub_inner` (Idle / Pending Profile, immediate reap; `via =
    /// Immediate`) and `finish_burst_to_idle` (deferred reap when [`BurstFinish::Reap`] was set
    /// mid-burst; `via = DeferredFromBurst`).
    pub(crate) fn reap_profile(
        &mut self,
        profile_id: ProfileId,
        via: ReapTrigger,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let anchor = p.resource();

        // Trichotomy invariant: Pending and AnchorClaim::Held are mutually exclusive. The descent's
        // terminus flips Pending → Idle before the tail installs the claim, in the same step.
        debug_assert!(
            !matches!(
                (p.state(), p.anchor_claim()),
                (ProfileState::Pending(_), AnchorClaim::Held),
            ),
            "reap_profile: Pending + AnchorClaim::Held must be mutually exclusive",
        );

        // Consume any in-flight probe BEFORE the descent-prefix helper transitions the Profile to
        // Idle. Idempotent: emits Cancel iff a probe was in flight (Pending with a descent probe in
        // flight for this call path; an Active burst never reaches `reap_profile`'s entry armed —
        // `finish_burst_to_idle` runs `reap_profile` only after the slot was consumed, either by the
        // burst response or by the overflow-reseed reap branch). Mirrors `on_watch_op_rejected`'s
        // descent-purge pattern. A missed disarm is not silently tolerated: the armed slot would
        // reach `release_descent_prefix_claim`'s state discard (or `profiles.detach`) and trip
        // `ProbeSlot`'s Drop tripwire in every build — the discard *is* the enforcement.
        self.cancel_owner_probe(profile_id, out);

        // Reap is the only "done with this owner forever" edge; drop its `DispatchLedger` high-water
        // so the debug-only `BTreeMap` doesn't grow with the cumulative count of ever-attached
        // Profiles. Correctness-preserving (the next attach at the same SlotMap slot bumps the
        // generation, producing a distinct `ProfileId`); release-only the call compiles out.
        #[cfg(debug_assertions)]
        self.dispatch_ledger.forget(profile_id);

        // Release quartet — group (2) of the partial order (see rustdoc). Members are mutually
        // independent; the four helpers touch disjoint state. Code order chosen for readability
        // (1-to-1 prefixed claims first, then the 1-to-N descendant walk, then the remaining 1-to-1
        // claims); any permutation is equally correct.
        self.release_descent_prefix_claim(profile_id, ParkNarration::Teardown, out);
        self.release_descendant_claim(profile_id, out);
        self.release_anchor_claim(profile_id, out);
        self.release_watch_root_parent_claim(profile_id, out);

        // Detach the Profile from the registry. No parent-edge cache to clean: the `Draining →
        // Verifying` reconfirm is a fresh query (`coverage::nearest_covering_ancestor`), so a
        // dependent resolving its covering ancestor re-derives against the post-detach topology on
        // its next query — there is no stored edge to rewrite, hence no post-detach fixup pass. The
        // returned `Option<Profile>` carries the detached payload for diagnostic use only at the
        // inline site; `_detached` names the discard so a reader doesn't have to chase whether a
        // field was needed.
        let _detached = self.profiles.detach(&mut self.tree, profile_id);

        // Try to reap the anchor's slot. No-op if it still has children (a descendant Profile /
        // scaffold survives here), other Profiles attached at the same slot, or any co-resident
        // contribution. On success, [`Tree::try_reap`] cascades upward through any now-orphaned
        // ancestors — the watch-root parent slot whose only remaining claim was *this* Profile's
        // anchor as its sole child is freed in the same step. `try_reap` folds in `Tree::vacate` as
        // its closing-emission step, so any residual kernel-watch protocol (the closing `Unwatch`)
        // is emitted before the slot leaves the Tree.
        self.tree.try_reap(anchor, out);

        out.diagnostics.push(Diagnostic::ProfileReaped {
            profile: profile_id,
            via,
        });
    }

    /// Recompute `Profile.settle = min(remaining_subs.settles)` after a Sub addition or removal.
    /// O(subs-on-profile), bounded — typically 1–2 in v1 because `max_settle` already partitions
    /// Profiles.
    ///
    /// Uses `.map(...).expect(...)` rather than a defensive `.filter_map(...)`:
    /// [`SubRegistry::insert`] and [`SubRegistry::remove`] keep the `by_profile` index and the
    /// slotmap in lockstep, so every id returned by [`SubRegistry::at`] is live in
    /// [`SubRegistry::get`] by construction. A `None` here would be a structural invariant breach;
    /// the `expect` is the surface that names it.
    pub(crate) fn recompute_profile_settle(&mut self, profile_id: ProfileId) {
        let new_min: Option<Duration> = self
            .subs
            .at(profile_id)
            .iter()
            .map(|sid| {
                self.subs
                    .get(*sid)
                    .expect("by_profile index keeps SubIds live for SubRegistry::at")
                    .settle
            })
            .min();
        if let (Some(s), Some(p)) = (new_min, self.profiles.get_mut(profile_id)) {
            p.settle = s;
        }
    }

    /// Read-only view of the Engine's `Tree`.
    ///
    /// The bin uses this to inspect Resource topology; tests use it for setup verification.
    #[must_use]
    pub const fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Mutable access for path-to-`ResourceId` materialization.
    ///
    /// The bin uses this at startup to walk a config's `path` strings into the Tree before calling
    /// `attach_sub`. Use the dedicated refcount helpers to modify `watch_demand` — direct mutation
    /// breaks the 0↔non-empty contributions-edge invariant.
    pub const fn tree_mut(&mut self) -> &mut Tree {
        &mut self.tree
    }

    /// Read-only view of the `ProfileMap`.
    ///
    /// For inspection only; state-machine mutations route through `step` and `attach_sub`.
    #[must_use]
    pub const fn profiles(&self) -> &ProfileMap {
        &self.profiles
    }

    /// Read-only view of the `SubRegistry`.
    #[must_use]
    pub const fn subs(&self) -> &SubRegistry {
        &self.subs
    }

    /// Earliest pending timer deadline, or `None` if no timers are armed.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        self.timers.peek_top().map(|e| e.deadline)
    }

    /// Pop the earliest expired-and-still-referenced timer. Stale entries (cancelled because the
    /// Profile's burst was reset) are silently dropped. The returned [`TimerEntry`] carries the
    /// owning profile, kind, and id together — the bin forwards it to [`Input::TimerExpired`]
    /// without any rediscovery.
    ///
    /// **Stale-drain bound (dev only).** A single call drains at most 32 stale entries before
    /// tripping a `debug_assert!`. Realistic per-call drain is single-digit — each Active Profile
    /// carries ~2 timer slots and orphans at most one per burst transition, and the bin's tick loop
    /// polls frequently enough that orphans collect incrementally. The bound is loose enough to
    /// absorb multi-Profile burst-end cleanup, tight enough to surface an "engine transition leaks
    /// timer references" regression. Release builds run unbounded (the lazy drain is correct either
    /// way; the bound is purely a developer-time invariant check).
    pub fn pop_expired(&mut self, now: Instant) -> Option<TimerEntry> {
        // Stale-drain accounting — see the rustdoc above for the bound's semantics. `u32` is more
        // than enough headroom even in a release build where the assert is compiled out.
        let mut stale_drops: u32 = 0;
        loop {
            let top = self.timers.peek_top()?;
            if top.deadline > now {
                return None;
            }
            let entry = self.timers.pop_top().expect("peek_top was Some");
            if is_timer_referenced(&self.profiles, entry.profile, entry.kind, entry.id) {
                return Some(entry);
            }
            // Stale — silently drop, continue draining.
            stale_drops += 1;
            debug_assert!(
                stale_drops <= STALE_DRAIN_BOUND,
                "pop_expired: drained {stale_drops} stale timers in a single call \
                 (bound = {STALE_DRAIN_BOUND}); a transition is leaking timer references. \
                 Last stale: kind = {:?}, profile = {:?}",
                entry.kind,
                entry.profile,
            );
        }
    }
}

/// Whether `id` is the live timer for `profile`'s `kind` slot — `pop_expired` uses this to filter
/// stale heap heads, and `on_timer_expired` re-runs it as defense-in-depth for direct
/// `step(Input::TimerExpired)` callers (tests, fuzzers).
///
/// Implemented as a thin lookup-and-compare against [`ProfileState::timer_token`]: every `(state,
/// kind)` pair routes to whichever burst-side type carries the field, with the type-impossible
/// pairs (e.g., `Settle` on `PostFire`) folding to `None` at the leaf without an explicit
/// fallthrough arm. Returns `false` for stale `profile` ids and for any state that doesn't own a
/// `kind` timer right now.
///
/// Free function rather than a method on [`Engine`]: the projection is purely a query over
/// [`ProfileMap`] and doesn't reach into the engine's other fields, so a free function keeps the
/// call-site shape (`is_timer_referenced(&self.profiles, …)`) honest about the dependency.
pub(crate) fn is_timer_referenced(
    profiles: &ProfileMap,
    profile: ProfileId,
    kind: TimerKind,
    id: TimerId,
) -> bool {
    profiles
        .get(profile)
        .and_then(|p| p.state().timer_token(kind))
        .is_some_and(|live| live == id)
}

/// Whether an attach or a mint found its Profile or founded it — read once, so no arm re-derives
/// "which Profile is this?" from state, where "no live Subs" cannot tell a first Sub from one
/// reviving a zombie and "no anchor claim" cannot tell a fresh Profile from a Pending one.
enum ProfileOrigin {
    /// Brand-new: it seeds at its anchor ([`Engine::seed_at_anchor`]), or descends to it first.
    Fresh,
    /// It already stood; the Sub joins it ([`Engine::join_profile`]).
    Existing(Join),
}

/// How a Profile that already stood takes a joining Sub ([`Engine::join_profile`]).
#[derive(Clone, Copy)]
enum Join {
    /// Its burst, descent or Idle baseline runs on.
    Live,
    /// [`ProfileState::Parked`] — anchorless, awaiting recovery: without a re-armed descent the new
    /// Sub would inherit the park and never fire, with a clean attach log.
    Parked,
    /// [`ProfileState::Active`] with [`BurstFinish::Reap`]: its last Sub left mid-burst, and the
    /// reap waits for the burst's end.
    Zombie,
}

/// Whether a live observation of the anchor's *own appearance* stands behind the anchor going live
/// — the parameter of the anchor-materialisation tail ([`Engine::seed_at_anchor`]). An attach never
/// has one (`Cold`); a mint reads it off its [`MintEntry`]; a descent's terminus off its
/// [`DescentWitness`].
///
/// Deliberately engine-internal, **not** a [`SubAttachRequest`] field: the request crosses channels
/// (config load, hot reload, IPC), and a fire-owing witness must not be claimable from the wire.
/// Payload-less: the witness names the anchor's own appearance, so the trigger derives from the
/// anchor and a witness disagreeing with its anchor is unrepresentable.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SeedWitness {
    Cold,
    Appeared,
}

impl From<MintEntry> for SeedWitness {
    /// A minted Profile's anchor was seen appearing exactly when its match appeared under watch.
    fn from(entry: MintEntry) -> Self {
        match entry {
            MintEntry::Enumerated => Self::Cold,
            MintEntry::Appeared => Self::Appeared,
        }
    }
}

impl From<DescentWitness> for SeedWitness {
    /// A descent saw its anchor appear only if it saw it absent first; an absence the terminus
    /// never followed with a presence is no appearance.
    fn from(witness: DescentWitness) -> Self {
        match witness {
            DescentWitness::Appeared => Self::Appeared,
            DescentWitness::Unobserved | DescentWitness::Absent => Self::Cold,
        }
    }
}

/// Outcome of `attach_sub_inner`'s Phase 1 anchor resolution. The two variants encode the semantic
/// divide between attaches whose anchor is already live on disk and those whose anchor is
/// scaffolded awaiting materialisation. The split drives Phase 3's `Fresh` arm: seed at the anchor
/// now, or descend to it first.
///
/// The `Immediate` arm subsumes both a fully-materialised [`SubAttachAnchor::Path`] and a
/// liveness-checked [`SubAttachAnchor::Resource`]; only `Path` can resolve `Pending`.
enum AnchorResolution {
    /// The anchor is a live, materialised Tree slot. A fresh Profile seeds at it at once
    /// ([`Engine::seed_at_anchor`]).
    Immediate { anchor: ResourceId },
    /// The anchor is a `DescentScaffold`-roled slot awaiting materialisation. `prefix` is the
    /// deepest existing ancestor (where the descent probe is currently watching) and `remaining`
    /// carries the path components from `prefix` (exclusive) down to `anchor` (inclusive). A fresh
    /// Profile enters `ProfileState::Pending(_)` and descends ([`Engine::enter_pending_descent`]).
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: DescentRemaining,
    },
}

impl AnchorResolution {
    /// The anchor slot — the Profile's resource on attach. Both variants expose the same field name
    /// so `find_or_create_profile` can key off it without matching on the variant.
    const fn anchor(&self) -> ResourceId {
        match self {
            Self::Immediate { anchor } | Self::Pending { anchor, .. } => *anchor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use specter_core::{
        ClassSet, DedupKey, EffectCompletion, EffectOutcome, FsEvent, Input, ProbeCorrelation,
        ProbeOutcome, ProbeResponse, ProfileId, ResourceId, StepOutput, SubId, SubRegistryDiff,
        SubtreeScan, TimerId, TimerKind, WatchOp,
    };
    use std::num::NonZeroU32;
    use std::time::{Duration, Instant};

    // Compile-time `Send + Sync` check on `Engine`. The bin loop parks `Engine` on its own thread;
    // `Send + Sync` is load-bearing for that.
    const _: fn() = || {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    };

    #[test]
    fn step_fs_event_for_unwatched_resource_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(
            Input::FsEvent {
                resource: ResourceId::default(),
                event: FsEvent::ContentChanged,
            },
            Instant::now(),
        );
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::EventOnUnwatchedResource { .. }));
        assert!(has_diag);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops().is_empty());
        assert!(out.effects().is_empty());
    }

    #[test]
    fn step_probe_response_unknown_profile_diagnoses() {
        let mut e = Engine::new();
        let resp = ProbeResponse {
            owner: ProfileId::default(),
            correlation: ProbeCorrelation::from(0),
            outcome: ProbeOutcome::Vanished,
        };
        let out = e.step(Input::ProbeResponse(resp), Instant::now());
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::StaleProbeResponse { .. }));
        assert!(has_diag);
    }

    #[test]
    fn step_timer_expired_stale_id_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(
            Input::TimerExpired {
                profile: ProfileId::default(),
                kind: TimerKind::Settle,
                id: TimerId::default(),
            },
            Instant::now(),
        );
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::StaleTimer { .. }));
        assert!(has_diag);
    }

    #[test]
    fn step_effect_complete_unknown_sub_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(
            Input::EffectComplete(EffectCompletion {
                sub: SubId::default(),
                key: DedupKey::default(),
                outcome: EffectOutcome::Ok,
            }),
            Instant::now(),
        );
        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::EffectCompleteForUnknownSub { .. }
            )
        });
        assert!(has_diag);
    }

    #[test]
    fn step_watch_op_rejected_emits_diagnostic_for_stale_resource() {
        // Stale ResourceId or already-Unwatched resource yields a Diagnostic + no other ops.
        let mut e = Engine::new();
        let out = e.step(
            Input::WatchOpRejected {
                resource: ResourceId::default(),
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops().is_empty());
        assert!(out.effects().is_empty());
        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::WatchOpRejected {
                    failure: specter_core::WatchFailure::Pressure { errno: 24 },
                    ..
                }
            )
        });
        assert!(has_diag);
    }

    #[test]
    fn step_config_diff_with_empty_diff_is_noop() {
        let mut e = Engine::new();
        let out = e.step(
            Input::ConfigDiff(SubRegistryDiff::default()),
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops().is_empty());
        assert!(out.effects().is_empty());
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn next_deadline_is_none_when_no_timers() {
        let e = Engine::new();
        assert!(e.next_deadline().is_none());
    }

    #[test]
    fn next_deadline_returns_top_after_schedule() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_millis(100);
        let _ = e
            .timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
        assert_eq!(e.next_deadline(), Some(when));
    }

    #[test]
    fn pop_expired_returns_none_when_top_in_future() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_secs(10);
        let _ = e
            .timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future-dated entries are not drained");
    }

    #[test]
    fn pop_expired_drains_stale_entries_silently() {
        // Schedule timers for null/unknown Profiles (no Active state holds them). The validating
        // drain consumes every stale entry, but returns None — there's nothing live to fire.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        let _ = e
            .timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        let _ = e
            .timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        let _ = e
            .timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 0, "stale heads were drained");
    }

    /// Stale-drain bound: a `pop_expired` call that drains more than [`STALE_DRAIN_BOUND`]
    /// consecutive stale entries trips the `debug_assert!`. Production code is structurally bounded
    /// (per-Profile orphan count is small); this test pins the regression sensor by deliberately
    /// exceeding the bound.
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "stale timers in a single call")]
    fn pop_expired_panics_on_excessive_stale_drain_in_debug() {
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        // Schedule one beyond the bound. All stale (default ProfileId has no Active state), so the
        // drain runs through them all.
        for _ in 0..=STALE_DRAIN_BOUND {
            let _ = e
                .timers
                .schedule(past, ProfileId::default(), TimerKind::Settle);
        }
        let _ = e.pop_expired(now);
    }

    #[test]
    fn pop_expired_stops_at_first_future_entry() {
        // Mix of expired-stale and future-dated. The drain consumes the stale expired heads, then
        // returns None when peeking a future-dated entry.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        let _ = e
            .timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        let _ = e.timers.schedule(
            now + Duration::from_secs(10),
            ProfileId::default(),
            TimerKind::Settle,
        );

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future entry remains");
        assert!(e.next_deadline().unwrap() > now);
    }

    #[test]
    fn mint_probe_correlation_is_monotone() {
        // Successive `mint_probe_correlation` calls yield strictly increasing, globally-unique
        // correlations drawn off the one engine-wide monotone floor — distinct ids in a single id
        // space (owners' slots hold their own correlation).
        let mut e = Engine::new();
        let a = e.mint_probe_correlation();
        let b = e.mint_probe_correlation();
        let c = e.mint_probe_correlation();
        assert!(a < b);
        assert!(b < c);
        assert_eq!(a, ProbeCorrelation::from(1));
        assert_eq!(b, ProbeCorrelation::from(2));
        assert_eq!(c, ProbeCorrelation::from(3));
    }

    /// Counter saturation on the probe side — release-runnable. Proves `mint_probe_correlation`
    /// routes through the panicking `MonotonicCounter::next` rather than reimplementing the bump,
    /// symmetric with `effect_correlations_panic_on_counter_saturation`. Pairs with the
    /// `MonotonicCounter` unit tests in `counter.rs`.
    #[test]
    #[should_panic(expected = "MonotonicCounter")]
    fn mint_probe_correlation_panics_on_counter_saturation() {
        let mut e = Engine::new();
        e.correlations.prime(u64::MAX);
        let _ = e.mint_probe_correlation();
    }

    #[test]
    fn engine_default_constructible_has_empty_state() {
        let e = Engine::new();
        assert!(e.tree.is_empty());
        assert!(e.profiles.is_empty());
        assert!(e.subs.is_empty());
        assert!(e.timers.is_empty());
        assert!(e.next_deadline().is_none());
        assert_eq!(e.correlations.peek(), 0);
        assert_eq!(e.effect_correlations.peek(), 0);
    }

    /// Counter saturation on the effect side — release-runnable. The effect counter has no
    /// per-call-site wrapper — the minting is inline at the two `emit_effects` push sites in
    /// `transitions.rs` — so this test exercises the counter directly to prove the field is wired
    /// up. Pairs with the `MonotonicCounter` unit tests in `counter.rs`.
    #[test]
    #[should_panic(expected = "MonotonicCounter")]
    fn effect_correlations_panic_on_counter_saturation() {
        let mut e = Engine::new();
        e.effect_correlations.prime(u64::MAX);
        let _ = e.effect_correlations.next();
    }

    // ===== Engine-level attach-rejection contract =====
    //
    #[test]
    fn attach_path_invalid_carries_offending_path() {
        let mut e = Engine::new();
        let bad = std::path::PathBuf::from("./relative/with/dot");
        let req = SubAttachRequest::for_anchor(
            "bad".into(),
            SubAttachAnchor::Path(bad.clone()),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());

        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::AttachPathInvalid { path, .. } if &**path == bad.as_path(),
            )
        });
        assert!(saw, "AttachPathInvalid must carry the offending path");
    }

    /// End-to-end gate enforcement: a relative-path attach request rolls up no `SubId`, no Tree
    /// slots, and no Profile — only the diagnostic surfaces. Pins the contract that
    /// `attach_sub_inner`'s `Tree::parse_attach_path` short-circuit is total: rejection produces
    /// `None` plus zero side-effects on engine state.
    #[test]
    fn attach_with_relative_path_emits_diagnostic_and_no_state() {
        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        let bad = std::path::PathBuf::from("relative/path");
        let req = SubAttachRequest::for_anchor(
            "rel".into(),
            SubAttachAnchor::Path(bad.clone()),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out);

        assert!(sid.is_none(), "rejected attach mints no SubId");
        assert_eq!(e.tree.len(), pre_tree_len, "no Tree slots created");
        assert_eq!(e.profiles.len(), pre_profile_count, "no Profile attached");
        assert!(e.subs.is_empty(), "no Sub recorded in registry");
        assert!(out.watch_ops.is_empty(), "no watch ops emitted");
        assert!(out.probe_ops().is_empty(), "no probe ops emitted");
        assert!(out.effects().is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if &**path == bad.as_path() && hint.contains("absolute"),
        )));
    }

    /// End-to-end counterpart for non-UTF-8 paths. The test fabricates a path with a non-UTF-8
    /// segment via `OsStr::from_bytes` (Unix-only) and confirms the same total-rejection contract:
    /// no SubId, no Tree slots, no Profile.
    #[cfg(unix)]
    #[test]
    fn attach_with_non_utf8_path_emits_diagnostic_and_no_state() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::path::PathBuf;

        let bad_seg = OsStr::from_bytes(&[0xFF, 0xFE]);
        let mut path = PathBuf::from("/foo");
        path.push(bad_seg);

        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        let req = SubAttachRequest::for_anchor(
            "bad".into(),
            SubAttachAnchor::Path(path.clone()),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out);

        assert!(sid.is_none(), "rejected attach mints no SubId");
        assert_eq!(e.tree.len(), pre_tree_len, "no Tree slots created");
        assert_eq!(e.profiles.len(), pre_profile_count, "no Profile attached");
        assert!(e.subs.is_empty(), "no Sub recorded in registry");
        assert!(out.watch_ops.is_empty(), "no watch ops emitted");
        assert!(out.probe_ops().is_empty(), "no probe ops emitted");
        assert!(out.effects().is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path: p, hint }
                if &**p == path.as_path() && hint.contains("non-UTF-8"),
        )));
    }

    /// A `Resource` anchor naming a dead slot is re-gated at the boundary — graceful
    /// `AttachResourceStale` + `None`, no panic, zero engine-state mutation. Symmetric with the
    /// `Path`-arm total-rejection tests above.
    #[test]
    fn attach_with_stale_resource_emits_diagnostic_and_no_state() {
        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        // `slotmap::new_key_type!` never mints the default key, so `ResourceId::default()` is
        // permanently stale.
        let stale = ResourceId::default();
        let req = SubAttachRequest::for_anchor(
            "stale".into(),
            SubAttachAnchor::Resource(stale),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out);

        assert!(sid.is_none(), "rejected attach mints no SubId");
        assert_eq!(e.tree.len(), pre_tree_len, "no Tree slots created");
        assert_eq!(e.profiles.len(), pre_profile_count, "no Profile attached");
        assert!(e.subs.is_empty(), "no Sub recorded in registry");
        assert!(out.watch_ops.is_empty(), "no watch ops emitted");
        assert!(out.probe_ops().is_empty(), "no probe ops emitted");
        assert!(out.effects().is_empty(), "no effects emitted");

        let stale_count = out
            .diagnostics
            .iter()
            .filter(|d| {
                matches!(
                    d,
                    specter_core::Diagnostic::AttachResourceStale { resource }
                        if *resource == stale,
                )
            })
            .count();
        assert_eq!(
            stale_count, 1,
            "exactly one AttachResourceStale carrying the offending id; got {:?}",
            out.diagnostics,
        );
    }

    #[test]
    fn detach_unknown_sub_emits_dedicated_diagnostic() {
        let mut e = Engine::new();
        let bogus = SubId::default();
        let out = e.step(Input::DetachSub(bogus), Instant::now());

        let saw_dedicated = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::DetachUnknownSub { sub } if *sub == bogus,
            )
        });
        let saw_wrong = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::EffectCompleteForUnknownSub { .. },
            )
        });
        assert!(saw_dedicated, "detach miss must emit DetachUnknownSub");
        assert!(
            !saw_wrong,
            "detach miss must NOT emit EffectCompleteForUnknownSub",
        );
    }

    // ===== Zombie revival =====

    fn revival_attach_req(anchor: ResourceId, name: &str, settle: Duration) -> SubAttachRequest {
        SubAttachRequest::for_anchor(
            name.into(),
            SubAttachAnchor::Resource(anchor),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_secs(6),
            settle,
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            ClassSet::EMPTY,
            false,
        )
    }

    #[test]
    fn attach_revives_reap_pending_profile() {
        // Detach A's Sub mid-Active to flip the Active burst's directive to [`BurstFinish::Reap`],
        // then re-attach B at the same `(anchor, config_hash)`. The revival path must:
        //   - reuse A's Profile (same ProfileId),
        //   - leave the anchor's watch_demand at 1 (no double-bump),
        //   - emit no spurious Watch op for the anchor,
        //   - flip the directive back to `BurstFinish::ReturnToIdle` via `clear_active_reap`,
        //   - keep `anchor_claim` Held,
        //   - recompute `Profile.settle` to B's settle (NOT min-update — A is gone, B is the only
        //     live Sub),
        //   - emit `Diagnostic::ReapPendingCancelled`.
        //
        // (No fire-history cleanup is asserted because there is none to assert: the history is
        // per-Sub (`FireHistory` on the Spawn reaction) and dies with the slotmap entry on detach.
        // A revived Profile's freshly-attached Subs start unfired structurally — there is no
        // per-Profile fire container to purge.)
        let mut e = Engine::new();
        let r = e
            .tree
            .ensure_root("anchor", specter_core::ResourceRole::User);
        e.tree.set_kind(r, specter_core::ResourceKind::Dir);
        let now = Instant::now();

        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "A", Duration::from_millis(50))),
            now,
        );
        let sid_a =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid_a).unwrap().profile();
        let watch_demand_after_attach = e.tree.get(r).unwrap().watch_demand();
        assert_eq!(watch_demand_after_attach, 1, "anchor watch_demand from A");

        // Detach A. Profile is Active → directive flips to `BurstFinish::Reap`; anchor watch
        // unchanged.
        let _ = e.step(Input::DetachSub(sid_a), Instant::now());
        assert!(matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap),
        ));
        assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);

        // Revive with B (settle=200ms; deliberately larger than A's stale 50ms so the min-update
        // would be visibly wrong).
        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "B", Duration::from_millis(200))),
            now,
        );
        let sid_b =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid_b = e.subs().get(sid_b).unwrap().profile();

        assert_eq!(pid_b, pid, "B reuses A's Profile");
        assert_eq!(
            e.tree.get(r).unwrap().watch_demand(),
            1,
            "anchor watch_demand unchanged on revival (no double-bump)",
        );
        let p = e.profiles().get(pid).unwrap();
        assert!(
            !matches!(p.state().burst_finish(), Some(BurstFinish::Reap)),
            "BurstFinish flipped back to ReturnToIdle",
        );
        assert_eq!(
            p.anchor_claim(),
            AnchorClaim::Held,
            "anchor_claim stays Held"
        );
        assert_eq!(
            p.settle,
            Duration::from_millis(200),
            "settle recomputed to B's (only live Sub) — min-update would yield 50ms",
        );
        let anchor_watch_ops: Vec<_> = attach_out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Watch { resource, .. } if *resource == r))
            .collect();
        assert!(
            anchor_watch_ops.is_empty(),
            "no spurious Watch op for the anchor on revival; got {anchor_watch_ops:?}",
        );
        assert!(
            attach_out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::ReapPendingCancelled { profile } if *profile == pid,
            )),
            "ReapPendingCancelled emitted on revival",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn finish_burst_to_idle_does_not_reap_revived_profile() {
        // After revival, the in-flight burst's lifecycle continues under the new Sub set. When the
        // probe responds and the burst ends, `finish_burst_to_idle` must NOT call `reap_profile`
        // (the revival cleared `reap_pending`).
        let mut e = Engine::new();
        let r = e
            .tree
            .ensure_root("anchor", specter_core::ResourceRole::User);
        e.tree.set_kind(r, specter_core::ResourceKind::Dir);
        let now = Instant::now();

        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "A", Duration::from_millis(50))),
            now,
        );
        let sid_a =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid_a).unwrap().profile();
        // Batching-first Seed: expire the settle timer (settle = 50ms) so the verify probe is in
        // flight before the revival below.
        let t_settle = now + Duration::from_millis(50);
        while let Some(entry) = e.pop_expired(t_settle) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t_settle,
            );
        }
        let seed_corr = e
            .pending_probe_for(pid)
            .expect("Seed verify probe in flight after settle expiry");

        let _ = e.step(Input::DetachSub(sid_a), Instant::now());
        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "B", Duration::from_millis(50))),
            now,
        );
        let sid_b =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");

        // Drive the in-flight Seed-Verifying burst to a terminal Vanished.
        // `dispatch_pre_fire_vanished → finalize_anchor_lost → finish_burst_to_idle` would reap if
        // `reap_pending` were still set; the revival cleared it, so the Profile transitions to Idle
        // (anchor lost) and stays.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: seed_corr,
                outcome: ProbeOutcome::Vanished,
            }),
            now,
        );

        assert!(
            e.profiles().get(pid).is_some(),
            "Profile alive (revival pre-empted reap)",
        );
        assert!(e.subs().get(sid_b).is_some(), "B still attached");
        assert!(
            !out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::ProfileReaped { profile, via: _ } if *profile == pid,
            )),
            "ProfileReaped must NOT emit for a revived Profile",
        );
    }

    // ===== reap_profile release-quartet permutation =====

    /// Snapshot of "claim state" after running a permutation of the release quartet. The fields are
    /// exactly the bookkeeping the four release helpers clear (per-Profile flags + per-Resource
    /// `watch_demand` counters); two runs that produce the same snapshot have produced equivalent
    /// observable effects on engine state. `StepOutput.watch_ops` is also compared by the caller
    /// because the helpers emit `Unwatch` operations whose order is part of the
    /// post-`sort_for_emission` contract.
    #[derive(Debug, Eq, PartialEq)]
    struct QuartetFinalState {
        anchor_claim: AnchorClaim,
        watch_root_parent: Option<ResourceId>,
        current_is_none: bool,
        anchor_watch_demand: u32,
        parent_watch_demand: u32,
    }

    /// Build a "materialised" Profile carrying three of the four quartet claims (anchor,
    /// watch_root_parent, descendants — the descent-prefix claim is mutually exclusive with anchor
    /// per the trichotomy invariant). Returns the IDs the test body needs to inspect the
    /// post-release state.
    ///
    /// The fixture uses `attach_sub_inner` to bootstrap so every per-Resource contribution lands
    /// through the canonical `add_watch` path rather than a hand-rolled mutation that would not
    /// exercise refcount discipline. The Dir snapshot installed onto `Profile.current` after attach
    /// gives `release_descendant_claim` something to release.
    fn build_materialised_profile_for_permutation() -> (Engine, ProfileId, ResourceId, ResourceId) {
        use specter_core::{
            ChildEntry, ClassSet, DirMeta, DirSnapshot, FsIdentity, LeafEntry, ResourceKind,
            ResourceRole,
        };
        use std::collections::BTreeMap;
        use std::sync::Arc;
        use std::time::UNIX_EPOCH;

        let mut e = Engine::new();
        let parent = e.tree.ensure_root("parent", ResourceRole::User);
        let anchor = e
            .tree
            .ensure_child(parent, "anchor", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(parent, ResourceKind::Dir);
        e.tree.set_kind(anchor, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "permtest".into(),
            SubAttachAnchor::Resource(anchor),
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            Duration::from_secs(6),
            Duration::from_millis(50),
            specter_core::testkit::single_exec_program(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            ClassSet::EMPTY,
            false,
        );
        let now = Instant::now();
        let out = e.step(Input::AttachSub(req), now);
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach succeeded");
        let pid = e.subs().get(sid).expect("Sub alive").profile();

        // Install a Dir snapshot on Profile.current so the descendant release has a snapshot to
        // take. Mirrors what `dispatch_quiescence_ok` would write at probe completion. One child
        // leaf gives `release_descendant_claim` a non-trivial diff to apply.
        let child_id = e
            .tree
            .ensure_child(anchor, "child", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(child_id, ResourceKind::File);
        let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        entries.insert(
            CompactString::from("child"),
            ChildEntry::Leaf(LeafEntry::synthetic(
                specter_core::EntryKind::File,
                0,
                UNIX_EPOCH,
                FsIdentity::synthetic(0, 0),
            )),
        );
        let dir = DirSnapshot::new(
            DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
            0,
            entries,
        );
        e.profiles
            .get_mut(pid)
            .expect("Profile alive")
            .install_dir_current(Arc::new(dir));

        // Drop the auto-emitted Watch operations from attach — the permutation test only cares
        // about deltas across release calls.
        let _ = out;

        (e, pid, anchor, parent)
    }

    /// Run the release quartet in a given index permutation and capture the post-release state.
    fn run_quartet_permutation(perm: [usize; 4]) -> QuartetFinalState {
        let (mut e, pid, anchor, parent) = build_materialised_profile_for_permutation();
        let releases: [fn(&mut Engine, ProfileId, &mut StepOutput); 4] = [
            // The reap-shaped quartet: the descent release rides the teardown narration exactly as
            // `reap_profile` invokes it.
            |e, pid, out| e.release_descent_prefix_claim(pid, ParkNarration::Teardown, out),
            Engine::release_descendant_claim,
            Engine::release_anchor_claim,
            Engine::release_watch_root_parent_claim,
        ];
        let mut out = StepOutput::default();
        for &idx in &perm {
            releases[idx](&mut e, pid, &mut out);
        }

        let final_state = QuartetFinalState {
            anchor_claim: e
                .profiles
                .get(pid)
                .map_or(AnchorClaim::None, specter_core::Profile::anchor_claim),
            watch_root_parent: e
                .profiles
                .get(pid)
                .and_then(specter_core::Profile::watch_root_parent),
            current_is_none: e.profiles.get(pid).is_none_or(|p| p.current().is_none()),
            anchor_watch_demand: e
                .tree
                .get(anchor)
                .map_or(0, specter_core::Resource::watch_demand),
            parent_watch_demand: e
                .tree
                .get(parent)
                .map_or(0, specter_core::Resource::watch_demand),
        };
        // The attach-time Seed-Verifying probe is still armed; the release quartet never consumes
        // it. Drain before `e` drops.
        let _ = e.cancel_all_in_flight_probes();
        final_state
    }

    /// Enumerate every permutation of `[0, 1, 2, 3]` via Heap's algorithm. 24 results.
    fn all_quartet_permutations() -> Vec<[usize; 4]> {
        fn heaps(arr: &mut [usize; 4], k: usize, out: &mut Vec<[usize; 4]>) {
            if k == 1 {
                out.push(*arr);
                return;
            }
            for i in 0..k {
                heaps(arr, k - 1, out);
                if k.is_multiple_of(2) {
                    arr.swap(i, k - 1);
                } else {
                    arr.swap(0, k - 1);
                }
            }
        }
        let mut buf = [0_usize, 1, 2, 3];
        let mut out = Vec::with_capacity(24);
        heaps(&mut buf, 4, &mut out);
        out
    }

    /// Pin the partial-order claim in `reap_profile`'s rustdoc: every permutation of the four release
    /// helpers yields the same observable claim state. The fixture exercises three live claims
    /// (anchor + watch_root_parent + descendants); the fourth helper (`release_descent_prefix_claim`)
    /// no-ops on the non-Pending Profile but still occupies a position in the permutation.
    #[test]
    fn release_quartet_is_permutation_invariant() {
        let perms = all_quartet_permutations();
        assert_eq!(perms.len(), 24, "Heap's algorithm enumerates 4! = 24");

        let canonical = run_quartet_permutation([0, 1, 2, 3]);
        // Final state pins the helpers actually did their work: anchor released, watch_root_parent
        // released, snapshot taken, both resource watch_demand counters at zero.
        assert_eq!(canonical.anchor_claim, AnchorClaim::None);
        assert_eq!(canonical.watch_root_parent, None);
        assert!(canonical.current_is_none);
        assert_eq!(canonical.anchor_watch_demand, 0);
        assert_eq!(canonical.parent_watch_demand, 0);

        for perm in perms {
            let observed = run_quartet_permutation(perm);
            assert_eq!(
                observed, canonical,
                "permutation {perm:?} produced state diverging from canonical [0,1,2,3]",
            );
        }
    }
}
