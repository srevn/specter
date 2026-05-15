//! `Engine` — pure, deterministic, total.
//!
//! The engine owns the data model (`Tree`, `ProfileMap`, `SubRegistry`),
//! the timer wheel, and the stability index; `step` consumes one [`Input`]
//! at a time and emits a sorted [`StepOutput`]. State-machine bodies live
//! in sibling modules:
//! - `burst.rs` — Idle ↔ Active phase transitions.
//! - `transitions.rs` — per-input handlers (`on_fs_event`, etc.).
//! - `reconcile.rs` — newly-discovered descendants.
//! - `refcounts.rs` — `watch_demand` / `suppress_count` edges.
//!
//! `step` is the single dispatch point; each `Input` variant routes to the
//! corresponding `on_*` handler. `attach_sub` is the engine's public
//! Sub-attachment API.

use crate::counter::MonotonicCounter;
use crate::refcounts::add_watch;
use crate::timer::{TimerEntry, TimerHeap};
use compact_str::CompactString;
// Identity.
use specter_core::{ProfileId, ResourceId, SubId, TimerId};
// Tree + path validation.
use specter_core::Tree;
// Profile state machine.
use specter_core::{
    AnchorClaim, BurstFinish, DescentRemaining, DescentState, DetachLifecycle, Profile, ProfileMap,
    ProfileState, ReapTrigger, TimerKind,
};
// Registries.
use specter_core::{PromoterRegistry, Sub, SubAttachRequest, SubRegistry};
// Per-Resource bookkeeping.
use specter_core::{ClassSet, ContribKey};
// Probe + effect correlation.
use specter_core::{CorrelationId, DedupKey, ProbeOwner};
// Engine step I/O.
use specter_core::{Diagnostic, Input, StepOutput};
// Helpers.
use specter_core::compute_config_hash;
use std::time::{Duration, Instant};

/// Per-call stale-drain bound for [`Engine::pop_expired`].
///
/// Realistic worst case per call is single-digit (each Active Profile
/// carries ~2 timer slots and orphans at most one per burst
/// transition; the bin's tick loop polls frequently enough that
/// orphans collect incrementally rather than piling at the heap top).
/// The bound is loose enough to absorb multi-Profile burst-end
/// cleanup yet tight enough to surface an "engine transition leaks
/// timer references" regression in dev/CI.
const STALE_DRAIN_BOUND: u32 = 32;

/// `pub(crate)` field visibility lets sibling modules read/write engine
/// state directly. External consumers go through the public methods.
///
/// Per-owner descent state lives inline on the owner's state enum
/// (`ProfileState::Pending(DescentState)` for Profiles,
/// `PromoterState::PrefixPending(DescentState)` for Promoters). Read through
/// the owner-polymorphic `Engine::descent_state` / `Engine::descent_state_mut`
/// (both `pub(crate)`); per-event fan-out lives next to its sole consumer
/// (`Engine::classify_event_carriers` in `transitions.rs`).
#[derive(Debug, Default)]
pub struct Engine {
    pub(crate) tree: Tree,
    pub(crate) profiles: ProfileMap,
    pub(crate) subs: SubRegistry,
    /// Engine-resident dynamic-watch sources. Promoter-side
    /// contributions to per-Resource watch demand are tracked in the
    /// per-Resource [`specter_core::Resource::contributions`] map via
    /// [`specter_core::ContribKey::PromoterPrefix`] /
    /// [`specter_core::ContribKey::PromoterProxy`]; this field is the
    /// registry the Promoter helpers mutate.
    pub(crate) promoters: PromoterRegistry,
    pub(crate) timers: TimerHeap,
    /// Probe-channel state. Owns the per-owner outstanding-probe map
    /// and the [`ProbeCorrelation`] monotonic counter. See
    /// [`crate::probe_channel::ProbeChannel`] for the channel's
    /// invariants and API.
    pub(crate) probe_channel: crate::probe_channel::ProbeChannel,
    /// Monotonic counter for [`CorrelationId`] minting. Bumped at
    /// every `Effect` push in `transitions.rs::emit_effects` so the
    /// actuator-side coalescer can correlate completions back to the
    /// originating Effect across the Latest dedup. Phantom-typed
    /// distinct from the channel's probe-side counter: the two id
    /// spaces stay structurally separate at the type level.
    pub(crate) effect_correlations: MonotonicCounter<CorrelationId>,
    /// Reusable scratch buffer for parent-edge recomputation. The
    /// `collect_in_subtree` / `collect_pointing_at` producers in
    /// [`crate::stability`] fill this from an immutable
    /// `&ProfileMap` borrow; [`crate::stability::recompute_parent_edges`]
    /// then drains it under the `&mut ProfileMap` borrow. Both
    /// producers `clear()` on entry — the buffer survives across
    /// `step` calls but never accumulates state between them. Sole
    /// rationale for the field: dodge the borrow-checker conflict
    /// without paying a per-call `Vec` allocation.
    pub(crate) scratch_profile_ids: Vec<ProfileId>,
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Owner-polymorphic descent state accessor. Returns the `DescentState`
    /// payload of the owner's "in-descent" state variant
    /// (`ProfileState::Pending` for Profiles, `PromoterState::PrefixPending`
    /// for Promoters). Returns `None` for owners not currently descending,
    /// stale ids, or any other state.
    ///
    /// Sole reader API for the descent-state payload outside the routing
    /// match sites in `on_*_probe_response`. The exhaustive `ProbeOwner`
    /// match enforces that adding a new owner kind requires extending
    /// this accessor; the per-state-type projection
    /// ([`ProfileState::descent_state`] /
    /// [`PromoterState::descent_state`]) owns the in-variant payload
    /// match, so the dispatcher here stays a thin two-line route.
    #[must_use]
    pub(crate) fn descent_state(&self, owner: ProbeOwner) -> Option<&DescentState> {
        match owner {
            ProbeOwner::Profile(pid) => self.profiles.get(pid)?.state().descent_state(),
            ProbeOwner::Promoter(pid) => self.promoters.get(pid)?.state.descent_state(),
        }
    }

    /// Mutable counterpart to [`Engine::descent_state`].
    pub(crate) fn descent_state_mut(&mut self, owner: ProbeOwner) -> Option<&mut DescentState> {
        match owner {
            ProbeOwner::Profile(pid) => self.profiles.get_mut(pid)?.descent_state_mut(),
            ProbeOwner::Promoter(pid) => self.promoters.get_mut(pid)?.state.descent_state_mut(),
        }
    }

    /// Pure, deterministic, total. Consumes one [`Input`], emits a sorted
    /// [`StepOutput`]. Each variant routes to the corresponding
    /// `on_*` handler (`transitions.rs`) or registration entry
    /// (`attach_sub_inner` / `detach_sub_inner` / `attach_promoter_inner`).
    /// Exhaustive — adding a variant to [`Input`] is a compile error
    /// here until a handler lands.
    ///
    /// Lifecycle inputs ([`Input::AttachSub`], [`Input::DetachSub`],
    /// [`Input::AttachPromoter`]) surface their minted ids via
    /// `out.diagnostics` ([`Diagnostic::SubAttached`] /
    /// [`Diagnostic::PromoterAttached`]), not via a synchronous
    /// return. The bin's loader maps `name → SubId` / `name →
    /// PromoterId` from those diagnostics, so the dispatcher's
    /// uniform shape (one input, one [`StepOutput`]) holds across
    /// every variant.
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
            Input::EffectComplete { sub, key, result } => {
                self.on_effect_complete(sub, &key, &result, now, &mut out);
            }
            Input::WatchOpRejected {
                resource,
                op,
                failure,
            } => {
                self.on_watch_op_rejected(resource, op, failure, &mut out);
            }
            Input::ConfigDiff(diff) => {
                self.on_config_diff(diff, now, &mut out);
            }
            Input::SensorOverflow { scope } => {
                self.on_sensor_overflow(scope, now, &mut out);
            }
            Input::AttachSub(req) => {
                let _ = self.attach_sub_inner(req, now, &mut out);
            }
            Input::DetachSub(sub) => {
                self.detach_sub_inner(sub, &mut out);
            }
            Input::AttachPromoter(req) => {
                let _ = self.attach_promoter_inner(req, now, &mut out);
            }
        }
        out.sort_for_emission();
        out
    }

    /// Attach a Sub to an existing Resource (`req.resource`) or to a
    /// path that the engine materialises (`req.path`). Reuses an
    /// existing Profile when `(resource, config_hash)` matches;
    /// otherwise creates a fresh Profile, emits `WatchOp::Watch` on its
    /// anchor, and starts a Seed burst (`PreFireBurst { intent: Seed,
    /// phase: Verifying }`) to establish the initial baseline.
    ///
    /// Three-phase pipeline; sole public entry is
    /// [`Input::AttachSub`] via [`Self::step`]. The inner is
    /// `pub(crate)` so [`Self::on_config_diff`] can compose multiple
    /// detach/attach operations into one [`StepOutput`] on hot reload.
    ///
    /// **Zombie revival.** When the matched Profile is in deferred-reap
    /// state ([`BurstFinish::Reap`], set by `detach_sub_inner` when
    /// the last Sub detached during an Active burst), the attach
    /// revives it: [`Diagnostic::ReapPendingCancelled`] emits, the
    /// directive flips back to [`BurstFinish::ReturnToIdle`], and the
    /// cleanup the deferred detach skipped (`recompute_profile_settle`)
    /// runs. The in-flight burst continues to completion under the
    /// new Sub set.
    ///
    /// On path rejection, returns `None` and emits a
    /// [`Diagnostic::AttachPathInvalid`]; the
    /// resource-based path cannot fail and always returns `Some`.
    /// External callers consume the [`Diagnostic::SubAttached`] /
    /// [`Diagnostic::AttachPathInvalid`] stream to reconcile their
    /// `name → SubId` index.
    ///
    /// # Production invariants (path-based attach)
    ///
    /// 1. **Absolute paths only.** `req.path` must be absolute and
    ///    UTF-8. [`Tree::parse_attach_path`] is the canonical gate; it
    ///    rejects non-absolute paths, non-UTF-8 segments, `.` / `..`
    ///    components, Windows path prefixes, and empty segments. The
    ///    bin layer's `canonicalize_lenient` already enforces absolute
    ///    paths for TOML-loaded configs, but hot-reload
    ///    `ConfigDiff::added` constructs `SubAttachRequest` from a
    ///    different path; the gate keeps the engine's contract
    ///    independent of every caller.
    /// 2. **Single FS-root.** Every validated [`TreePath`] starts with
    ///    [`FS_ROOT_SEGMENT`]; `materialize_path_or_pending` lazily
    ///    bootstraps a synthetic `/` slot (role
    ///    `ResourceRole::DescentScaffold`) before the pre-existence
    ///    walk so every Profile's rewind chain terminates at this
    ///    shared slot. The FS-root invariant is documented here rather
    ///    than enforced at the Tree type level — unit tests for
    ///    lower-level Tree functions (`coverage`, `refcounts`) still
    ///    construct multi-root trees outside of the attach pipeline.
    /// 3. **`Tree::path_of` reconstructs absolute paths.**
    ///    `PathBuf::push("/")` resets the buffer to absolute, so the
    ///    Sensor's `WatchOp::Watch { path }` always carries an absolute
    ///    path for any Profile registered through this pipeline.
    ///
    /// # Panics
    /// Panics if `req.resource` is stale (no live Tree slot) on the
    /// resource-based attach path. The Engine must construct the
    /// Resource before attaching a Sub to it.
    ///
    /// # Pipeline (three phases)
    ///
    /// 1. **Identity resolution.** `resolve_attach_anchor` parses the
    ///    request's `path` (or trusts `resource`), materialises the
    ///    Tree, and yields a typed [`AnchorResolution`] indicating
    ///    whether the anchor is materialised (`Immediate`) or
    ///    scaffolded (`Pending`). `find_or_create_profile` then
    ///    classifies the `(anchor, config_hash)` lookup into a
    ///    [`ProfileOrigin`] trichotomy.
    /// 2. **Sub registration.** `register_sub` consumes the request to
    ///    mint the [`Sub`] and emit [`Diagnostic::SubAttached`] — the
    ///    single point at which a SubId enters the registry.
    /// 3. **Per-origin bookkeeping.** Existing-Profile arms run their
    ///    targeted cleanup (`revive_zombie` /
    ///    `join_existing`); the `Fresh` arm dispatches on the
    ///    anchor resolution to either `bootstrap_pending` (Idle →
    ///    Pending) or `bootstrap_immediate` (anchor watch +
    ///    `watch_root_parent` + parent edges + Seed burst).
    ///
    /// `bootstrap_pending` and `bootstrap_immediate` are *not*
    /// interchangeable orderings of the same operations — they encode
    /// a real semantic divide:
    /// - **Pending** runs parent-edge work at attach time (Tree
    ///   topology is available the moment scaffolds materialise) and
    ///   defers anchor-watch installation to descent's anchor branch
    ///   (`dispatch_descent_ok`).
    /// - **Immediate** runs anchor-watch installation, the
    ///   `watch_root_parent` bump, and parent-edge work all at attach
    ///   time, then starts the Seed burst directly.
    pub(crate) fn attach_sub_inner(
        &mut self,
        req: SubAttachRequest,
        now: Instant,
        out: &mut StepOutput,
    ) -> Option<SubId> {
        // Phase 1 — Identity resolution. The trichotomy below is the
        // structural source of truth for "what state is this Profile
        // entering on this attach?". Two predicates the pre-Phase-4
        // shape derived ambiguously are now exhaustively typed:
        // - "no live Subs on the Profile" is ambiguous against
        //   `ZombieRevival` (the prior burst hasn't released its anchor
        //   claim yet) — the origin tells you whether the Sub is the
        //   *first* on the Profile or the *first since reap was
        //   deferred*.
        // - `anchor_claim == None` is ambiguous against `Fresh` (no
        //   bump yet) vs `Pending` revival (descent prefix carried it
        //   instead). The fresh-Profile arm structurally cannot mean
        //   "Profile existed but its anchor was unbumped."
        let resolved = self.resolve_attach_anchor(&req, out)?;
        let anchor = resolved.anchor();
        let cfg_hash = compute_config_hash(&req.config, req.max_settle, req.events);
        let (profile_id, origin) = self.find_or_create_profile(anchor, &req, cfg_hash);

        // Phase 2 — Sub registration. Consumes `req` for the
        // `Sub::new` move; captures `settle` first for the
        // `ExistingJoin` arm below (the request is no longer
        // accessible after this point).
        let attach_settle = req.settle;
        let sub_id = self.register_sub(req, profile_id, out);

        // Phase 3 — Per-origin bookkeeping. Existing-Profile arms run
        // their targeted cleanup and stop; the `Fresh` arm dispatches
        // on the anchor resolution.
        //
        // The events mask folds into `config_hash`, so a Sub joining
        // an existing Profile shares its mask by construction —
        // `events_union` and `has_per_file_fds` are invariant for the
        // Profile's lifetime. No retroactive per-leaf `watch_demand`
        // bump is needed on either existing-Profile arm.
        match origin {
            ProfileOrigin::ZombieRevival => self.revive_zombie(profile_id, out),
            ProfileOrigin::ExistingJoin => self.join_existing(profile_id, attach_settle),
            ProfileOrigin::Fresh => match resolved {
                AnchorResolution::Immediate { anchor } => {
                    self.bootstrap_immediate(profile_id, anchor, now, out);
                }
                AnchorResolution::Pending {
                    prefix, remaining, ..
                } => {
                    self.bootstrap_pending(profile_id, prefix, remaining, out);
                }
            },
        }

        Some(sub_id)
    }

    /// Phase 1 of `attach_sub_inner` — resolve the request's anchor to
    /// a Tree slot, classifying the outcome as `Immediate` (anchor is
    /// materialised on disk) or `Pending` (anchor is a scaffold; the
    /// engine must descend before the burst can start).
    ///
    /// Path-based attach (`req.path` set) routes through
    /// [`Tree::parse_attach_path`] (which rejects non-absolute,
    /// non-UTF-8, `.` / `..`, Windows-prefix, and empty-segment paths)
    /// and then [`Self::materialize_path_or_pending`]. On parse
    /// rejection, emits a [`Diagnostic::AttachPathInvalid`] and
    /// returns `None`.
    ///
    /// Resource-based attach (`req.path` `None`) trusts the caller's
    /// `req.resource` — the [`AnchorResolution::Immediate`] variant
    /// applies unconditionally because the caller has already
    /// guaranteed a live Tree slot.
    ///
    /// Promotes a `DescentScaffold` anchor to `User` on the
    /// `Immediate` path (the scaffold may have been left over from an
    /// earlier attach's `ensure_path` intermediate or from the FS-root
    /// bootstrap). The role is metadata — retention runs through the
    /// `Profile` back-ref installed by `find_or_create_profile`. The
    /// `Pending` arm defers role promotion to descent's anchor branch
    /// (`dispatch_descent_ok::materialize_profile_anchor`) where the
    /// slot becomes live on disk.
    fn resolve_attach_anchor(
        &mut self,
        req: &SubAttachRequest,
        out: &mut StepOutput,
    ) -> Option<AnchorResolution> {
        let resolved = match req.path.as_ref() {
            Some(path) => {
                let parsed = match Tree::parse_attach_path(path) {
                    Ok(p) => p,
                    Err(err) => {
                        out.diagnostics.push(Diagnostic::AttachPathInvalid {
                            path: path.clone(),
                            hint: err.hint(),
                        });
                        return None;
                    }
                };
                match self.materialize_path_or_pending(&parsed) {
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
                }
            }
            None => AnchorResolution::Immediate {
                anchor: req.resource,
            },
        };

        // Promote a `DescentScaffold` anchor to `User` on the
        // Immediate path. The Pending arm's anchor stays scaffolded
        // until descent's anchor branch flips it.
        if let AnchorResolution::Immediate { anchor } = resolved {
            self.tree
                .promote_scaffold(anchor, specter_core::ResourceRole::User);
        }

        Some(resolved)
    }

    /// Phase 2 of `attach_sub_inner` — register the Sub and emit
    /// [`Diagnostic::SubAttached`].
    ///
    /// Consumes `req` for the [`Sub::new`] move; captures diagnostic
    /// fields up front so the closure can take `req.name` and
    /// `req.source_promoter` by value. Cheap: a `CompactString` copy
    /// of a typically-short user name (inline storage at ≤24 bytes)
    /// and an `Option<PromoterId>` Copy.
    ///
    /// Sole emitter of `SubAttached`; downstream Phase-3 helpers
    /// never re-emit, and the bin's `loader.subs.name → SubId` map
    /// derives exclusively from this diagnostic stream.
    fn register_sub(
        &mut self,
        req: SubAttachRequest,
        profile_id: ProfileId,
        out: &mut StepOutput,
    ) -> SubId {
        let diag_name = CompactString::from(req.name.as_str());
        let diag_source_promoter = req.source_promoter;
        let sub_id = self.subs.insert(|sid| {
            Sub::new(
                sid,
                req.name,
                profile_id,
                req.program,
                req.scope,
                req.settle,
                req.max_settle,
                req.events,
                req.log_output,
                req.source_promoter,
            )
        });
        out.diagnostics.push(Diagnostic::SubAttached {
            sub: sub_id,
            name: diag_name,
            source_promoter: diag_source_promoter,
        });
        sub_id
    }

    /// Phase 3 of `attach_sub_inner` — zombie-revival arm.
    ///
    /// The deferred-reap detach branch flipped the Active burst's
    /// finish directive to [`BurstFinish::Reap`] and skipped the
    /// `fired_subs` purge + `recompute_profile_settle` the
    /// refcount-still-positive detach path performs. This helper
    /// un-defers the reap and runs the cleanup symmetrically:
    /// [`ProfileState::clear_active_reap`] flips
    /// `BurstFinish::Reap → ReturnToIdle` on Active (returning `true`
    /// by construction of the [`ProfileOrigin::ZombieRevival`]
    /// classification — the `debug_assert!` pins the invariant against
    /// a future routing breach), emits
    /// [`Diagnostic::ReapPendingCancelled`], and recomputes
    /// `Profile.settle` over the live Sub set (just the attaching Sub
    /// on first revival; further attaches in the same step take the
    /// `ExistingJoin` arm because the directive is already cleared).
    fn revive_zombie(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let cleared = self
            .profiles
            .get_mut(profile_id)
            .is_some_and(specter_core::Profile::clear_active_reap);
        debug_assert!(
            cleared,
            "revive_zombie: ZombieRevival origin must clear an active Reap directive \
             (profile = {profile_id:?})",
        );
        out.diagnostics.push(Diagnostic::ReapPendingCancelled {
            profile: profile_id,
        });
        self.recompute_profile_settle(profile_id);
    }

    /// Phase 3 of `attach_sub_inner` — existing-join arm.
    ///
    /// `attach_settle` is the request's `settle`; the Profile's
    /// `settle` is the min over its live Subs' settles, so the
    /// attaching Sub only shrinks it (and only when its settle is
    /// strictly lower). No other bookkeeping is required: the events
    /// mask folds into `config_hash`, so a joining Sub shares the
    /// existing Profile's mask by construction.
    fn join_existing(&mut self, profile_id: ProfileId, attach_settle: Duration) {
        if let Some(p) = self.profiles.get_mut(profile_id)
            && attach_settle < p.settle
        {
            p.settle = attach_settle;
        }
    }

    /// Phase 3 of `attach_sub_inner` — fresh-Profile, immediate-Seed
    /// arm (anchor materialised on disk at attach time).
    ///
    /// Sequence:
    /// 1. Install the Profile's anchor [`ContribKey::ProfileAnchor`]
    ///    contribution at `events_union` mask.
    /// 2. Flip [`AnchorClaim::Held`].
    /// 3. Set up the `watch_root_parent` (STRUCTURE contribution at
    ///    the anchor's parent, for anchor-reappearance detection).
    /// 4. Compute the Profile's parent edge and recompute parent edges
    ///    of any strict descendants that may now re-parent to this
    ///    Profile.
    /// 5. Start the Seed burst (`PreFire(Verifying)`); the post-probe
    ///    `dispatch_seed_ok` establishes the baseline.
    ///
    /// Contrast with [`Self::bootstrap_pending`]: the Pending arm runs
    /// only the parent-edge work + descent entry — the anchor watch
    /// and `watch_root_parent` bump are deferred to descent's anchor
    /// branch (`dispatch_descent_ok::materialize_profile_anchor`),
    /// which runs them when the anchor becomes live on disk.
    fn bootstrap_immediate(
        &mut self,
        profile_id: ProfileId,
        anchor: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Capture the Profile's mask before any &mut borrows.
        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, |p| p.events_union);

        add_watch(
            &mut self.tree,
            anchor,
            ContribKey::ProfileAnchor(profile_id),
            events_union,
            out,
        );
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.install_anchor_claim_held();
        }
        self.set_watch_root_parent(profile_id, anchor, out);
        self.install_parent_edges_for(profile_id);

        self.start_seed_burst(profile_id, now, out);
    }

    /// Phase 3 of `attach_sub_inner` — fresh-Profile, pending-descent
    /// arm (anchor scaffolded but not yet materialised on disk).
    ///
    /// Parent-edge work runs at attach time: the Tree topology is
    /// available the moment the scaffolds are written by
    /// [`Self::materialize_path_or_pending`], even though the leaf
    /// isn't yet a live `User`-roled slot. The anchor-watch
    /// installation, the `watch_root_parent` bump, and the Seed-burst
    /// launch are all deferred to descent's anchor branch
    /// (`dispatch_descent_ok::materialize_profile_anchor`).
    ///
    /// `enter_pending_descent` itself handles the four-step
    /// `Idle → Pending` entry sequence
    /// (`mint correlation → state-flip → add_watch on prefix → emit probe`)
    /// — by contract the helper does NOT touch parent edges (the
    /// recovery path's call site doesn't need it). Keeping the helper
    /// minimal preserves that contract.
    fn bootstrap_pending(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        remaining: DescentRemaining,
        out: &mut StepOutput,
    ) {
        self.install_parent_edges_for(profile_id);
        self.enter_pending_descent(profile_id, prefix, remaining, out);
    }

    /// Find an existing Profile at `(anchor, cfg_hash)` or create a
    /// fresh one. Returns the [`ProfileId`] and a [`ProfileOrigin`]
    /// classifying the outcome — `Fresh`, `ExistingJoin`, or
    /// `ZombieRevival` (existing Profile carrying
    /// [`BurstFinish::Reap`]).
    ///
    /// The slim three-variant enum supersedes the prior
    /// `is_fresh_profile + was_zombie` two-read pattern: the
    /// trichotomy is captured in one read, and downstream branches
    /// dispatch on the typed origin rather than re-deriving zombie
    /// state from `reap_pending`.
    ///
    /// **Fresh-Profile bookkeeping that lives here.** The anchor's
    /// classified kind is read from the Tree slot and threaded through
    /// [`Profile::new`]: `None` for a `DescentScaffold` anchor (descent
    /// materialisation classifies it) or a freshly-`ensure`d-but-unprobed
    /// slot (first Seed-Ok classifies it). Existing Profiles already
    /// carry the field from their own first-classify moment.
    fn find_or_create_profile(
        &mut self,
        anchor: ResourceId,
        req: &SubAttachRequest,
        cfg_hash: u64,
    ) -> (ProfileId, ProfileOrigin) {
        if let Some(pid) = self.profiles.find(anchor, cfg_hash) {
            let zombie = self
                .profiles
                .get(pid)
                .and_then(|p| p.state().burst_finish())
                == Some(BurstFinish::Reap);
            let origin = if zombie {
                ProfileOrigin::ZombieRevival
            } else {
                ProfileOrigin::ExistingJoin
            };
            return (pid, origin);
        }
        // Read the anchor's classified kind before construction:
        // `profiles.attach` only registers Profile-side indices on the
        // anchor slot, never its `kind`, so the slot's classification is
        // identical before and after. Threading it through the
        // constructor removes the post-attach re-borrow + `expect`.
        let anchor_kind = self.tree.get(anchor).and_then(specter_core::Resource::kind);
        let p = Profile::new(
            anchor,
            req.config.clone(),
            req.max_settle,
            req.settle,
            req.events,
            anchor_kind,
        );
        let pid = self.profiles.attach(&mut self.tree, p);
        (pid, ProfileOrigin::Fresh)
    }

    /// Set up the Profile's watch-root parent contribution. For each
    /// User-role Profile P, the Engine ensures `P.resource.parent` (if
    /// it exists) carries a `+1` `watch_demand` contribution from P. The
    /// parent's role is promoted to `WatchRootParent` only if it was
    /// previously a bare `DescentScaffold`; `User` parents stay `User`
    /// (never demote User). The role tag is metadata — retention runs
    /// through the [`ContribKey::ProfileParent`] entry installed below.
    ///
    /// Caches the parent id on `Profile.watch_root_parent` so
    /// `reap_profile` can release the contribution without re-deriving.
    /// `None` if the anchor has no parent in the Tree (a root anchor) —
    /// root rename detection is then unavailable.
    ///
    /// Sole call sites: `attach_sub_inner` (immediate-Seed path, where
    /// the anchor exists on disk and so does its parent) and
    /// `descent::dispatch_descent_ok` (anchor materialization).
    pub(crate) fn set_watch_root_parent(
        &mut self,
        profile_id: ProfileId,
        anchor: ResourceId,
        out: &mut StepOutput,
    ) {
        let Some(parent_id) = self.tree.parent(anchor) else {
            return;
        };

        // Cache-coherence invariant. If the cache already names a parent,
        // it must equal the Tree's current `parent(anchor)` — the anchor's
        // parent in the Tree is structurally stable for the Profile's
        // lifetime (Tree slot identity is `(parent, segment)` and the
        // anchor's `(parent, segment)` doesn't migrate). A mismatched
        // cache would mean either the parent migrated under us (impossible
        // by Tree invariants) or a prior `set_watch_root_parent` wrote
        // against a different anchor for this Profile (a routing breach).
        // The release path (`release_watch_root_parent_claim`) reads the
        // cache to key the contribution removal, so a stale cache would
        // leak the old parent's `+1`.
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_none_or(|p| p.watch_root_parent.is_none_or(|cached| cached == parent_id)),
            "set_watch_root_parent: cached parent must agree with the materialised \
             anchor's parent (profile = {profile_id:?}, cached = {:?}, tree_parent = {parent_id:?})",
            self.profiles
                .get(profile_id)
                .and_then(|p| p.watch_root_parent),
        );

        // Idempotent: "Watch root deletion" recovery re-enters descent
        // on a Profile whose `watch_root_parent` field was set at the
        // original materialization and never cleared on
        // `on_anchor_terminal_event`. When recovery's descent advances
        // back to anchor materialization it would otherwise call this
        // helper again, double-bumping the parent's `watch_demand` for
        // the same Profile. Skip the bump if the cache already points
        // at the same parent id.
        let already_set = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.watch_root_parent == Some(parent_id));
        if already_set {
            return;
        }

        // Promote role: DescentScaffold → WatchRootParent. User and
        // existing WatchRootParent stay as they are (the helper
        // preserves non-scaffold roles).
        self.tree
            .promote_scaffold(parent_id, specter_core::ResourceRole::WatchRootParent);

        // The watch-root parent is engine infrastructure (used to detect
        // anchor reappearance after a `rm -rf` of the anchor).
        // Contribution is `STRUCTURE` regardless of the Sub's user mask.
        // The corresponding bookkeeping flag is `Profile.watch_root_parent
        // == Some(parent_id)`, written below.
        add_watch(
            &mut self.tree,
            parent_id,
            ContribKey::ProfileParent(profile_id),
            ClassSet::STRUCTURE,
            out,
        );

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.watch_root_parent = Some(parent_id);
        }
    }

    /// Rewrite parent-edge cache entries affected by attaching
    /// `new_profile`. Two classes of edges may move:
    ///
    /// 1. **The new Profile's own edge.** Derived from its anchor's
    ///    strict ancestors via [`crate::coverage::nearest_covering_ancestor`].
    /// 2. **Strict-descendant Profiles' edges.** A Profile P' at a
    ///    strict descendant of `new_profile.resource` may now name
    ///    `new_profile` as its nearest covering ancestor — the new
    ///    Profile interposes between P' and P''s prior parent.
    ///    Profiles at sibling subtrees, at ancestor positions, or at
    ///    the same anchor (different `config_hash`) are not affected
    ///    (the new Profile is not a covering ancestor for them).
    ///
    /// Routes through [`crate::stability::collect_in_subtree`] +
    /// [`crate::stability::recompute_parent_edges`] with the engine's
    /// `scratch_profile_ids` buffer mediating the borrow. The
    /// `collect_in_subtree` call fills the scratch with strict
    /// descendants; the `push(new_profile)` appends `new_profile`
    /// itself so the recompute loop handles both classes in one pass.
    ///
    /// Recompute order is irrelevant for correctness:
    /// `nearest_covering_ancestor` reads
    /// `Resource.profiles` (the back-index) at each ancestor, never
    /// the `Profile.parent_profile` field rewritten in the loop —
    /// per-iteration writes do not affect subsequent iterations.
    fn install_parent_edges_for(&mut self, new_profile: ProfileId) {
        // The caller (`attach_sub_inner`) just inserted-or-found the
        // Profile and the slot is still live; a missing slot here is
        // a structural invariant breach.
        let new_anchor = self
            .profiles
            .get(new_profile)
            .map(|p| p.resource)
            .expect("install_parent_edges_for: caller's profile_id must be live");
        crate::stability::collect_in_subtree(
            &self.tree,
            &self.profiles,
            new_anchor,
            &mut self.scratch_profile_ids,
        );
        self.scratch_profile_ids.push(new_profile);
        crate::stability::recompute_parent_edges(
            &self.tree,
            &mut self.profiles,
            self.scratch_profile_ids.drain(..),
        );
    }

    /// Detach a Sub by id.
    ///
    /// Recomputes `Profile.settle = min(remaining_subs.settles)`. If no
    /// Subs remain on the Profile (`subs.at(pid)` empty):
    /// - **Idle / Pending Profile:** reap immediately. Release anchor
    ///   `watch_demand` (1→0 emits Unwatch), release
    ///   `watch_root_parent` contribution, clear parent edge,
    ///   recompute parent edges of dependents, and `try_reap` the
    ///   anchor Resource.
    /// - **Active Profile:** flip the burst's [`BurstFinish::Reap`]
    ///   directive via [`ProfileState::mark_active_for_reap`]. The
    ///   active burst runs to completion; on `finish_burst_to_idle`,
    ///   the Engine skips Effect emission (`emit_effects` reads the
    ///   directive) and reaps the Profile in the same step as the
    ///   Active → Idle transition (any pre-fire phase converges
    ///   through `finish_burst_to_idle`).
    ///
    /// If the count remains > 0, the Profile stays alive; only
    /// `Profile.settle` is recomputed.
    ///
    /// Idempotent on stale `SubId` ([`Diagnostic::DetachUnknownSub`] +
    /// drop). Sole public entry is [`Input::DetachSub`] via
    /// [`Self::step`]; the `pub(crate)` inner survives because
    /// [`Self::on_config_diff`] composes multiple detach/attach
    /// operations into one [`StepOutput`] on hot reload.
    ///
    /// Time-independent: detach is a pure registry/refcount operation
    /// (no timer scheduling, no burst transitions that need a `now`).
    /// Bursts running on detached Profiles continue under their existing
    /// schedule until `finish_burst_to_idle`.
    pub(crate) fn detach_sub_inner(&mut self, sub: SubId, out: &mut StepOutput) {
        let profile_id = match self.subs.remove(sub) {
            Some(s) => s.profile,
            None => {
                out.diagnostics.push(Diagnostic::DetachUnknownSub { sub });
                return;
            }
        };

        // A live Sub's `.profile` is live by the attach invariant; this
        // guard is defence-in-depth — same effect as the get_mut-borrow
        // bail it replaces, but no Profile write happens here (the
        // post-detach count is derived from the registry).
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        // Post-`remove` count, read straight from the Sub registry —
        // `Profile.sub_refcount` was a denormalised mirror of this and
        // has been removed.
        let remaining_subs = self.subs.at(profile_id).len();

        // Purge `fired_subs` entries keyed by the detached Sub. The fire
        // history must drop with the Sub: a future drift verdict on the
        // Profile must not re-fire an Effect for a Sub the user has
        // detached. The full reap path below drops the whole set
        // alongside the Profile, so this targeted purge runs only on the
        // subs-remaining branch.
        if remaining_subs > 0 {
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.fired_subs.retain(|k| match k {
                    DedupKey::Subtree { sub: s, .. } | DedupKey::PerFile { sub: s, .. } => {
                        *s != sub
                    }
                });
            }
            // Recompute Profile.settle = min(remaining_subs.settles).
            //
            // Every Sub on a Profile shares the same `events` mask
            // (events folds into `config_hash`); detaching one Sub
            // cannot flip `Profile.has_per_file_fds` or
            // `Profile.events_union`.
            self.recompute_profile_settle(profile_id);
            return;
        }

        // No Subs remain: classify the reap path via the typed
        // [`ProfileState::detach_lifecycle`] projection — `ReapNow` for
        // Idle / Pending Profiles (no burst to drain), `DeferToBurstEnd`
        // for Active Profiles (the burst's `propagate(-1) / sub_suppress`
        // drain must run first). Pending Profiles reap synchronously
        // alongside Idle: there is no `finish_burst_to_idle` to
        // resolve a deferred reap.
        let lifecycle = self
            .profiles
            .get(profile_id)
            .map(|p| p.state().detach_lifecycle());
        match lifecycle {
            Some(DetachLifecycle::ReapNow) => {
                self.reap_profile(profile_id, ReapTrigger::Immediate, out);
            }
            Some(DetachLifecycle::DeferToBurstEnd) => {
                // `fired_subs` purge and `recompute_profile_settle` are
                // deliberately skipped — the Profile is about to drop on
                // burst end, so the cleanup would be wasted. A revival
                // via fresh `attach_sub_inner` (zombie-revival branch)
                // un-defers the reap and runs the cleanup symmetrically.
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

    /// Reap a Profile: release every contribution it holds (anchor watch,
    /// watch-root parent watch, descent prefix watch, per-descendant
    /// watches), clear its parent edge, recompute parent edges of any
    /// dependents, detach from `ProfileMap`, try-reap the anchor Resource,
    /// and emit a [`Diagnostic::ProfileReaped`] carrying the
    /// [`ReapTrigger`] that drove this reap. The trigger is supplied by
    /// the caller (not derived from state) because the two paths reach
    /// `reap_profile` with structurally distinct preconditions:
    /// `Immediate` from `detach_sub_inner` on Idle/Pending (no burst),
    /// and `DeferredFromBurst` from `finish_burst_to_idle` honouring a
    /// prior [`BurstFinish::Reap`] (the burst's drain has just run).
    ///
    /// **Quartet.** A Profile may hold up to four kinds of contribution
    /// to per-Resource `watch_demand`:
    ///
    ///   - **Anchor** (1-to-1): `Profile.resource.watch_demand` carries
    ///     `+1` from this Profile while
    ///     `Profile.anchor_claim == AnchorClaim::Held`.
    ///   - **Watch-root parent** (1-to-1): `Profile.watch_root_parent`'s
    ///     resource carries `+1` `STRUCTURE` for anchor-reappearance
    ///     detection.
    ///   - **Descent prefix** (1-to-1): the deepest existing prefix on
    ///     a Pending Profile's path carries `+1` `STRUCTURE`.
    ///   - **Per-descendant** (1-to-N): every covered Tree slot in
    ///     `Profile.current` carries `+1` (Dir always; Leaf under
    ///     `has_per_file_fds`). The 1-to-N source-of-truth is the
    ///     snapshot itself, not a per-Profile flag.
    ///
    /// **Trichotomy invariant** (preserved from prior shape, now within
    /// the quartet). Anchor and descent-prefix are mutually exclusive at
    /// any moment: either the Profile is `Pending` (descent prefix only)
    /// or materialized (anchor + descendants + watch-root parent). The
    /// clamp recovery path (`Input::WatchOpRejected`) leaves the Profile
    /// with no contributions; the purge fan-out cleans up the
    /// bookkeeping.
    ///
    /// # Partial order
    ///
    /// `reap_profile`'s steps partition into five strictly-ordered groups
    /// where group N must complete before group N+1 begins. Within
    /// groups (1) and (4) members are unordered — any permutation
    /// produces an equivalent [`StepOutput`]:
    ///
    /// 1. **Probe channel close.** [`Engine::cancel_owner_probe`] emits a
    ///    `ProbeOp::Cancel` for any in-flight probe (Pending Profile's
    ///    descent probe; Active never reaches this entry — the response-
    ///    dispatch path closes the channel before `finish_burst_to_idle`
    ///    runs `reap_profile`). Must precede (2) because
    ///    [`Engine::release_descent_prefix_claim`] debug-asserts the
    ///    channel is closed (the cancel-first contract; see the helper's
    ///    rustdoc).
    ///
    /// 2. **Release quartet** — `release_descent_prefix_claim`,
    ///    `release_descendant_claim`, `release_anchor_claim`,
    ///    `release_watch_root_parent_claim`. Each is idempotent,
    ///    counter-aware, and safe on a post-vacate slot. None reads or
    ///    mutates state another might also touch:
    ///    - `release_descendant_claim` `take()`s `Profile.current`; the
    ///      other three never read it.
    ///    - `release_anchor_claim` removes `ProfileAnchor(pid)` from the
    ///      anchor's contributions; the other three never read the
    ///      anchor's contribution map.
    ///    - `release_watch_root_parent_claim` calls `try_reap` on the
    ///      *parent* slot (no-op while the anchor is still a child); the
    ///      anchor's eventual `try_reap` in (5) cascades upward and
    ///      reaps the parent then.
    ///    - `release_descent_prefix_claim` releases the descent-prefix
    ///      contribution and transitions `Pending → Idle`; mutually
    ///      exclusive with anchor by the trichotomy invariant.
    ///
    ///    Cardinality order chosen for readability: 1-to-1 prefixed
    ///    claims first, the 1-to-N descendant walk, then the remaining
    ///    1-to-1 claims. Any of 4! = 24 permutations is correct.
    ///
    /// 3. **`ProfileMap::detach`.** Must follow (2) — the release
    ///    helpers read `&Profile` (anchor, watch_root_parent,
    ///    snapshot). Must precede (4) — `collect_pointing_at` filters
    ///    against the post-detach map.
    ///
    /// 4. **Parent-edge recompute and anchor try-reap.** Two operations,
    ///    mutually independent because they touch disjoint state:
    ///    - `collect_pointing_at` + `recompute_parent_edges` rewrites
    ///      dependents' `Profile.parent_profile` fields (touches
    ///      `ProfileMap` only).
    ///    - `Tree::try_reap(anchor)` cascades upward through any now-
    ///      orphaned ancestors (touches `Tree` only; the watch-root
    ///      parent slot is freed in this step if the anchor was its
    ///      sole remaining child).
    ///
    ///    Code order is parent-edge → try_reap, but the reverse is
    ///    equally correct.
    ///
    /// 5. **`Diagnostic::ProfileReaped` emit.** Last by convention so the
    ///    diagnostic ordering across a step (which is sorted by emission
    ///    site rather than by the [`StepOutput::sort_for_emission`] pass)
    ///    reads "do the work, then announce it."
    ///
    /// **Note on `discard_anchor_state` overlap.** This helper performs
    /// `release_descendant_claim` + `release_anchor_claim` inline
    /// rather than via [`Engine::discard_anchor_state`]. The two
    /// helpers differ in purpose:
    ///
    /// - `discard_anchor_state` exists for the "anchor lost, Profile
    ///   lives" case — the seven `dispatch_*_vanished/failed` +
    ///   `finalize_anchor_lost` sites. Its `kind = None` and
    ///   `baseline = None` writes prepare the Profile for the next
    ///   Seed burst's probe-shape dispatch, and it deliberately
    ///   preserves `watch_root_parent` (the recovery channel).
    /// - `reap_profile` is "Profile dies entirely." There is no next
    ///   Seed burst — the Profile detaches in group (3) — so the
    ///   `kind` and `baseline` writes that `discard_anchor_state` would
    ///   perform are wasted on a struct about to drop. Reap also
    ///   releases `watch_root_parent`, which `discard_anchor_state`
    ///   deliberately preserves.
    ///
    /// The structural overlap (both call `release_descendant_claim +
    /// release_anchor_claim`) is intentional; the field clears and
    /// `watch_root_parent` release are deliberately partitioned across
    /// the two helpers.
    ///
    /// Sole call sites: `detach_sub_inner` (Idle / Pending Profile,
    /// immediate reap; `via = Immediate`),
    /// `on_anchor_terminal_all_dynamic` (non-Active arm of the
    /// all-dynamic Promoter teardown path; `via = Immediate`), and
    /// `finish_burst_to_idle` (deferred reap when
    /// [`BurstFinish::Reap`] was set mid-burst; `via =
    /// DeferredFromBurst`).
    pub(crate) fn reap_profile(
        &mut self,
        profile_id: ProfileId,
        via: ReapTrigger,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let anchor = p.resource;

        // Trichotomy invariant: Pending and AnchorClaim::Held are mutually
        // exclusive. Descent flips Pending → Idle and bumps the anchor
        // atomically in `dispatch_descent_ok`'s anchor branch.
        debug_assert!(
            !matches!(
                (p.state(), p.anchor_claim()),
                (ProfileState::Pending(_), AnchorClaim::Held),
            ),
            "reap_profile: Pending + AnchorClaim::Held must be mutually exclusive",
        );

        // Close the probe channel BEFORE the descent-prefix helper
        // transitions the Profile to Idle. Idempotent: emits Cancel
        // iff a probe was in flight (Pending with a descent probe in
        // flight for this call path; Active+Verifying never reaches
        // `reap_profile`'s entry — `finish_burst_to_idle` runs
        // `reap_profile` only after the burst response cleared the
        // channel). Mirrors `on_watch_op_rejected`'s descent-purge
        // pattern.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);
        debug_assert!(
            self.probe_channel
                .correlation_for(ProbeOwner::Profile(profile_id))
                .is_none(),
            "reap_profile: probe channel still open for profile = {profile_id:?} \
             after cancel_owner_probe; channel-close contract violated",
        );

        // Release quartet — group (2) of the partial order (see rustdoc).
        // Members are mutually independent; the four helpers touch
        // disjoint state. Code order chosen for readability (1-to-1
        // prefixed claims first, then the 1-to-N descendant walk, then
        // the remaining 1-to-1 claims); any permutation is equally
        // correct.
        self.release_descent_prefix_claim(profile_id, out);
        self.release_descendant_claim(profile_id, out);
        self.release_anchor_claim(profile_id, out);
        self.release_watch_root_parent_claim(profile_id, out);

        // Detach the Profile from the registry. The Profile's
        // `parent_profile` field dies with the struct — no separate
        // clear step is needed. Dependents whose `parent_profile`
        // still points at the now-removed slot are rewritten by the
        // `collect_pointing_at + recompute_parent_edges` flow below.
        // The returned `Option<Profile>` carries the detached payload
        // for diagnostic use only at the inline site; `_detached`
        // names the discard so a reader doesn't have to chase whether
        // a field was needed.
        let _detached = self.profiles.detach(&mut self.tree, profile_id);

        crate::stability::collect_pointing_at(
            &self.profiles,
            profile_id,
            &mut self.scratch_profile_ids,
        );
        crate::stability::recompute_parent_edges(
            &self.tree,
            &mut self.profiles,
            self.scratch_profile_ids.drain(..),
        );

        // Try to reap the anchor's slot. No-op if it still has
        // children (a descendant Profile / Promoter / scaffold survives
        // here), other Profiles attached at the same slot, a Promoter
        // back-ref, or any co-resident contribution. On success,
        // [`Tree::try_reap`] cascades upward through any now-orphaned
        // ancestors — the watch-root parent slot whose only remaining
        // claim was *this* Profile's anchor as its sole child is freed
        // in the same step. `try_reap` folds in `Tree::vacate` as its
        // closing-emission step, so any residual per-slot protocol
        // (kernel-watch / burst-suppress) is emitted before the slot
        // leaves the Tree.
        self.tree.try_reap(anchor, out);

        out.diagnostics.push(Diagnostic::ProfileReaped {
            profile: profile_id,
            via,
        });
    }

    /// Recompute `Profile.settle = min(remaining_subs.settles)` after a
    /// Sub addition or removal. O(subs-on-profile), bounded — typically
    /// 1–2 in v1 because `max_settle` already partitions Profiles.
    ///
    /// Uses `.map(...).expect(...)` rather than a defensive
    /// `.filter_map(...)`: [`SubRegistry::insert`] and
    /// [`SubRegistry::remove`] keep the `by_profile` index and the
    /// slotmap in lockstep, so every id returned by [`SubRegistry::at`]
    /// is live in [`SubRegistry::get`] by construction. A `None`
    /// here would be a structural invariant breach; the `expect` is
    /// the surface that names it.
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
    /// The bin uses this to inspect Resource topology; tests use it for
    /// setup verification.
    #[must_use]
    pub const fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Mutable access for path-to-`ResourceId` materialization.
    ///
    /// The bin uses this at startup to walk a config's `path` strings into
    /// the Tree before calling `attach_sub`. Use the dedicated refcount
    /// helpers to modify `watch_demand` / `suppress_count` — direct
    /// mutation breaks the 0↔1 edge invariant.
    pub const fn tree_mut(&mut self) -> &mut Tree {
        &mut self.tree
    }

    /// Read-only view of the `ProfileMap`.
    ///
    /// For inspection only; state-machine mutations route through `step`
    /// and `attach_sub`.
    #[must_use]
    pub const fn profiles(&self) -> &ProfileMap {
        &self.profiles
    }

    /// Read-only view of the `SubRegistry`.
    #[must_use]
    pub const fn subs(&self) -> &SubRegistry {
        &self.subs
    }

    /// Read-only view of the `PromoterRegistry`.
    ///
    /// Symmetric with [`Self::subs`] / [`Self::profiles`]; integration
    /// tests inspect Promoter state through this accessor (the field
    /// itself is `pub(crate)` because all Promoter mutations go
    /// through engine-internal paths).
    #[must_use]
    pub const fn promoters(&self) -> &PromoterRegistry {
        &self.promoters
    }

    /// Earliest pending timer deadline, or `None` if no timers are armed.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        self.timers.peek_top().map(|e| e.deadline)
    }

    /// Pop the earliest expired-and-still-referenced timer. Stale entries
    /// (cancelled because the Profile's burst was reset) are silently
    /// dropped. The returned [`TimerEntry`] carries the owning profile,
    /// kind, and id together — the bin forwards it to
    /// [`Input::TimerExpired`] without any rediscovery.
    ///
    /// **Stale-drain bound (dev only).** A single call drains at most
    /// 32 stale entries before tripping a `debug_assert!`. Realistic
    /// per-call drain is single-digit — each Active Profile carries
    /// ~2 timer slots and orphans at most one per burst transition,
    /// and the bin's tick loop polls frequently enough that orphans
    /// collect incrementally. The bound is loose enough to absorb
    /// multi-Profile burst-end cleanup, tight enough to surface an
    /// "engine transition leaks timer references" regression. Release
    /// builds run unbounded (the lazy drain is correct either way; the
    /// bound is purely a developer-time invariant check).
    pub fn pop_expired(&mut self, now: Instant) -> Option<TimerEntry> {
        // Stale-drain accounting — see the rustdoc above for the bound's
        // semantics. `u32` is more than enough headroom even in a
        // release build where the assert is compiled out.
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

/// Whether `id` is the live timer for `profile`'s `kind` slot —
/// `pop_expired` uses this to filter stale heap heads, and
/// `on_timer_expired` re-runs it as defense-in-depth for direct
/// `step(Input::TimerExpired)` callers (tests, fuzzers).
///
/// Implemented as a thin lookup-and-compare against
/// [`ProfileState::timer_token`]: every `(state, kind)` pair routes
/// to whichever burst-side type carries the field, with the
/// type-impossible pairs (e.g., `Settle` on `PostFire`) folding to
/// `None` at the leaf without an explicit fallthrough arm. Returns
/// `false` for stale `profile` ids and for any state that doesn't
/// own a `kind` timer right now.
///
/// Free function rather than a method on [`Engine`]: the projection
/// is purely a query over [`ProfileMap`] and doesn't reach into the
/// engine's other fields, so a free function keeps the call-site
/// shape (`is_timer_referenced(&self.profiles, …)`) honest about
/// the dependency.
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

/// Find-or-create-or-revive outcome for `attach_sub_inner`. The three
/// arms drive distinct downstream bookkeeping:
/// - `Fresh`: brand-new Profile, no prior burst history. Bumps the
///   anchor's `watch_demand`, sets up `watch_root_parent`, computes
///   parent edges, starts a Seed burst (immediate) or enters Pending
///   descent.
/// - `ExistingJoin`: Profile is alive with at least one live Sub; the
///   attaching Sub joins the existing burst lifecycle (or shares the
///   existing baseline if Idle). Only `Profile.settle` may need
///   `min`-recompute.
/// - `ZombieRevival`: Profile is in [`ProfileState::Active`] with
///   [`BurstFinish::Reap`] (deferred reap). Clear the directive via
///   [`ProfileState::clear_active_reap`], emit
///   [`Diagnostic::ReapPendingCancelled`], and run the
///   `recompute_profile_settle` the deferred-reap detach skipped.
///
/// Engine-local because no external caller distinguishes the arms —
/// the engine dispatches on the typed origin inside `attach_sub_inner`
/// and returns only `Option<SubId>` from the public surface.
enum ProfileOrigin {
    Fresh,
    ExistingJoin,
    ZombieRevival,
}

/// Outcome of `attach_sub_inner`'s Phase 1 anchor resolution. The two
/// variants encode the semantic divide between attaches whose anchor
/// is already live on disk and those whose anchor is scaffolded
/// awaiting materialisation. The split drives Phase 3's
/// `bootstrap_immediate` vs `bootstrap_pending` dispatch.
///
/// The `Immediate` arm subsumes both path-based attach
/// (fully-materialised path) and resource-based attach (caller
/// supplies the live `ResourceId` directly via `req.resource`).
enum AnchorResolution {
    /// The anchor is a live, materialised Tree slot.
    /// `bootstrap_immediate` will install the anchor watch contribution
    /// and start the Seed burst.
    Immediate { anchor: ResourceId },
    /// The anchor is a `DescentScaffold`-roled slot awaiting
    /// materialisation. `prefix` is the deepest existing ancestor
    /// (where the descent probe is currently watching) and `remaining`
    /// carries the path components from `prefix` (exclusive) down to
    /// `anchor` (inclusive). `bootstrap_pending` will enter
    /// `ProfileState::Pending(_)` and start descent.
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: DescentRemaining,
    },
}

impl AnchorResolution {
    /// The anchor slot — the Profile's resource on attach. Both
    /// variants expose the same field name so `find_or_create_profile`
    /// can key off it without matching on the variant.
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
        DedupKey, EffectOutcome, FsEvent, Input, ProbeCorrelation, ProbeOutcome, ProbeResponse,
        ProfileId, ResourceId, ScanConfig, StepOutput, SubId, TimerId, TimerKind, WatchOp,
        WatchRegistryDiff,
    };
    use std::time::{Duration, Instant};

    // Compile-time `Send + Sync` check on `Engine`. The bin loop parks
    // `Engine` on its own thread; `Send + Sync` is load-bearing for that.
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
                event: FsEvent::Modified,
            },
            Instant::now(),
        );
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::EventOnUnwatchedResource { .. }));
        assert!(has_diag);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
    }

    #[test]
    fn step_probe_response_unknown_profile_diagnoses() {
        let mut e = Engine::new();
        let resp = ProbeResponse {
            owner: ProbeOwner::Profile(ProfileId::default()),
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
            Input::EffectComplete {
                sub: SubId::default(),
                key: DedupKey::default(),
                result: EffectOutcome::Ok,
            },
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
        // Stale ResourceId or already-Unwatched resource yields a
        // Diagnostic + no other ops.
        let mut e = Engine::new();
        let op = WatchOp::Unwatch {
            resource: ResourceId::default(),
        };
        let out = e.step(
            Input::WatchOpRejected {
                resource: ResourceId::default(),
                op,
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
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
            Input::ConfigDiff(WatchRegistryDiff::default()),
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
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
        e.timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
        assert_eq!(e.next_deadline(), Some(when));
    }

    #[test]
    fn pop_expired_returns_none_when_top_in_future() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_secs(10);
        e.timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future-dated entries are not drained");
    }

    #[test]
    fn pop_expired_drains_stale_entries_silently() {
        // Schedule timers for null/unknown Profiles (no Active state holds
        // them). The validating drain consumes every stale entry, but returns
        // None — there's nothing live to fire.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 0, "stale heads were drained");
    }

    /// Stale-drain bound: a `pop_expired` call that drains more than
    /// [`STALE_DRAIN_BOUND`] consecutive stale entries trips the
    /// `debug_assert!`. Production code is structurally bounded
    /// (per-Profile orphan count is small); this test pins the
    /// regression sensor by deliberately exceeding the bound.
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
        // Schedule one beyond the bound. All stale (default ProfileId
        // has no Active state), so the drain runs through them all.
        for _ in 0..=STALE_DRAIN_BOUND {
            e.timers
                .schedule(past, ProfileId::default(), TimerKind::Settle);
        }
        let _ = e.pop_expired(now);
    }

    #[test]
    fn pop_expired_stops_at_first_future_entry() {
        // Mix of expired-stale and future-dated. The drain consumes the stale
        // expired heads, then returns None when peeking a future-dated entry.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers.schedule(
            now + Duration::from_secs(10),
            ProfileId::default(),
            TimerKind::Settle,
        );

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future entry remains");
        assert!(e.next_deadline().unwrap() > now);
    }

    #[test]
    fn probe_channel_open_is_monotonic_per_owner() {
        // Three Profiles, each opened once: the channel's correlation
        // counter advances monotonically across opens regardless of
        // which Profile owns each open channel. Distinct owners avoid
        // the I5 double-open panic (one open channel per owner).
        use crate::probe_channel::OpenKind;
        let mut e = Engine::new();
        let r1 = e.tree.ensure_root("x", specter_core::ResourceRole::User);
        let r2 = e.tree.ensure_root("y", specter_core::ResourceRole::User);
        let r3 = e.tree.ensure_root("z", specter_core::ResourceRole::User);
        let cfg = ScanConfig::builder().build();
        let pid1 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r1,
                cfg.clone(),
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
                None,
            ),
        );
        let pid2 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r2,
                cfg.clone(),
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
                None,
            ),
        );
        let pid3 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r3,
                cfg,
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
                None,
            ),
        );

        let a = e
            .probe_channel
            .open(ProbeOwner::Profile(pid1), OpenKind::ProfileVerifying);
        let b = e
            .probe_channel
            .open(ProbeOwner::Profile(pid2), OpenKind::ProfileVerifying);
        let c = e
            .probe_channel
            .open(ProbeOwner::Profile(pid3), OpenKind::ProfileVerifying);
        assert!(a < b);
        assert!(b < c);
        assert_eq!(a, ProbeCorrelation::from(1));
        assert_eq!(b, ProbeCorrelation::from(2));
        assert_eq!(c, ProbeCorrelation::from(3));

        // Channel populated symmetrically.
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid1)), Some(a));
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid2)), Some(b));
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid3)), Some(c));
    }

    #[test]
    fn engine_default_constructible_has_empty_state() {
        let e = Engine::new();
        assert!(e.tree.is_empty());
        assert!(e.profiles.is_empty());
        assert!(e.subs.is_empty());
        assert!(e.timers.is_empty());
        assert!(e.next_deadline().is_none());
        assert_eq!(e.probe_channel.counter_peek(), 0);
        assert_eq!(e.effect_correlations.peek(), 0);
    }

    /// Counter saturation on the effect side — release-runnable. The
    /// effect counter has no per-call-site wrapper (Phase 1 inlined the
    /// minting at the two `emit_effects` push sites in
    /// `transitions.rs`), so this test exercises the counter directly to
    /// prove the field is wired up. Pairs with the `MonotonicCounter`
    /// unit tests in `counter.rs`.
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
        let req = SubAttachRequest::for_path(
            "bad".to_string(),
            bad.clone(),
            ScanConfig::builder().recursive(false).build(),
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
                specter_core::Diagnostic::AttachPathInvalid { path, .. } if path == &bad,
            )
        });
        assert!(saw, "AttachPathInvalid must carry the offending path");
    }

    /// End-to-end gate enforcement: a relative-path attach request rolls
    /// up no `SubId`, no Tree slots, and no Profile — only the diagnostic
    /// surfaces. Pins the contract that `attach_sub_inner`'s
    /// `Tree::parse_attach_path` short-circuit is total: rejection
    /// produces `None` plus zero side-effects on engine state.
    #[test]
    fn attach_with_relative_path_emits_diagnostic_and_no_state() {
        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        let bad = std::path::PathBuf::from("relative/path");
        let req = SubAttachRequest::for_path(
            "rel".to_string(),
            bad.clone(),
            ScanConfig::builder().recursive(false).build(),
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
        assert!(out.probe_ops.is_empty(), "no probe ops emitted");
        assert!(out.effects.is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == &bad && hint.contains("absolute"),
        )));
    }

    /// End-to-end counterpart for non-UTF-8 paths. The test fabricates a
    /// path with a non-UTF-8 segment via `OsStr::from_bytes` (Unix-only)
    /// and confirms the same total-rejection contract: no SubId, no Tree
    /// slots, no Profile.
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

        let req = SubAttachRequest::for_path(
            "bad".to_string(),
            path.clone(),
            ScanConfig::builder().recursive(false).build(),
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
        assert!(out.probe_ops.is_empty(), "no probe ops emitted");
        assert!(out.effects.is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path: p, hint }
                if p == &path && hint.contains("non-UTF-8"),
        )));
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

    // ===== set_watch_root_parent idempotence =====

    #[test]
    fn set_watch_root_parent_idempotent_on_recovery_path() {
        // "Watch root deletion" recovery re-enters descent on a Profile
        // whose `watch_root_parent` was set at the original materialization.
        // When recovery's descent advances back to anchor materialization,
        // set_watch_root_parent must not double-bump the parent's
        // watch_demand.
        let mut e = Engine::new();
        let parent = e.tree.ensure_root("p", specter_core::ResourceRole::User);
        let anchor = e
            .tree
            .ensure_child(parent, "a", specter_core::ResourceRole::User)
            .expect("test live parent");
        let profile = specter_core::Profile::new(
            anchor,
            ScanConfig::builder().build(),
            Duration::from_secs(1),
            Duration::from_millis(50),
            specter_core::ClassSet::EMPTY,
            None,
        );
        let pid = e.profiles.attach(&mut e.tree, profile);

        // First call: bumps parent's watch_demand and caches it on Profile.
        let mut out = StepOutput::default();
        e.set_watch_root_parent(pid, anchor, &mut out);
        let after_first = e.tree.get(parent).unwrap().watch_demand();
        assert_eq!(after_first, 1, "first call bumps parent watch_demand");
        assert_eq!(e.profiles.get(pid).unwrap().watch_root_parent, Some(parent));

        // Second call with the same anchor: must be a no-op (no bump).
        let mut out2 = StepOutput::default();
        e.set_watch_root_parent(pid, anchor, &mut out2);
        let after_second = e.tree.get(parent).unwrap().watch_demand();
        assert_eq!(after_second, 1, "second call does NOT double-bump");
        assert!(
            out2.watch_ops.is_empty(),
            "no Watch op emitted on second call"
        );
    }

    // ===== Zombie revival =====

    fn revival_attach_req(anchor: ResourceId, name: &str, settle: Duration) -> SubAttachRequest {
        SubAttachRequest::for_resource(
            name.into(),
            anchor,
            ScanConfig::builder().build(),
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
        // Detach A's Sub mid-Active to flip the Active burst's directive
        // to [`BurstFinish::Reap`], then re-attach B at the same
        // `(anchor, config_hash)`. The revival path must:
        //   - reuse A's Profile (same ProfileId),
        //   - leave the anchor's watch_demand at 1 (no double-bump),
        //   - emit no spurious Watch op for the anchor,
        //   - flip the directive back to `BurstFinish::ReturnToIdle`
        //     via `clear_active_reap`,
        //   - keep `anchor_claim` Held,
        //   - recompute `Profile.settle` to B's settle (NOT min-update —
        //     A is gone, B is the only live Sub),
        //   - emit `Diagnostic::ReapPendingCancelled`.
        //
        // (`fired_subs` cleanup is deliberately not asserted: the prior
        // `purge_dead_fired_subs` was functionally inert under v1's
        // workload — `emit_effects` iterates live SubIds and the dedup
        // check uses fresh keys, so stale entries never participate in
        // emission. Bound is O(1) per Profile per revival cycle.)
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
        let pid = e.subs().get(sid_a).unwrap().profile;
        let watch_demand_after_attach = e.tree.get(r).unwrap().watch_demand();
        assert_eq!(watch_demand_after_attach, 1, "anchor watch_demand from A");

        // Detach A. Profile is Active → directive flips to
        // `BurstFinish::Reap`; anchor watch unchanged.
        let _ = e.step(Input::DetachSub(sid_a), Instant::now());
        assert!(matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap),
        ));
        assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);

        // Revive with B (settle=200ms; deliberately larger than A's stale
        // 50ms so the min-update would be visibly wrong).
        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "B", Duration::from_millis(200))),
            now,
        );
        let sid_b =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid_b = e.subs().get(sid_b).unwrap().profile;

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
    }

    #[test]
    fn finish_burst_to_idle_does_not_reap_revived_profile() {
        // After revival, the in-flight burst's lifecycle continues under
        // the new Sub set. When the probe responds and the burst ends,
        // `finish_burst_to_idle` must NOT call `reap_profile` (the
        // revival cleared `reap_pending`).
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
        let pid = e.subs().get(sid_a).unwrap().profile;
        let seed_corr = attach_out
            .probe_ops
            .iter()
            .find_map(|op| match op {
                specter_core::ProbeOp::Probe { request } => Some(request.correlation()),
                specter_core::ProbeOp::Cancel { .. } => None,
            })
            .expect("attach emitted Probe");

        let _ = e.step(Input::DetachSub(sid_a), Instant::now());
        let attach_out = e.step(
            Input::AttachSub(revival_attach_req(r, "B", Duration::from_millis(50))),
            now,
        );
        let sid_b =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");

        // Drive the in-flight Seed-Verifying burst to a terminal Vanished.
        // `dispatch_seed_vanished → finalize_anchor_lost → finish_burst_to_idle`
        // would reap if `reap_pending` were still set; the revival cleared
        // it, so the Profile transitions to Idle (anchor lost) and stays.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
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

    /// Snapshot of "claim state" after running a permutation of the
    /// release quartet. The fields are exactly the bookkeeping the four
    /// release helpers clear (per-Profile flags + per-Resource
    /// `watch_demand` counters); two runs that produce the same snapshot
    /// have produced equivalent observable effects on engine state.
    /// `StepOutput.watch_ops` is also compared by the caller because
    /// the helpers emit `Unwatch` operations whose order is part of the
    /// post-`sort_for_emission` contract.
    #[derive(Debug, Eq, PartialEq)]
    struct QuartetFinalState {
        anchor_claim: AnchorClaim,
        watch_root_parent: Option<ResourceId>,
        current_is_none: bool,
        anchor_watch_demand: u32,
        parent_watch_demand: u32,
    }

    /// Build a "materialised" Profile carrying three of the four
    /// quartet claims (anchor, watch_root_parent, descendants — the
    /// descent-prefix claim is mutually exclusive with anchor per the
    /// trichotomy invariant). Returns the IDs the test body needs to
    /// inspect the post-release state.
    ///
    /// The fixture uses `attach_sub_inner` to bootstrap so every
    /// per-Resource contribution lands through the canonical
    /// `add_watch` path rather than a hand-rolled mutation that would
    /// not exercise refcount discipline. The Dir snapshot installed
    /// onto `Profile.current` after attach gives `release_descendant_claim`
    /// something to release.
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

        let req = SubAttachRequest::for_resource(
            "permtest".into(),
            anchor,
            ScanConfig::builder().build(),
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
        let pid = e.subs().get(sid).expect("Sub alive").profile;

        // Install a Dir snapshot on Profile.current so the descendant
        // release has a snapshot to take. Mirrors what
        // `dispatch_seed_ok` would write at probe completion. One
        // child leaf gives `release_descendant_claim` a non-trivial
        // diff to apply.
        let child_id = e
            .tree
            .ensure_child(anchor, "child", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(child_id, ResourceKind::File);
        let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        entries.insert(
            CompactString::from("child"),
            ChildEntry::Leaf(LeafEntry::new(
                specter_core::EntryKind::File,
                0,
                UNIX_EPOCH,
                FsIdentity {
                    inode: 0,
                    device: 0,
                },
            )),
        );
        let dir = DirSnapshot::new(
            DirMeta {
                mtime: UNIX_EPOCH,
                fs_id: FsIdentity {
                    inode: 0,
                    device: 0,
                },
            },
            0,
            entries,
        );
        e.profiles
            .get_mut(pid)
            .expect("Profile alive")
            .install_dir_current(Arc::new(dir));

        // Drop the auto-emitted Watch operations from attach — the
        // permutation test only cares about deltas across release
        // calls.
        let _ = out;

        (e, pid, anchor, parent)
    }

    /// Run the release quartet in a given index permutation and
    /// capture the post-release state.
    fn run_quartet_permutation(perm: [usize; 4]) -> QuartetFinalState {
        let (mut e, pid, anchor, parent) = build_materialised_profile_for_permutation();
        let releases: [fn(&mut Engine, ProfileId, &mut StepOutput); 4] = [
            Engine::release_descent_prefix_claim,
            Engine::release_descendant_claim,
            Engine::release_anchor_claim,
            Engine::release_watch_root_parent_claim,
        ];
        let mut out = StepOutput::default();
        for &idx in &perm {
            releases[idx](&mut e, pid, &mut out);
        }

        QuartetFinalState {
            anchor_claim: e
                .profiles
                .get(pid)
                .map_or(AnchorClaim::None, specter_core::Profile::anchor_claim),
            watch_root_parent: e.profiles.get(pid).and_then(|p| p.watch_root_parent),
            current_is_none: e.profiles.get(pid).is_none_or(|p| p.current().is_none()),
            anchor_watch_demand: e
                .tree
                .get(anchor)
                .map_or(0, specter_core::Resource::watch_demand),
            parent_watch_demand: e
                .tree
                .get(parent)
                .map_or(0, specter_core::Resource::watch_demand),
        }
    }

    /// Enumerate every permutation of `[0, 1, 2, 3]` via Heap's
    /// algorithm. 24 results.
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

    /// Pin the partial-order claim in `reap_profile`'s rustdoc: every
    /// permutation of the four release helpers yields the same
    /// observable claim state. The fixture exercises three live claims
    /// (anchor + watch_root_parent + descendants); the fourth helper
    /// (`release_descent_prefix_claim`) no-ops on the non-Pending
    /// Profile but still occupies a position in the permutation.
    #[test]
    fn release_quartet_is_permutation_invariant() {
        let perms = all_quartet_permutations();
        assert_eq!(perms.len(), 24, "Heap's algorithm enumerates 4! = 24");

        let canonical = run_quartet_permutation([0, 1, 2, 3]);
        // Final state pins the helpers actually did their work: anchor
        // released, watch_root_parent released, snapshot taken, both
        // resource watch_demand counters at zero.
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
