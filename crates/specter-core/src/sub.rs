//! Subscription and `EffectScope`.
//!
//! A `Sub` is a *reaction declaration*: it names what to watch and what
//! program should run when the watched tree settles. The program is a
//! CFG-shaped op IR — [`ActionProgram`] (see [`crate::program`]) holds a
//! `Box<[ProgramOp]>` walked by a `u32` cursor at the actuator. The
//! surface syntax (validation-side `Action` tree, lives in
//! `specter-config`) folds into the program at validation time; the
//! engine and actuator see only the lowered form.
//!
//! `Sub.needs_diff` is derived at construction: true iff the `EffectScope`
//! is `PerStableFile` *or* the program references any diff-derived
//! placeholder (`Created`/`Deleted`/`Modified`/`RenamedFrom`/`RenamedTo`).
//!
//! v1 surface is argv-only — no shell variant.

use crate::ids::{ProfileId, PromoterId, ResourceId, SubId};
use crate::program::ActionProgram;
use crate::scan_config::{ProfileIdentity, ScanConfig};
use compact_str::CompactString;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Where a Sub anchors.
///
/// `Resource` names a slot the caller has already materialised; the
/// engine trusts it after an O(1) liveness check. `Path` names an
/// absolute path the engine resolves at attach time — immediate when
/// every component already exists, otherwise pending descent until the
/// anchor materialises and a Seed burst establishes the baseline.
#[derive(Clone, Debug)]
pub enum SubAttachAnchor {
    Resource(ResourceId),
    Path(PathBuf),
}

/// The per-Sub reaction declaration: everything that is *not* Profile
/// identity or the anchor.
///
/// `name` is `CompactString`, moved end to end: `SubSpec.name`
/// (config) is already `CompactString`, so the attach request carries
/// it without a `String` round-trip and [`Sub::from_request`] moves it
/// into `Sub.name` unchanged. `program` is `Arc<ActionProgram>` so the
/// Arc minted by the config layer's `lower_to_program` flows through to
/// `Sub.program` without a re-allocation.
///
/// Carries no Profile-identity field, so a Sub cannot express (or
/// leak) a sibling Profile's config/mask — demonstrated
/// unrepresentable:
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
    /// Per-Sub debounce floor — min-folded across the Profile's Subs by
    /// the engine's `recompute_profile_settle`. Distinct from
    /// `max_settle`, which is identity (folds into `config_hash`) and
    /// lives on [`ProfileIdentity`].
    pub settle: Duration,
    /// Forward subprocess stdout/stderr to Specter's own stdio
    /// (`Stdio::inherit()`); `false` routes child output to
    /// `/dev/null`. Threaded to `Effect.capture_output`; not identity.
    pub log_output: bool,
    /// Promoter that synthesised this Sub — `None` for static
    /// (operator-declared) Subs, `Some(pid)` for dynamic Subs. Read at
    /// the engine's recovery fan-out (`on_anchor_terminal_event`) to
    /// distinguish all-dynamic Profiles (wholesale teardown) from
    /// mixed/static ones.
    pub source_promoter: Option<PromoterId>,
}

/// Public-API request to attach a Sub.
///
/// Three orthogonal parts: *where* ([`SubAttachAnchor`]), *which
/// Profile* ([`ProfileIdentity`]), *what the Sub does* ([`SubParams`]).
/// Identity decides Profile partitioning; the anchor resolves
/// separately (not in the hash preimage); params are per-Sub. The
/// split makes a Sub leaking a sibling's identity field structurally
/// unrepresentable.
///
/// Lives in `core::sub` (not `engine`) so [`SubRegistryDiff`] can carry
/// pre-id requests via [`crate::Input::ConfigDiff`] without a
/// `core → engine` cycle. `Clone` serves the rare multi-Engine
/// fan-out; production consumes by value.
#[derive(Clone, Debug)]
pub struct SubAttachRequest {
    pub anchor: SubAttachAnchor,
    pub identity: ProfileIdentity,
    pub params: SubParams,
}

impl SubAttachRequest {
    /// Canonical constructor. [`Self::for_anchor`] is the
    /// flat-argument ergonomic over this for the config layer and
    /// tests; the engine's `try_promote` builds dynamic
    /// (Promoter-synthesised) Subs through this directly.
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

    /// Build a static (operator-declared) attach request —
    /// `source_promoter` is `None`. Dynamic (Promoter-synthesised)
    /// Subs are built by the engine's `try_promote` via
    /// [`Self::from_parts`] directly.
    #[must_use]
    pub const fn for_anchor(
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
            ProfileIdentity {
                config,
                max_settle,
                events,
            },
            SubParams {
                name,
                program,
                scope,
                settle,
                log_output,
                source_promoter: None,
            },
        )
    }
}

/// Hot-reload diff. Computed by the TOML loader; consumed by
/// `Engine::step(Input::ConfigDiff(_))`.
///
/// Name-keyed: `removed` carries operator watch names; `added` and
/// `modified` carry pre-id [`SubAttachRequest`]s (the name lives
/// inside `params.name`). The engine resolves name → [`SubId`] at
/// apply time through its own authoritative `by_name` index —
/// identity resolution is a registry-owner operation, not the
/// loader's. Engine processes `removed → modified → added` atomically
/// in one step, with parent-edge recompute after each detach/attach.
#[derive(Clone, Debug, Default)]
pub struct SubRegistryDiff {
    pub added: Vec<SubAttachRequest>,
    pub removed: Vec<CompactString>,
    pub modified: Vec<SubAttachRequest>,
}

impl SubRegistryDiff {
    /// True iff every bucket is empty — the "no Sub-side changes"
    /// short-circuit the reload pipeline tests before handing the diff
    /// to the engine. Single point of truth: future bucket additions
    /// (the validate-then-act split into `modified_identity` /
    /// `modified_params`) extend this method, never the callers.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum EffectScope {
    #[default]
    SubtreeRoot,
    PerStableFile,
}

/// User-facing event-class set on a [`Sub`] — the surface of the event
/// filtering primitive.
///
/// A class set names *what kinds of change* a watch cares about, in
/// backend-agnostic vocabulary. The kqueue translator (sensor side) is the
/// only place that maps the set onto `NOTE_*` fflags; inotify gets a
/// sibling translator. Engine and core never see backend bits.
///
/// Three classes:
/// - **STRUCTURE** — directory entries added / removed / renamed (Dir-only).
/// - **CONTENT**   — file bytes changed *or* file identity changed
///   (delete / rename / revoke). File-only.
/// - **METADATA**  — attribute change (perms, owner, link count,
///   timestamps). Both Files and Dirs.
///
/// The set is backed by a `u8` bitmask — `bits()` is the canonical
/// representation folded into the Profile config hash via
/// [`ProfileIdentity::config_hash`] (two Subs differing only on classes
/// fork separate Profiles).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct ClassSet(u8);

impl ClassSet {
    pub const EMPTY: Self = Self(0);
    pub const STRUCTURE: Self = Self(1 << 0);
    pub const CONTENT: Self = Self(1 << 1);
    pub const METADATA: Self = Self(1 << 2);

    /// Default for [`EffectScope::SubtreeRoot`] — STRUCTURE | CONTENT.
    /// Closes E2E #3 (in-place edits surface as events through the per-file
    /// FDs implied by CONTENT).
    pub const DEFAULT_SUBTREE_ROOT: Self = Self(0b011);

    /// Default for [`EffectScope::PerStableFile`] — CONTENT | METADATA.
    /// The user opted into per-file granularity; metadata is part of
    /// "this file's state changed".
    pub const DEFAULT_PER_FILE: Self = Self(0b110);

    /// True iff every bit in `other` is set in `self` AND `other` is
    /// non-empty.
    ///
    /// The `other.0 != 0` clause sidesteps the `bitflags`-crate footgun
    /// where `contains(EMPTY) == true` for every set: at every call site
    /// the question being asked is "do we hold this *specific*, non-empty
    /// class?", and reporting "yes" for `EMPTY` would silently green-light
    /// a no-class translator branch.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0 && other.0 != 0
    }

    /// True iff `self` and `other` share at least one bit. `EMPTY`
    /// intersects nothing (including itself).
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

#[derive(Debug)]
pub struct Sub {
    pub name: CompactString,
    pub profile: ProfileId,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    /// Per-Sub debounce floor. `max_settle` and `events` are *not*
    /// stored here — they are Profile identity (fold into `config_hash`,
    /// invariant for the Profile's lifetime); read them off the Profile.
    pub settle: Duration,
    pub needs_diff: bool,
    /// Forward subprocess stdout/stderr to Specter's own stdio. Threaded
    /// onto each emitted [`crate::Effect`] as `capture_output`; the actuator
    /// switches between `Stdio::null()` (false, the default) and
    /// `Stdio::inherit()` (true).
    pub log_output: bool,
    /// Promoter that synthesised this Sub — `None` for static
    /// (operator-declared) Subs, `Some(pid)` for dynamic Subs spawned by
    /// a Promoter's `try_promote`. Read at the engine's recovery
    /// fan-out (`on_anchor_terminal_event`); never mutated post-attach.
    pub source_promoter: Option<PromoterId>,
    /// The per-Sub Effect fire history: `true` once this Sub has
    /// emitted at least one Effect. Sole load-bearing reader is the B1
    /// dedup suppress (`!forced && nothing_changed && has_fired` — a
    /// never-fired Sub is its own first emission); the per-Profile
    /// SeedDrift filter ([`SubRegistry::fired_in`]) and the
    /// recovery-drift short-circuit ([`SubRegistry::any_fired`]) read
    /// it Profile-wide.
    ///
    /// Lives here, not on [`crate::Profile`]: a Sub attaches to exactly
    /// one Profile, so "has this Sub fired?" is per-Sub, not a
    /// per-Profile join table. Its lifetime *is* the slotmap entry's —
    /// detach ([`SubRegistry::remove`]) drops it with the Sub, so a
    /// detached or hot-reload-modified reaction can never re-fire on a
    /// later drift verdict and a revived Profile's fresh Subs start
    /// `false` structurally (no purge step to forget). Mutated only
    /// through [`SubRegistry::mark_fired`] / [`SubRegistry::clear_fired`]
    /// (the registry holds the sole `&mut Sub`). The invariant is the
    /// weakest tier in the engine — drift self-corrects on the next
    /// real change, no refcount/state-machine corruption — so a plain
    /// `bool` carries it with no edge-method ceremony.
    pub has_fired: bool,
}

impl Sub {
    /// Construct a Sub from its [`ProfileId`] and the per-Sub
    /// [`SubParams`]. `needs_diff` is derived: true iff
    /// `scope == PerStableFile` OR the program references any
    /// diff-derived placeholder. Pre-computed once; never re-evaluated.
    ///
    /// The slotmap key is the Sub's identity authority — there is no
    /// `id` field. `params.name` (`CompactString`) and
    /// `params.program`'s Arc both move through unchanged (no
    /// re-allocation, no Arc re-wrap); one Arc per Sub, refcount-bumped
    /// on each emitted [`crate::Effect`].
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
            source_promoter: params.source_promoter,
            has_fired: false,
        }
    }
}

/// Slotmap-backed Sub store with two secondary indices.
///
/// - `by_profile` groups Subs by `ProfileId` (insertion order within a
///   Profile).
/// - `by_name` resolves an operator-facing watch name to its `SubId`.
///   **Static Subs only** (`source_promoter.is_none()`): dynamic Subs
///   carry synthesised `<promoter>@<path>` names the config diff never
///   references, and indexing them would let a synthesised name alias
///   an operator watch name. Dynamic Subs are reached through the
///   `by_profile` index and their `source_promoter` tag (the engine's
///   derived dedup gate and Promoter-reap scan), never `by_name`. The
///   index is load-bearing — hot-reload resolves every
///   `removed`/`modified` name to an id through [`Self::find_by_name`]
///   (O(log N)).
///
/// `by_name` mirrors the slotmap entry's lifetime for static Subs:
/// [`Self::insert`] populates it, [`Self::remove`] clears it
/// id-checked. The `insert` `debug_assert!` is the dev/CI signal for a
/// duplicate static name; the hot-reload diff invariant
/// (`added = new ∖ old`, where `old` is the applied config and so
/// equals `by_name`'s contents) makes a duplicate insert unreachable
/// in correct operation. A release-mode breach is contained by the
/// id-checked `remove`: the *mapping* stays 1:1; only a hypothetical
/// orphaned slotmap entry (never the wrong name→id edge) could
/// survive.
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

    /// Insert a Sub; the returned slotmap [`SubId`] is its identity
    /// authority (the Sub carries no `id` field). Both secondary
    /// indices update in lockstep: `by_profile` always; `by_name` only
    /// for static Subs (`source_promoter.is_none()` — see the struct
    /// rustdoc for why dynamic Subs are excluded).
    ///
    /// The `debug_assert!` fires on a duplicate static name — the
    /// dev/CI signal only. The hot-reload diff invariant makes a
    /// duplicate insert unreachable in correct operation; a
    /// release-mode breach is contained by the id-checked
    /// [`Self::remove`] (the mapping stays consistent).
    pub fn insert(&mut self, sub: Sub) -> SubId {
        let profile = sub.profile;
        let static_name = sub.source_promoter.is_none().then(|| sub.name.clone());
        let id = self.subs.insert(sub);
        self.by_profile.entry(profile).or_default().push(id);
        if let Some(name) = static_name {
            debug_assert!(
                !self.by_name.contains_key(&name),
                "duplicate static Sub name escaped config validation: {name:?}",
            );
            self.by_name.insert(name, id);
        }
        id
    }

    /// Remove a Sub by id, returning the owned value. Clears both
    /// secondary indices. The `by_name` clear is **id-checked** — the
    /// entry drops only if it still points at `id`, so removing a
    /// duplicate-name escape's shadowed id (a release-mode diff bug)
    /// cannot clobber the live id's mapping. Returns `None` for a
    /// stale id.
    pub fn remove(&mut self, id: SubId) -> Option<Sub> {
        let sub = self.subs.remove(id)?;
        if let Some(v) = self.by_profile.get_mut(&sub.profile) {
            v.retain(|sid| *sid != id);
            if v.is_empty() {
                self.by_profile.remove(&sub.profile);
            }
        }
        if sub.source_promoter.is_none() && self.by_name.get(&sub.name) == Some(&id) {
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

    /// Resolve a static Sub's user-facing `name` to its [`SubId`] in
    /// O(log N) via `by_name`. `None` if no static Sub holds `name`
    /// (dynamic Subs are not indexed — see the struct rustdoc).
    ///
    /// Load-bearing: the engine's hot-reload shim resolves every
    /// `removed`/`modified` name through here. Config validation
    /// rejects duplicate operator names upstream and [`Self::insert`]
    /// `debug_assert!`s the same invariant, so the mapping is 1:1 for
    /// every name the diff can reference.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<SubId> {
        self.by_name.get(name).copied()
    }

    /// Record that `sub` emitted an Effect — the B1-dedup / SeedDrift
    /// fire-history mark, written by `emit_effects`' SubtreeRoot arm on
    /// a successful push. Idempotent. A stale `SubId` (the Sub detached
    /// between the emit decision and here) is a silent no-op: the flag
    /// already died with the slotmap entry.
    pub fn mark_fired(&mut self, sub: SubId) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.has_fired = true;
        }
    }

    /// Clear `sub`'s fire history — the `EffectComplete::Failed` clear.
    /// A failed Effect produced no observation worth deduplicating
    /// against, so the next stable verdict at this Sub must re-fire
    /// even on an unchanged tree. No-op on a stale `SubId` (already
    /// detached ⇒ its history is already gone).
    pub fn clear_fired(&mut self, sub: SubId) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.has_fired = false;
        }
    }

    /// Whether any Sub on `profile` has fired — the fast
    /// `seed_drift_observed` short-circuit ("never fired ⇒ no prior
    /// emission to re-fire on recovery"). Replaces the prior
    /// per-Profile `fired_is_empty` negation.
    #[must_use]
    pub fn any_fired(&self, profile: ProfileId) -> bool {
        self.at(profile)
            .iter()
            .any(|sid| self.subs.get(*sid).is_some_and(|s| s.has_fired))
    }

    /// The Subs on `profile` that have fired — the SeedDrift
    /// conservative-recovery fire-filter basis.
    ///
    /// **Order is membership only.** The caller filters with
    /// `.contains`; the observable Effect order is established globally
    /// by [`crate::StepOutput::sort_for_emission`] (the load-bearing
    /// `(SubId, ResourceId)` canonicalisation every step applies before
    /// returning), so the insertion order `at` yields here is
    /// sufficient and deterministic — there is no per-call re-sort to
    /// justify.
    #[must_use]
    pub fn fired_in(&self, profile: ProfileId) -> SmallVec<[SubId; 2]> {
        self.at(profile)
            .iter()
            .copied()
            .filter(|sid| self.subs.get(*sid).is_some_and(|s| s.has_fired))
            .collect()
    }

    /// Whether `profile` has at least one attached `PerStableFile`
    /// Sub — the scope test behind the per-file recovery-drop signal.
    ///
    /// **Must not be collapsed into [`crate::Profile::has_per_file_fds`].**
    /// That predicate is events-mask derived (`CONTENT | METADATA`
    /// present) and a `SubtreeRoot` Sub watching `CONTENT` sets it
    /// just as much as a `PerStableFile` Sub does — it is *necessary*
    /// for per-file FDs but *not sufficient* for "this Profile carries
    /// a per-file-*scoped* reaction". Swapping this scan for
    /// `has_per_file_fds` would false-positive the recovery-drop
    /// diagnostic on Subtree-only Profiles that happen to watch
    /// content. The `scope` field is the only sound witness; the scan
    /// stays.
    #[must_use]
    pub fn has_per_stable_file_sub(&self, profile: ProfileId) -> bool {
        self.at(profile).iter().any(|sid| {
            self.subs
                .get(*sid)
                .is_some_and(|s| s.scope == EffectScope::PerStableFile)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionProgram, EffectScope, Sub, SubParams, SubRegistry, SubRegistryDiff};
    use crate::ids::{ProfileId, PromoterId, SubId};
    use crate::program::{
        ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder, SpawnBody,
    };
    use compact_str::CompactString;
    use std::sync::Arc;
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);

    /// Build a one-op program holding a single Exec body. Equivalent to
    /// the lowering of a single `[[watch.actions]] exec = [...]` entry.
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
        // The full non-diff-derived set: every single-value placeholder
        // PLUS `Excluded` (multi-value but not diff-derived). Including
        // `Excluded` here is the load-bearing assertion of the
        // `is_multivalue` / `is_diff_derived` split — using the
        // `Excluded` variant in a template must NOT ratchet
        // `Sub.needs_diff` true.
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
            SubParams {
                name: "fmt".into(),
                program: anchor_only_program(),
                scope: EffectScope::PerStableFile,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_set_for_diff_placeholder_in_subtree_scope() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams {
                name: "report".into(),
                program: program_with(Placeholder::Created),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_false_for_anchor_subtree_combo() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        );
        assert!(!sub.needs_diff);
    }

    /// A freshly built Sub starts with no fire history — the B1-dedup
    /// / SeedDrift baseline. Relocated from the deleted per-Profile
    /// `new_profile_initialises_fired_subs_empty`: the history now
    /// lives per-Sub, so the "starts empty" contract is asserted on
    /// the Sub, not the Profile.
    #[test]
    fn fresh_sub_starts_unfired() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        );
        assert!(!sub.has_fired, "fresh Sub: no prior Effect emission");
    }

    #[test]
    fn registry_at_groups_by_profile() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();

        let s1 = reg.insert(Sub::from_request(
            pid,
            SubParams {
                name: "a".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        ));
        let s2 = reg.insert(Sub::from_request(
            pid,
            SubParams {
                name: "b".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
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

    /// After a multi-insert/remove sequence, every key `iter()` yields
    /// re-looks-up via `get` to the same Sub, and `at(profile)` equals
    /// the live key set. The slotmap key is the sole identity authority
    /// (a `Sub` carries no `id`) — this replaces the removed
    /// `Sub.id == minted key` test.
    #[test]
    fn registry_iter_keys_round_trip_through_get() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |name: &str| {
            Sub::from_request(
                pid,
                SubParams {
                    name: name.into(),
                    program: anchor_only_program(),
                    scope: EffectScope::SubtreeRoot,
                    settle: SETTLE,
                    log_output: false,
                    source_promoter: None,
                },
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
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
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
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        ));
        reg.remove(id);
        assert!(reg.find_by_name("build").is_none());
    }

    /// `by_name` indexes static Subs only. A dynamic Sub
    /// (`source_promoter = Some(_)`) is reachable by id but never via
    /// `find_by_name`, even when its synthesised name collides with a
    /// static watch name — structurally preventing a dynamic name from
    /// aliasing an operator watch.
    #[test]
    fn by_name_indexes_static_subs_only() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |source: Option<PromoterId>| {
            Sub::from_request(
                pid,
                SubParams {
                    name: "shared".into(),
                    program: anchor_only_program(),
                    scope: EffectScope::SubtreeRoot,
                    settle: SETTLE,
                    log_output: false,
                    source_promoter: source,
                },
            )
        };
        let dynamic = reg.insert(mk(Some(PromoterId::default())));
        assert!(
            reg.find_by_name("shared").is_none(),
            "dynamic Sub is not indexed by name",
        );
        let static_id = reg.insert(mk(None));
        assert_eq!(
            reg.find_by_name("shared"),
            Some(static_id),
            "static Sub resolves; the colliding dynamic name stays invisible",
        );
        assert!(
            reg.get(dynamic).is_some(),
            "dynamic Sub remains reachable by id",
        );
    }

    /// `remove` clears `by_name` id-checked: removing the dynamic Sub
    /// whose name collides with a live static Sub must not drop the
    /// static mapping (the dynamic side never owned the entry).
    /// Removing the static Sub then clears it.
    #[test]
    fn remove_is_id_checked_against_by_name() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |source: Option<PromoterId>| {
            Sub::from_request(
                pid,
                SubParams {
                    name: "x".into(),
                    program: anchor_only_program(),
                    scope: EffectScope::SubtreeRoot,
                    settle: SETTLE,
                    log_output: false,
                    source_promoter: source,
                },
            )
        };
        let dynamic = reg.insert(mk(Some(PromoterId::default())));
        let static_id = reg.insert(mk(None));
        assert_eq!(reg.find_by_name("x"), Some(static_id));

        reg.remove(dynamic).expect("dynamic removed");
        assert_eq!(
            reg.find_by_name("x"),
            Some(static_id),
            "removing the dynamic twin leaves the static mapping intact",
        );

        reg.remove(static_id).expect("static removed");
        assert!(
            reg.find_by_name("x").is_none(),
            "removing the static Sub clears its by_name entry",
        );
    }

    #[test]
    fn registry_remove_clears_by_profile_and_drops_empty_bucket() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        ));

        let removed = reg.remove(sid);
        assert!(removed.is_some());
        assert!(reg.get(sid).is_none());
        assert!(reg.at(pid).is_empty());
        assert_eq!(reg.len(), 0);
    }

    /// `Sub.program` is reference-counted: cloning the field bumps the
    /// strong count without copying the inner [`ActionProgram`].
    #[test]
    fn sub_program_is_arc_wrapped() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams {
                name: "build".into(),
                program: anchor_only_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
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

    /// `Sub::from_request` does not re-wrap the program: the caller's Arc
    /// is the same allocation the Sub stores. The minted Arc from the
    /// config layer's `lower_to_program` flows through without churn.
    #[test]
    fn sub_new_does_not_rewrap_program_arc() {
        let program = anchor_only_program();
        let before = Arc::as_ptr(&program);
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams {
                name: "build".into(),
                program: Arc::clone(&program),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
        );
        assert!(
            std::ptr::eq(before, Arc::as_ptr(&sub.program)),
            "Sub::from_request must not allocate a new ActionProgram",
        );
    }

    /// Diff is plain data — pins the `Default` shape and the
    /// [`SubRegistryDiff::is_empty`] contract in one place. A populated
    /// bucket flips the predicate to `false`, so the AND-over-buckets
    /// impl stays load-bearing under future field growth.
    #[test]
    fn sub_registry_diff_is_empty_predicate() {
        let mut d = SubRegistryDiff::default();
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
        assert!(d.is_empty(), "default is empty");
        d.removed.push(CompactString::from("a"));
        assert!(!d.is_empty(), "populated bucket flips is_empty");
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
        // All four named values pairwise distinct: each occupies its own
        // bit position (verifies the constants haven't drifted).
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

    /// `contains(EMPTY)` returns `false` — guards against the bitflags
    /// footgun where `contains(EMPTY) == true` for every set.
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
        // Each case pairs a `ClassSet` with an *independently spelled*
        // bitmask. Asserting the exact `u8` (not popcount) catches a
        // bit-position swap in the constants or `BitOr` — an
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
            "subtree-root default must include STRUCTURE+CONTENT (E2E #3 closure)"
        );
        assert_eq!(
            ClassSet::DEFAULT_PER_FILE,
            ClassSet::CONTENT | ClassSet::METADATA,
            "per-stable-file default must include CONTENT+METADATA"
        );
    }
}
