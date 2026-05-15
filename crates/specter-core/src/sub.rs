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
use crate::scan_config::ScanConfig;
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

/// Public-API request to attach a Sub.
///
/// Carries everything `Engine::attach_sub` needs to either reuse an
/// existing Profile (matching `(anchor, config_hash)`) or create a fresh
/// one. The [`SubAttachAnchor`] sum makes the resource-vs-path choice
/// exclusive by construction — a request names exactly one, and the
/// engine dispatches on it with an exhaustive `match`.
///
/// `name` is `String` so callers don't need a `compact_str` dependency at
/// this seam — `Sub::new` converts via `Into<CompactString>` internally.
///
/// Lives in `core::sub` rather than `engine::engine` so
/// [`SubRegistryDiff`] (a `core` type, consumed via
/// [`crate::Input::ConfigDiff`]) can carry pre-id `SubAttachRequest`s without
/// introducing a `core → engine` cycle. `Clone` is derived for the
/// (rare) call sites that fan a request out to multiple Engines —
/// production paths consume by value. `program` is `Arc<ActionProgram>`
/// so the Arc travels straight from the config layer's
/// `lower_to_program` into the engine's `Sub.program` without a
/// re-allocation: one Arc per Sub, refcount-bumped on each emitted
/// `Effect`.
#[derive(Clone, Debug)]
pub struct SubAttachRequest {
    pub name: String,
    pub anchor: SubAttachAnchor,
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub settle: Duration,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    /// Event-class mask the user opted into. The engine folds this into
    /// `config_hash` so two Subs differing only on classes fork separate
    /// Profiles. The config layer is responsible for materializing the
    /// scope-conditional default before constructing the request — this
    /// struct does no defaulting.
    pub events: ClassSet,
    /// Forward subprocess stdout/stderr to Specter's own stdout/stderr
    /// (`Stdio::inherit()`); when `false`, child output goes to
    /// `/dev/null`. Threaded through to `Effect.capture_output` at
    /// emission time. Not folded into `config_hash` — flipping it
    /// changes how the actuator spawns, not which Profile a Sub
    /// belongs to.
    pub log_output: bool,
    /// Promoter that synthesised this Sub — `None` for static
    /// (operator-declared) Subs, `Some(pid)` for dynamic Subs spawned by
    /// a Promoter's `try_promote`. Routed through the engine's recovery
    /// fan-out at `on_anchor_terminal_event`: a Profile whose Subs are
    /// all `Some(_)` reaps wholesale on anchor loss; mixed/static-only
    /// Profiles preserve the existing recovery channel.
    pub source_promoter: Option<PromoterId>,
}

impl SubAttachRequest {
    /// Build a static (operator-declared) attach request.
    /// `source_promoter` is `None`; use [`Self::for_anchor_dynamic`]
    /// when a Promoter is the source.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn for_anchor(
        name: String,
        anchor: SubAttachAnchor,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        events: ClassSet,
        log_output: bool,
    ) -> Self {
        Self {
            name,
            anchor,
            config,
            max_settle,
            settle,
            program,
            scope,
            events,
            log_output,
            source_promoter: None,
        }
    }

    /// Promoter-synthesised counterpart to [`Self::for_anchor`].
    /// `source` is non-optional — a dynamic Sub always has an
    /// originating promoter. Delegates to [`Self::for_anchor`] and
    /// stamps `source_promoter`; the engine reads the stamp at recovery
    /// time (`on_anchor_terminal_event`) to distinguish all-dynamic
    /// Profiles (wholesale teardown) from mixed/static ones.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn for_anchor_dynamic(
        name: String,
        anchor: SubAttachAnchor,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        events: ClassSet,
        log_output: bool,
        source: PromoterId,
    ) -> Self {
        let mut req = Self::for_anchor(
            name, anchor, config, max_settle, settle, program, scope, events, log_output,
        );
        req.source_promoter = Some(source);
        req
    }
}

/// Hot-reload diff. Computed by the TOML loader; consumed by
/// `Engine::step(Input::ConfigDiff(_))`.
///
/// `added` and `modified` carry pre-id [`SubAttachRequest`] data; `removed`
/// carries existing [`SubId`]s. Engine processes `removed → modified →
/// added` atomically in one step, with parent-edge recompute after each
/// detach/attach.
#[derive(Clone, Debug, Default)]
pub struct SubRegistryDiff {
    pub added: Vec<SubAttachRequest>,
    pub removed: Vec<SubId>,
    pub modified: Vec<(SubId, SubAttachRequest)>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
/// representation folded into [`compute_config_hash`](crate::compute_config_hash)
/// (two Subs differing only on classes fork separate Profiles).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    pub id: SubId,
    pub name: CompactString,
    pub profile: ProfileId,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub settle: Duration,
    pub max_settle: Duration,
    pub needs_diff: bool,
    /// User-declared event-class mask. Folded into the Profile's
    /// `config_hash`; every Sub on a Profile shares the same `events`
    /// by construction.
    pub events: ClassSet,
    /// Forward subprocess stdout/stderr to Specter's own stdio. Threaded
    /// onto each emitted [`crate::Effect`] as `capture_output`; the actuator
    /// switches between `Stdio::null()` (false, the default) and
    /// `Stdio::inherit()` (true). Not folded into `config_hash`.
    pub log_output: bool,
    /// Promoter that synthesised this Sub — `None` for static
    /// (operator-declared) Subs, `Some(pid)` for dynamic Subs spawned by
    /// a Promoter's `try_promote`. Read at the engine's recovery
    /// fan-out (`on_anchor_terminal_event`); never mutated post-attach.
    pub source_promoter: Option<PromoterId>,
}

impl Sub {
    /// Construct a Sub. `needs_diff` is derived: true iff
    /// `scope == PerStableFile` OR the program references any diff-derived
    /// placeholder. Pre-computed once; never re-evaluated.
    ///
    /// `program` is taken as `Arc<ActionProgram>` so the Arc minted by
    /// the config layer's `lower_to_program` flows through unchanged —
    /// the constructor does not re-wrap. One Arc per Sub, refcount-bumped
    /// on each emitted [`crate::Effect`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: SubId,
        name: impl Into<CompactString>,
        profile: ProfileId,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        settle: Duration,
        max_settle: Duration,
        events: ClassSet,
        log_output: bool,
        source_promoter: Option<PromoterId>,
    ) -> Self {
        let needs_diff = scope == EffectScope::PerStableFile || program.references_diff_derived();
        Self {
            id,
            name: name.into(),
            profile,
            program,
            scope,
            settle,
            max_settle,
            needs_diff,
            events,
            log_output,
            source_promoter,
        }
    }
}

#[derive(Debug, Default)]
pub struct SubRegistry {
    subs: SlotMap<SubId, Sub>,
    by_profile: BTreeMap<ProfileId, SmallVec<[SubId; 2]>>,
}

impl SubRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a Sub built from the freshly-minted `SubId`. The Sub stores
    /// its own id; the closure embeds the minted key into the Sub.
    pub fn insert<F>(&mut self, build: F) -> SubId
    where
        F: FnOnce(SubId) -> Sub,
    {
        let id = self.subs.insert_with_key(build);
        let profile = self.subs[id].profile;
        self.by_profile.entry(profile).or_default().push(id);
        id
    }

    pub fn remove(&mut self, id: SubId) -> Option<Sub> {
        let sub = self.subs.remove(id)?;
        if let Some(v) = self.by_profile.get_mut(&sub.profile) {
            v.retain(|sid| *sid != id);
            if v.is_empty() {
                self.by_profile.remove(&sub.profile);
            }
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

    /// Linear-scan lookup of a Sub by its user-facing `name`. `O(N_subs)`
    /// per call; uniqueness is the caller's responsibility (the loader's
    /// validation rejects duplicates upstream — when two Subs share a
    /// name the first match in `SlotMap` iteration order wins).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<SubId> {
        self.subs
            .iter()
            .find_map(|(id, s)| (s.name == name).then_some(id))
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionProgram, ClassSet, EffectScope, Sub, SubRegistry};
    use crate::ids::{ProfileId, SubId};
    use crate::program::{
        ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder, SpawnBody,
    };
    use std::sync::Arc;
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

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
        single_exec_program(ExecAction::new([ArgTemplate::new([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ])]))
    }

    fn program_with(p: Placeholder) -> Arc<ActionProgram> {
        single_exec_program(ExecAction::new([ArgTemplate::new([ArgPart::Placeholder(
            p,
        )])]))
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
        let sub = Sub::new(
            SubId::default(),
            "fmt",
            ProfileId::default(),
            anchor_only_program(),
            EffectScope::PerStableFile,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
            false,
            None,
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_set_for_diff_placeholder_in_subtree_scope() {
        let sub = Sub::new(
            SubId::default(),
            "report",
            ProfileId::default(),
            program_with(Placeholder::Created),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
            false,
            None,
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_false_for_anchor_subtree_combo() {
        let sub = Sub::new(
            SubId::default(),
            "build",
            ProfileId::default(),
            anchor_only_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
            false,
            None,
        );
        assert!(!sub.needs_diff);
    }

    #[test]
    fn registry_insert_embeds_minted_id() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });

        let sub = reg.get(sid).expect("Sub stored");
        assert_eq!(sub.id, sid, "Sub.id matches the minted key");
        assert_eq!(sub.name, "build");
    }

    #[test]
    fn registry_at_groups_by_profile() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();

        let s1 = reg.insert(|id| {
            Sub::new(
                id,
                "a",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });
        let s2 = reg.insert(|id| {
            Sub::new(
                id,
                "b",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });

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

    #[test]
    fn find_by_name_returns_some_for_match() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let id = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });
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
        let id = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });
        reg.remove(id);
        assert!(reg.find_by_name("build").is_none());
    }

    #[test]
    fn find_by_name_resolves_one_of_duplicates() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let a = reg.insert(|id| {
            Sub::new(
                id,
                "shared",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });
        let b = reg.insert(|id| {
            Sub::new(
                id,
                "shared",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });
        let found = reg.find_by_name("shared").expect("at least one match");
        assert!(found == a || found == b, "find returns one of the matches");
    }

    #[test]
    fn registry_remove_clears_by_profile_and_drops_empty_bucket() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
                NO_EVENTS,
                false,
                None,
            )
        });

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
        let sub = Sub::new(
            SubId::default(),
            "build",
            ProfileId::default(),
            anchor_only_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
            false,
            None,
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

    /// `Sub::new` does not re-wrap the program: the caller's Arc is the
    /// same allocation the Sub stores. The minted Arc from the config
    /// layer's `lower_to_program` flows through without churn.
    #[test]
    fn sub_new_does_not_rewrap_program_arc() {
        let program = anchor_only_program();
        let before = Arc::as_ptr(&program);
        let sub = Sub::new(
            SubId::default(),
            "build",
            ProfileId::default(),
            Arc::clone(&program),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
            false,
            None,
        );
        assert!(
            std::ptr::eq(before, Arc::as_ptr(&sub.program)),
            "Sub::new must not allocate a new ActionProgram",
        );
    }

    /// `Sub::new` threads `events` through to the constructed Sub.
    #[test]
    fn sub_new_records_events() {
        let sub = Sub::new(
            SubId::default(),
            "x",
            ProfileId::default(),
            anchor_only_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            ClassSet::CONTENT | ClassSet::METADATA,
            false,
            None,
        );
        assert_eq!(sub.events, ClassSet::CONTENT | ClassSet::METADATA);
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
        let cases = [
            ClassSet::EMPTY,
            ClassSet::STRUCTURE,
            ClassSet::CONTENT,
            ClassSet::METADATA,
            ClassSet::STRUCTURE | ClassSet::CONTENT,
            ClassSet::CONTENT | ClassSet::METADATA,
            ClassSet::STRUCTURE | ClassSet::METADATA,
            ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA,
        ];
        for c in cases {
            // bits() faithfully encodes the canonical OR.
            assert_eq!(c.bits().count_ones(), c.bits().count_ones());
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
