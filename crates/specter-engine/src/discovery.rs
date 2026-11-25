//! Discovery — what a stable verdict on a `MatchChain` Profile does: the Subs each of its templates
//! minted are made to match what the certified snapshot lists.
//!
//! A discovery Profile's Subs are templates, each carrying a [`specter_core::MintSpec`]. It fires
//! attachments and detachments, never Effects, so nothing here writes burst state or meets the
//! Draining gate: the burst leaves through the ordinary silent seal. The Subs a template mints come
//! and go from here alone — [`Engine::reconcile_matches`] as the tree moves,
//! [`Engine::reap_orphans`] when the template leaves.
//!
//! A pass reads the post-graft `current` once, beside the Tree and the baseline, into a
//! [`MatchSet`] — the matches `T`, each at its slot — and holds it against each template's minted
//! set `M` ([`specter_core::SubRegistry::minted`]): it mints `T ∖ M`, reaps `M ∖ T` where a listing
//! answers for it, and closes. It reads the whole set, not the graft's diff, which names only what
//! the tree did — a template that joins over an unchanged tree owes mints no diff carries — so a
//! cold Seed's first enumeration, a Standard burst, a recovery and a forced ceiling are one
//! idempotent reconciliation.
//!
//! A miss enters as its template's miss ([`specter_core::MintState::miss`]) in a [`Branch`] that
//! stood, and as an enumeration in one that opened. The rule reads the template and the baseline,
//! never the burst's intent, so a reseed after an overflow or a recovery still fires for a match that
//! appeared in the blind window: `M` outlives anchor loss, overflow and baseline clears. The baseline
//! does not outlive an anchor loss, so a branch that opened across one enters by the template's miss.
//!
//! Membership is slot identity: `(parent, segment)` survives delete-and-recreate, so a match
//! replaced in place stays in `M ∩ T` and keeps its Sub — id, fire history, dedup identity — while
//! its own anchor loss drives the recovery fire.

use crate::Engine;
use compact_str::CompactString;
use smallvec::SmallVec;
use specter_core::{
    ChildEntry, DetachReason, Diagnostic, DirChild, DirSnapshot, EntryKind, MintEntry,
    MintNarration, NonEmptyPathSet, PriorDir, ProfileId, Reaction, ResourceId, ResourceKind,
    ResourceRole, StepOutput, Sub, SubId, Tree,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// A template whose minted set holds more Subs than this makes
/// [`Diagnostic::DiscoveryFanoutThreshold`], once in its lifetime: the pattern is matching more
/// targets than typical — likely a too-broad pattern.
const FANOUT_WARNING_THRESHOLD: usize = 1000;

/// How the branch a match sits in stood in the baseline the pass reconciles against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Branch {
    /// The baseline read the branch, or held nothing of it: a match it lacks appeared under watch.
    Standing,
    /// A chain directory above the match is this same directory, recorded denied in the baseline:
    /// the match is first seen now that its branch opened, and may predate the watch.
    Opened,
}

/// A match a watch can anchor: its slot, the kind its listing gives, how its branch stood.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Match {
    slot: ResourceId,
    kind: ResourceKind,
    branch: Branch,
}

/// A match no watch can anchor ([`Diagnostic::DiscoveryUnsupportedAnchorKind`]), kept as its
/// directory's path and its name: it may have no slot, and the two are joined only for the line a
/// template makes once.
struct Unsupported {
    dir: Arc<Path>,
    name: CompactString,
    kind: EntryKind,
}

/// What a certified snapshot says of a pattern, read beside the Tree and the baseline.
#[derive(Default)]
struct MatchSet {
    /// The matches a watch can anchor, in path order: the order a pass mints in.
    matched: Vec<Match>,
    /// Their slots: the set a minted Sub is held against.
    slots: BTreeSet<ResourceId>,
    /// The slots of the chain directories the snapshot holds denied, where the Tree holds one.
    denied: SmallVec<[ResourceId; 2]>,
    /// The first match no watch can anchor, in path order.
    unsupported: Option<Unsupported>,
}

impl MatchSet {
    /// Walk `now`, the Tree under its slot `anchor` and `before` — what the baseline holds for
    /// `now`'s root — in lockstep, down to the terminus `terminus_depth` levels below the anchor
    /// (at least one, by the pattern's parser). Each chain directory resolves to its slot once:
    /// found where the graft made it, made where a reap took it since. The walk writes no kind; a
    /// mint kinds its own terminus.
    fn resolve(
        tree: &mut Tree,
        anchor: ResourceId,
        now: &DirSnapshot,
        before: PriorDir<'_>,
        terminus_depth: u32,
    ) -> Self {
        let mut set = Self::default();
        set.resolve_level(tree, anchor, now, before, Branch::Standing, terminus_depth);
        set
    }

    /// One level of [`Self::resolve`]: `dir` is listed at `slot`, the baseline holds `before` for
    /// it, its branch stood as `branch`, and its entries are matches at `depth` one.
    fn resolve_level(
        &mut self,
        tree: &mut Tree,
        slot: ResourceId,
        dir: &DirSnapshot,
        before: PriorDir<'_>,
        branch: Branch,
        depth: u32,
    ) {
        if depth == 1 {
            for (name, child) in dir.entries() {
                match child.kind() {
                    kind @ (EntryKind::File | EntryKind::Dir) => {
                        let slot = listed_slot(tree, slot, name);
                        self.slots.insert(slot);
                        self.matched.push(Match {
                            slot,
                            kind: kind.into(),
                            branch,
                        });
                    }
                    kind @ (EntryKind::Symlink | EntryKind::Other) => {
                        self.unsupported.get_or_insert_with(|| Unsupported {
                            dir: tree.path_of(slot).expect("the walk stands on live slots"),
                            name: name.clone(),
                            kind,
                        });
                    }
                }
            }
            return;
        }
        for (name, child) in dir.entries() {
            match child {
                ChildEntry::Dir(DirChild::Covered(sub)) => {
                    // `Opened` holds down the branch: below a directory the baseline could not read
                    // it saw nothing, so `before` turns `Unseen` there while the branch stays opened.
                    let before = before.child(name, sub.root_meta().fs_id());
                    let branch = match before {
                        PriorDir::Denied => Branch::Opened,
                        PriorDir::Listed(_) | PriorDir::Unseen => branch,
                    };
                    let slot = listed_slot(tree, slot, name);
                    self.resolve_level(tree, slot, sub, before, branch, depth - 1);
                }
                ChildEntry::Dir(DirChild::Denied(_)) => {
                    self.denied.extend(tree.lookup(Some(slot), name));
                }
                // The walker lists neither above a terminus.
                ChildEntry::Dir(DirChild::Uncovered(_)) | ChildEntry::Leaf(_) => {}
            }
        }
    }

    /// True iff a listing answers for `slot` and does not hold it: its directory lists no such
    /// name, or one no watch can anchor, or a directory above it is itself absent. Beneath a
    /// directory the snapshot holds denied nothing is certified, present or absent; a slot there
    /// has that directory's slot for an ancestor, since identity is `(parent, segment)`, and where
    /// the Tree holds none for it, no minted Sub lies beneath it.
    fn certifies_absent(&self, tree: &Tree, slot: ResourceId) -> bool {
        !self.slots.contains(&slot)
            && !tree
                .ancestors(slot)
                .any(|above| self.denied.contains(&above))
    }
}

/// The slot of the listed entry `name` in the directory at `dir`.
fn listed_slot(tree: &mut Tree, dir: ResourceId, name: &str) -> ResourceId {
    tree.ensure_child(dir, name, ResourceRole::User)
        .expect("the walk descends from a live Profile's anchor, through slots it resolved")
}

impl Engine {
    /// The templates on the discovery Profile `discovery`, in id order — the order a pass mints in
    /// and a ceiling names what it went on without. `SubRegistry::at` yields attach order, which
    /// slot reuse can decouple from id order.
    fn templates_at(&self, discovery: ProfileId) -> SmallVec<[SubId; 2]> {
        let mut templates: SmallVec<[SubId; 2]> = self
            .subs
            .at(discovery)
            .iter()
            .copied()
            .filter(|&sid| self.subs.get(sid).is_some_and(Sub::is_template))
            .collect();
        templates.sort_unstable();
        templates
    }

    /// Name, for each template on `discovery` and in `(template, path)` order, every directory its
    /// walk had cause to read and could not: what a pass at its ceiling goes on without.
    pub(crate) fn narrate_denied(
        &self,
        discovery: ProfileId,
        denied: &NonEmptyPathSet,
        out: &mut StepOutput,
    ) {
        for template in self.templates_at(discovery) {
            out.diagnostics
                .extend(denied.iter().map(|path| Diagnostic::DiscoveryDenied {
                    template,
                    path: Arc::clone(path),
                }));
        }
    }

    /// Reconcile the match set of the discovery Profile `discovery` against its post-graft
    /// `current` — the [`Consequence::Reconcile`](crate::transitions) body: one read of the
    /// snapshot ([`MatchSet::resolve`]), then each template's [`Self::mint_missing`],
    /// [`Self::reap_vanished`] and [`Self::close_pass`].
    ///
    /// Matches go in path order, templates in id order. Diagnostics are not resealed, so this is
    /// the `StepOutput`'s order, and identically driven engines mint identical ids. Every mint
    /// precedes every reap, so no id a reap frees is reused by a mint of the same pass; every close
    /// follows the last mint, so each miss of a pass enters by the state its template came in with
    /// — nothing a mint runs closes an enumeration.
    ///
    /// A pass with no Dir `current` (a file, or absent) or no template (a zombie burst, whose
    /// templates' detach already took what they minted) reads nothing, makes no slot and closes no
    /// enumeration, so a template's first real walk still enumerates.
    pub(crate) fn reconcile_matches(
        &mut self,
        discovery: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(profile) = self.profiles.get(discovery) else {
            return;
        };
        let Some(pattern) = profile.config().match_chain() else {
            debug_assert!(
                false,
                "reconcile_matches: the classify pre-check admits only MatchChain-shaped \
                 Profiles (profile = {discovery:?})",
            );
            return;
        };
        let Some(root) = profile.current_dir() else {
            return;
        };
        let templates = self.templates_at(discovery);
        if templates.is_empty() {
            return;
        }
        let matches = MatchSet::resolve(
            &mut self.tree,
            profile.resource(),
            root,
            // The seal that follows rebases the baseline onto `root`; until then it is the prior.
            PriorDir::root(profile.baseline_dir(), root.root_meta().fs_id()),
            pattern.terminus_depth(),
        );
        for &m in &matches.matched {
            for &template in &templates {
                self.mint_missing(template, m, now, out);
            }
        }
        for &template in &templates {
            self.reap_vanished(template, &matches, out);
        }
        for template in templates {
            self.close_pass(template, &matches, out);
        }
    }

    /// Seat `template`'s Sub at `m` unless it holds one there. In a branch that stood the miss enters
    /// as the template's miss; in one that opened it is an enumeration, whatever the template holds.
    fn mint_missing(&mut self, template: SubId, m: Match, now: Instant, out: &mut StepOutput) {
        let Some(Reaction::Mint { spec, state }) = self.subs.get(template).map(Sub::reaction)
        else {
            return;
        };
        // What `template` minted here sits on the Profile its minted identity keys at this slot.
        // The identity is frozen for the template's lifetime — an edit to it replaces the template
        // — so every mint of it carries this hash.
        if self
            .profiles
            .find(m.slot, spec.identity.config_hash())
            .and_then(|profile| self.subs.minted_at(template, profile))
            .is_some()
        {
            return;
        }
        let entry = match m.branch {
            Branch::Standing => state.miss(),
            Branch::Opened => MintEntry::Enumerated,
        };
        self.mint_sub(template, m.slot, m.kind, entry, now, out);
    }

    /// Detach each Sub `template` minted that `matches` certifies absent, in slot order, each after
    /// its line: diagnostics keep their order, so the reap reads before the detach it causes.
    fn reap_vanished(&mut self, template: SubId, matches: &MatchSet, out: &mut StepOutput) {
        // Collected before the first detach, which writes the set it is read from.
        let mut vanished: SmallVec<[(ResourceId, SubId); 4]> = self
            .subs
            .minted(template)
            .filter_map(|sub| {
                let slot = self.profiles.get(self.subs.get(sub)?.profile())?.resource();
                matches
                    .certifies_absent(&self.tree, slot)
                    .then_some((slot, sub))
            })
            .collect();
        vanished.sort_unstable();
        for (slot, sub) in vanished {
            out.diagnostics.push(Diagnostic::DiscoverySubReaped {
                source: template,
                sub,
                path: self.anchor_path(slot),
            });
            self.detach_sub_inner(sub, DetachReason::MatchVanished, out);
        }
    }

    /// Name the match the pass went on without, hold the fan-out to its threshold, and close the
    /// template's enumeration: the set the pass leaves is certified.
    fn close_pass(&mut self, template: SubId, matches: &MatchSet, out: &mut StepOutput) {
        if let Some(skipped) = &matches.unsupported
            && self
                .subs
                .narrate_once(template, MintNarration::UnsupportedAnchorKind)
        {
            out.diagnostics
                .push(Diagnostic::DiscoveryUnsupportedAnchorKind {
                    source: template,
                    path: Arc::from(skipped.dir.join(skipped.name.as_str())),
                    kind: skipped.kind,
                });
        }
        // The live total, a Sub kept beneath a denied directory included, compared before the line
        // is spent: a pass below the threshold leaves it unmade.
        let count = self.subs.minted(template).len();
        if count > FANOUT_WARNING_THRESHOLD
            && self
                .subs
                .narrate_once(template, MintNarration::FanoutThreshold)
        {
            out.diagnostics.push(Diagnostic::DiscoveryFanoutThreshold {
                source: template,
                count,
            });
        }
        self.subs.close_enumeration(template);
    }

    /// A template left the registry: what it minted follows, whatever removed it, each under
    /// [`DetachReason::DiscoverySourceDetached`]. Depth one — a minted Sub mints nothing, so no
    /// detach here hands back orphans of its own.
    pub(crate) fn reap_orphans(&mut self, orphans: BTreeSet<SubId>, out: &mut StepOutput) {
        for orphan in orphans {
            self.detach_sub_inner(orphan, DetachReason::DiscoverySourceDetached, out);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit pins for the pass's walk ([`MatchSet::resolve`]), the template ⟺ `MatchChain` attach
    //! boundary, and the pass over a Profile with no Dir `current`. The end-to-end lifecycle lives
    //! in `tests/discovery_lifecycle.rs`.

    use super::{Branch, Match, MatchSet};
    use crate::Engine;
    use crate::testkit::{MAX_SETTLE, SETTLE};
    use crate::testkit::{attach_discovery, discovery_subs_of, mint_spec, pre_place_dir};
    use specter_core::testkit::{
        covered, covered_at, denied, dir_snap, dir_snap_nested, empty_program, leaf, uncovered,
    };
    use specter_core::{
        ClassSet, EffectScope, EntryKind, FS_ROOT_SEGMENT, Input, MintEntry, PatternSpec, PriorDir,
        ProfileIdentity, ReactionSpec, ResourceId, ResourceKind, ResourceRole, ScanConfig,
        StepOutput, SubAttachAnchor, SubAttachRequest, SubParams, Tree,
    };
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    /// A Tree holding the FS root and the anchor `/srv` beneath it.
    fn anchored() -> (Tree, ResourceId) {
        let mut tree = Tree::new();
        let root = tree.ensure_root(FS_ROOT_SEGMENT, ResourceRole::User);
        let srv = tree
            .ensure_child(root, "srv", ResourceRole::User)
            .expect("live root");
        (tree, srv)
    }

    /// At depth one every entry of the anchor is a match. Those a watch can anchor resolve at their
    /// slots, kinded as listed, in path order; of the others the walk keeps the first by path, and
    /// makes a slot for none.
    #[test]
    fn depth_one_resolves_matches_in_path_order_and_keeps_the_first_unsupported() {
        let (mut tree, srv) = anchored();
        let now = dir_snap(&[
            ("c", EntryKind::Symlink, 3),
            ("a", EntryKind::Dir, 1),
            ("d", EntryKind::Other, 4),
            ("b.log", EntryKind::File, 2),
        ]);

        let set = MatchSet::resolve(&mut tree, srv, &now, PriorDir::Unseen, 1);

        let at = |name| tree.lookup(Some(srv), name).expect("resolved");
        assert_eq!(
            set.matched,
            [
                Match {
                    slot: at("a"),
                    kind: ResourceKind::Dir,
                    branch: Branch::Standing,
                },
                Match {
                    slot: at("b.log"),
                    kind: ResourceKind::File,
                    branch: Branch::Standing,
                },
            ],
        );
        let skipped = set.unsupported.expect("a symlink and a fifo match");
        assert_eq!(
            (&*skipped.dir, skipped.name.as_str(), skipped.kind),
            (Path::new("/srv"), "c", EntryKind::Symlink),
        );
        assert_eq!(
            (tree.lookup(Some(srv), "c"), tree.lookup(Some(srv), "d")),
            (None, None),
            "no slot is made for a match no watch can anchor",
        );
    }

    /// At depth three the walk descends the `Covered` chain directories, making each slot the Tree
    /// lacks, and passes by what the walker never lists above a terminus; a `Covered` directory at
    /// the terminus is a match, not a level. A denied chain directory is recorded by its slot, so
    /// nothing beneath it is certified absent, and a match beneath a directory the baseline held
    /// denied is in a branch that opened.
    #[test]
    fn depth_three_walks_the_chain_beside_the_tree_and_the_baseline() {
        let (mut tree, srv) = anchored();
        // A Sub minted beneath `locked` before it was denied holds that slot chain.
        let kept = ["locked", "q", "m"].iter().fold(srv, |dir, name| {
            tree.ensure_child(dir, name, ResourceRole::User)
                .expect("live parent")
        });
        let before = dir_snap_nested(&[("o", covered_at(40, &[("p", denied(41))]))]);
        let now = dir_snap_nested(&[
            ("early.txt", leaf(EntryKind::File, 1)),
            ("sealed", uncovered(2)),
            ("locked", denied(3)),
            (
                "o",
                covered_at(40, &[("p", covered_at(41, &[("m", uncovered(42))]))]),
            ),
            (
                "x",
                covered_at(
                    10,
                    &[(
                        "y",
                        covered_at(
                            11,
                            &[
                                ("log", uncovered(12)),
                                ("w", covered(dir_snap(&[("deep", EntryKind::File, 99)]))),
                                ("z.txt", leaf(EntryKind::File, 13)),
                            ],
                        ),
                    )],
                ),
            ),
        ]);
        assert_eq!(tree.lookup(Some(srv), "x"), None, "premise: no slot for x");

        let set = MatchSet::resolve(
            &mut tree,
            srv,
            &now,
            PriorDir::root(Some(&before), now.root_meta().fs_id()),
            3,
        );

        let at = |path: &[&str]| {
            path.iter()
                .try_fold(srv, |dir, name| tree.lookup(Some(dir), name))
        };
        let slot = |path: &[&str]| at(path).expect("resolved");
        let matched = |path: &[&str], kind, branch| Match {
            slot: slot(path),
            kind,
            branch,
        };
        assert_eq!(
            set.matched,
            [
                matched(&["o", "p", "m"], ResourceKind::Dir, Branch::Opened),
                matched(&["x", "y", "log"], ResourceKind::Dir, Branch::Standing),
                matched(&["x", "y", "w"], ResourceKind::Dir, Branch::Standing),
                matched(&["x", "y", "z.txt"], ResourceKind::File, Branch::Standing),
            ],
        );
        assert_eq!(
            [
                at(&["early.txt"]),
                at(&["sealed"]),
                at(&["x", "y", "w", "deep"])
            ],
            [None; 3],
            "what is passed by gets no slot",
        );
        assert!(
            !set.certifies_absent(&tree, kept),
            "nothing is certified beneath a denied directory",
        );
        assert!(!set.certifies_absent(&tree, slot(&["x", "y", "log"])));
        let y = slot(&["x", "y"]);
        let gone = tree
            .ensure_child(y, "gone", ResourceRole::User)
            .expect("live y");
        assert!(
            set.certifies_absent(&tree, gone),
            "a listing answers for a name it lacks",
        );
    }

    /// The ⟺ attach boundary, template direction: a template on a non-chain Profile is
    /// unconstructable — its Profile would classify a firing consequence it can never use.
    #[test]
    #[should_panic(expected = "ReactionSpec::Mint ⟺ ScanConfig::MatchChain")]
    fn template_on_non_chain_profile_is_unconstructable() {
        let mut e = Engine::new();
        let srv = pre_place_dir(&mut e, &["srv"]);
        let _ = e.step(
            Input::AttachSub(SubAttachRequest::from_parts(
                SubAttachAnchor::Resource(srv),
                ProfileIdentity::new(
                    ScanConfig::builder().build(),
                    MAX_SETTLE,
                    ClassSet::STRUCTURE,
                ),
                SubParams {
                    name: "disc".into(),
                    settle: SETTLE,
                    reaction: ReactionSpec::Mint(mint_spec()),
                },
            )),
            Instant::now(),
        );
    }

    /// The ⟺ attach boundary, shape direction: a plain Sub on a chain Profile is unconstructable — it
    /// could never react (a chain Profile mints attachments, never Effects). This same assert fires
    /// transitively on a chain-shaped *template*: its mint is a template-less Sub on a chain Profile.
    #[test]
    #[should_panic(expected = "ReactionSpec::Mint ⟺ ScanConfig::MatchChain")]
    fn plain_sub_on_chain_profile_is_unconstructable() {
        let mut e = Engine::new();
        let srv = pre_place_dir(&mut e, &["srv"]);
        let _ = e.step(
            Input::AttachSub(SubAttachRequest::from_parts(
                SubAttachAnchor::Resource(srv),
                ProfileIdentity::new(
                    ScanConfig::MatchChain(Arc::new(
                        PatternSpec::parse("/srv/*").expect("valid pattern"),
                    )),
                    MAX_SETTLE,
                    ClassSet::STRUCTURE,
                ),
                SubParams::spawn(
                    "plain".into(),
                    empty_program(),
                    EffectScope::SubtreeRoot,
                    SETTLE,
                    false,
                ),
            )),
            Instant::now(),
        );
    }

    /// The reconcile's non-Dir-anchor totality arm: with no Dir `current` (the cold probe hasn't
    /// answered yet — the same shape as an anchor replaced by a file), reconcile walks no termini
    /// and mints nothing; the recovery machinery owns whatever replaced the anchor. An
    /// early-returned pass is **not** an enumeration — the template's enumeration stays open, so
    /// its first real walk still mints enumerations, not a storm of appearances.
    #[test]
    fn reconcile_without_dir_current_mints_nothing() {
        let mut e = Engine::new();
        let srv = pre_place_dir(&mut e, &["srv"]);
        let now = Instant::now();
        let (sid, pid) = attach_discovery(
            &mut e,
            "disc",
            SubAttachAnchor::Resource(srv),
            "/srv/*",
            mint_spec(),
            now,
        );

        let mut out = StepOutput::default();
        e.reconcile_matches(pid, now, &mut out);
        assert!(
            out.diagnostics.is_empty(),
            "no termini ⇒ no mints, no narration; got {:?}",
            out.diagnostics,
        );
        assert!(discovery_subs_of(&e, sid).is_empty(), "nothing minted");
        assert_eq!(
            e.subs().get(sid).unwrap().mint_state().unwrap().miss(),
            MintEntry::Enumerated,
            "an early-returned pass walked nothing — the enumeration stays open",
        );
        let _ = e.cancel_all_in_flight_probes();
    }
}
