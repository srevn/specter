//! Discovery reconcile — the `MatchChain` Profile's stable-verdict consequence.
//!
//! A discovery Profile's scan shape is [`specter_core::ScanConfig::MatchChain`] and its Subs are
//! discovery templates (each carrying a [`specter_core::MintSpec`] — the attach boundary asserts
//! the coupling in both directions). A stable verdict on such a Profile *reconciles the match set*
//! in both directions: mint a dynamic Sub per (chain terminus × template) the registry doesn't
//! know, reap every minted Sub whose terminus left the certified set. The burst then exits through
//! the ordinary silent seal (`seal_baseline_silently`) — discovery fires attachments and
//! detachments, never Effects, so nothing here touches burst state or crosses the Draining gate.
//!
//! This module is the **single lifecycle authority for minted Subs**: the pass mints and reaps them
//! as the tree moves, and [`Engine::reap_orphans`] detaches what a template minted when the
//! template leaves, whatever removed it.
//!
//! The pass is idempotent: one walk of `current` yields the certified terminus set `T`, the registry
//! keeps the minted set `M` ([`specter_core::SubRegistry::minted`]), and the pass mints `T ∖ M` and
//! reaps `M ∖ T` ([`specter_core::DetachReason::MatchVanished`]) — so cold-Seed first enumeration,
//! Standard re-reconcile, post-recovery reconcile, and forced-ceiling reconcile are the same set
//! reconciliation (a diff-based fast path would see nothing on the Seed pass, where `baseline ==
//! current`). A mint enters as its template's miss ([`specter_core::MintState::miss`]: an enumeration
//! until the template's first completed pass, an appearance ever after) in a [`Branch`] that stood,
//! and as an enumeration beneath a chain directory the baseline recorded denied — a branch that
//! opened, whose matches are first seen now and may predate the watch. An enumeration mints cold
//! (restart parity — re-enumeration must not fire); an appearance's triggered Seed owes the first
//! fire. The classifier keys on the template's state and the baseline, never on the reconciling
//! burst's intent, so an overflow or recovery reseed still fires for termini that appeared inside the
//! blind window — the minted set `M` is the durable prior-match-set witness (it survives anchor loss,
//! overflow, and baseline clears; `baseline` does not, so a branch that opened across an anchor loss
//! enters by the template's miss, as appearances). Membership is anchor-*slot* identity: `(parent,
//! segment)` survives delete-and-recreate, so an atomically replaced terminus stays in `M ∩ T` — the
//! minted Sub keeps its `SubId`, fire history, and B1 dedup identity across the replace, and its own
//! anchor-loss descent (not discovery) drives the recovery fire. Removal is certified absence: it
//! consumes only certified post-graft snapshots — reconcile is reached from `Stable` verdicts alone,
//! and an unenumerable root returns early rather than reading as "all matches vanished" — and it
//! keeps a minted Sub beneath a chain directory the snapshot holds denied, where the walk certified
//! nothing present or absent. So a degraded read can never reap a live match.
//!
//! Determinism: termini surface in `BTreeMap` (lexicographic) order, templates in sorted-`SubId`
//! order, victims in sorted `(template, anchor slot)` order, so mint and reap order — and therefore
//! minted `SubId`s and the `StepOutput` — is deterministic across identically-driven engines.

use crate::Engine;
use crate::path::empty_path;
use compact_str::CompactString;
use smallvec::SmallVec;
use specter_core::{
    ChildEntry, DetachReason, Diagnostic, DirChild, DirSnapshot, EntryKind, MintEntry,
    MintNarration, NonEmptyPathSet, PriorDir, ProfileId, ResourceId, ResourceRole, StepOutput, Sub,
    SubId,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// A template whose minted set holds more Subs than this makes
/// [`Diagnostic::DiscoveryFanoutThreshold`], once in its lifetime: the pattern is matching more
/// targets than typical — likely a too-broad pattern.
pub(crate) const FANOUT_WARNING_THRESHOLD: usize = 1000;

/// How the branch a terminus sits in stood in the baseline the pass reconciles against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Branch {
    /// The baseline read the branch, or did not hold it at all: a terminus it lacks appeared under
    /// watch.
    Standing,
    /// A chain directory above the terminus was this same directory, recorded denied in the
    /// baseline: the terminus was first seen when its branch opened.
    Opened,
}

/// One matched chain terminus: the anchor-relative path as root-first snapshot entry names, the
/// snapshot's kind for the matched entry, and how its branch stood in the baseline.
///
/// Segments stay the snapshot's own `CompactString` keys end to end — the slot walk keys
/// `Tree::ensure_child` per segment and the absolute path is built by joining them onto the anchor
/// path, so no intermediate `PathBuf` is parsed back into components.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ChainTerminus {
    pub(crate) segments: SmallVec<[CompactString; 4]>,
    pub(crate) kind: EntryKind,
    pub(crate) branch: Branch,
}

/// Collect every chain terminus reachable from `now` — the entries at anchor-relative depth
/// `terminus_depth` under `Covered` directories only — each with its [`Branch`], read off `before`:
/// what the baseline holds for `now`'s root.
///
/// Pure free function over the pruned snapshot a `MatchChain` walk produces: chain directories
/// strictly above the terminus are `Covered` (the shape's `descends_into` recursed) or `Denied`,
/// terminus directories are `Uncovered`, terminus files ordinary leaves. A `Denied` chain directory
/// holds no listing, so nothing beneath it is collected — and nothing beneath it is certified
/// absent either ([`denied_above`]). A `Leaf` or `Uncovered` Dir above the terminus is skipped as
/// totality, not policy: the walker never emits one (mid-chain non-dirs are filter-dropped at the
/// kinded gate), so the skip only absorbs adversarial hand-built snapshots.
///
/// Per-level `BTreeMap` iteration ⇒ lexicographic terminus order ⇒ deterministic mint order.
pub(crate) fn collect_chain_termini(
    now: &DirSnapshot,
    before: PriorDir<'_>,
    terminus_depth: u32,
) -> Vec<ChainTerminus> {
    let mut out = Vec::new();
    let mut prefix: SmallVec<[CompactString; 4]> = SmallVec::new();
    collect_into(
        now,
        before,
        Branch::Standing,
        terminus_depth,
        1,
        &mut prefix,
        &mut out,
    );
    out
}

/// DFS body of [`collect_chain_termini`]. `entry_depth` is the depth of `dir`'s *entries* (anchor
/// children = 1); recursion depth is bounded by the chain length, so the stack stays shallow.
/// `before` is what the baseline holds for `dir`, and `branch` how `dir`'s branch stood.
fn collect_into(
    dir: &DirSnapshot,
    before: PriorDir<'_>,
    branch: Branch,
    terminus_depth: u32,
    entry_depth: u32,
    prefix: &mut SmallVec<[CompactString; 4]>,
    out: &mut Vec<ChainTerminus>,
) {
    for (name, child) in dir.entries() {
        if entry_depth == terminus_depth {
            let mut segments = prefix.clone();
            segments.push(name.clone());
            out.push(ChainTerminus {
                segments,
                kind: child.kind(),
                branch,
            });
        } else if let ChildEntry::Dir(DirChild::Covered(sub)) = child {
            // `Opened` is sticky down the branch: below a directory the baseline could not read it
            // saw nothing, so `before` turns `Unseen` there while the branch stays opened.
            let before = before.child(name, sub.root_meta().fs_id());
            let branch = match before {
                PriorDir::Denied => Branch::Opened,
                PriorDir::Listed(_) | PriorDir::Unseen => branch,
            };
            prefix.push(name.clone());
            collect_into(
                sub,
                before,
                branch,
                terminus_depth,
                entry_depth + 1,
                prefix,
                out,
            );
            prefix.pop();
        }
    }
}

/// True iff `root` holds a chain directory on the way to `rel` as denied: the walk certified
/// nothing beneath it, present or absent. A path a listing answers for — a chain directory that
/// lists no such name, or a directory above it that is itself absent — is not.
fn denied_above(root: &DirSnapshot, rel: &Path) -> bool {
    let mut frame = root;
    for component in rel.components() {
        match component
            .as_os_str()
            .to_str()
            .and_then(|name| frame.entries().get(name))
        {
            Some(ChildEntry::Dir(DirChild::Covered(sub))) => frame = sub,
            Some(ChildEntry::Dir(DirChild::Denied(_))) => return true,
            _ => return false,
        }
    }
    false
}

/// What the pass reads of one template, captured before its `&mut self` work begins.
struct TemplateCapture {
    sid: SubId,
    /// The hash of the identity the template mints under — the key its mints' Profiles partition
    /// on, so the dedup's.
    minted_hash: u64,
    /// The template's miss, captured **pre-pass**: every dedup miss in one pass enters by the state
    /// the template came in with — within a template's first pass, later termini stay enumerations
    /// even though earlier ones in the same pass already minted for it.
    miss: MintEntry,
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

    /// Reconcile the discovery Profile's match set against the post-graft `current` — the
    /// [`Consequence::Reconcile`](crate::transitions) body. Mints a dynamic Sub per (chain terminus
    /// × template) missing from the template's minted set, and reaps every minted Sub whose
    /// terminus the certified walk no longer enumerates ([`DetachReason::MatchVanished`]).
    ///
    /// Registry/tree/attach work only — no burst-state writer: the caller (`fire_or_seal`) runs the
    /// silent seal *after* this returns, so the burst exits through the existing category-(a)
    /// terminus. The template set is derived from the live registry at entry, which makes the
    /// zombie case self-correcting: a template detached mid-burst (its cascade already reaped the
    /// minted set) is simply absent here, so the in-flight burst's reconcile mints nothing, reaps
    /// nothing, and the seal reaps the Profile.
    ///
    /// Each mint enters through [`Engine::mint_sub`] with its [`MintEntry`] (read off the pre-pass
    /// capture of the template's miss and the terminus's [`Branch`]): an enumeration enters its own
    /// cold Seed (probe emitted within the same `StepOutput`); an appearance's Seed opens triggered
    /// (Batching-first, debouncing a terminus still being written) and owes the first fire.
    /// Discovery's "fire" is a batch of attachments. Each reap runs the ordinary `detach_sub_inner`
    /// pipeline, so a victim's Profile follows the standard detach lifecycle (`ReapNow` from Idle /
    /// Pending — common, since a vanished terminus usually parked its minted Profile in a recovery
    /// descent — deferred to burst end from Active).
    pub(crate) fn reconcile_matches(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Owned pre-borrow captures: everything the mint loop needs off the Profile and the
        // registry, taken before `&mut self` work begins.
        let Some(profile) = self.profiles.get(profile_id) else {
            return;
        };
        let Some(spec) = profile.config().match_chain().map(Arc::clone) else {
            debug_assert!(
                false,
                "reconcile_matches: the classify pre-check admits only MatchChain-shaped \
                 Profiles (profile = {profile_id:?})",
            );
            return;
        };
        let anchor = profile.resource();
        let Some(root) = profile.current_dir().map(Arc::clone) else {
            // File-kind or absent anchor ⇒ no termini to walk. Nothing to mint; the recovery
            // machinery owns whatever replaced the anchor.
            return;
        };
        // The baseline the pass reconciles against, read before the seal rebases it onto `root`: a
        // chain directory it recorded denied marks what is found beneath it as a branch that opened.
        let before = profile.baseline_dir().map(Arc::clone);

        let templates: Vec<TemplateCapture> = self
            .templates_at(profile_id)
            .into_iter()
            .filter_map(|sid| {
                let s = self.subs.get(sid)?;
                Some(TemplateCapture {
                    sid,
                    minted_hash: s.mint_spec()?.identity.config_hash(),
                    miss: s.mint_state()?.miss(),
                })
            })
            .collect();
        if templates.is_empty() {
            return;
        }

        let Some(anchor_path) = self.tree.path_of(anchor) else {
            debug_assert!(
                false,
                "reconcile_matches: a live mid-burst Profile's anchor claim holds its slot \
                 (profile = {profile_id:?}, resource = {anchor:?})",
            );
            // Reconcile is idempotent; the next burst retries with a resolvable anchor.
            return;
        };

        // The certified terminus-slot set `T`, collected alongside the mint walk. `Symlink` /
        // `Other` termini are deliberately absent — see the skip arm — so a terminus degraded to an
        // unmintable kind reaps its minted Sub in the removal pass below rather than
        // thrash-recovering against a kind that can never anchor a watch.
        let mut live_slots: BTreeSet<ResourceId> = BTreeSet::new();

        let termini = collect_chain_termini(
            &root,
            PriorDir::root(before.as_ref(), root.root_meta().fs_id()),
            spec.terminus_depth(),
        );
        for terminus in termini {
            // The absolute path materialises only for a template's first unsupported-kind line: a
            // steady-state pass allocates no paths — O(termini) allocations would otherwise recur
            // on every no-op reconcile.
            let build_abs = |segments: &[CompactString]| -> Arc<Path> {
                let mut p = anchor_path.to_path_buf();
                for seg in segments {
                    p.push(seg.as_str());
                }
                Arc::from(p)
            };

            // A `Symlink` / `Other` (fifo / socket / device) terminus skips the mint wholesale — the
            // slot dance never runs (the post-graft reconciler's own diff bookkeeping is independent
            // of this arm), no kind is stamped, no Sub attaches, and the slot stays out of
            // `live_slots`. Minting one would be a lie the state machine immediately unwinds: the
            // kind projects to a File slot, the minted Profile's first anchor probe `lstat`s and
            // folds `Vanished` on `!is_file()`, parking it in a recovery descent the next reconcile's
            // removal pass then reaps — a mint→park→reap round-trip per chain event, forever, exactly
            // on the patterns symlink farms match (`/srv/*/current`). Narrated once per template
            // lifetime; only the line is once — kind is read fresh off the snapshot each pass, so a
            // real file replacing the symlink at the same path mints normally below.
            if !matches!(terminus.kind, EntryKind::File | EntryKind::Dir) {
                let mut abs: Option<Arc<Path>> = None;
                for t in &templates {
                    if !self
                        .subs
                        .narrate_once(t.sid, MintNarration::UnsupportedAnchorKind)
                    {
                        continue;
                    }
                    let abs = abs.get_or_insert_with(|| build_abs(&terminus.segments));
                    out.diagnostics
                        .push(Diagnostic::DiscoveryUnsupportedAnchorKind {
                            source: t.sid,
                            path: Arc::clone(abs),
                            kind: terminus.kind,
                        });
                }
                continue;
            }

            // Slot dance, get-or-create per segment. The graft has already ensured every slot this
            // snapshot lists (`apply_diff_to_tree` ensures each created entry, termini included),
            // so `ensure_child` re-creates a slot only where a minted Profile's reap took it since:
            // a terminus holds no watch of this Profile's — a Dir terminus is the shape's boundary
            // (`wants_descendant_watch` is interior-gated), a File terminus a leaf under a mask
            // with no per-file watches — so watch demand there is the minted Profile's alone, and
            // the slot lives only while a minted Profile anchors it. Stamping the observed kind —
            // only `File | Dir` reaches here, so the `EntryKind → ResourceKind` projection is
            // faithful — lets `Profile.kind` cache at attach instead of waiting for the minted
            // Profile's first Seed probe.
            let mut slot = anchor;
            for seg in &terminus.segments {
                slot = self
                    .tree
                    .ensure_child(slot, seg, ResourceRole::User)
                    .expect("chain slots held alive by the discovery Profile's anchor claim");
            }
            self.tree.set_kind(slot, terminus.kind.into());
            live_slots.insert(slot);

            for t in &templates {
                // What `t` minted here sits on the Profile its minted identity keys at this slot —
                // the mint's find-or-create partition. A template's identity is frozen for its
                // lifetime (an edit to it replaces the template), so every mint of `t` carries that
                // hash.
                if self
                    .profiles
                    .find(slot, t.minted_hash)
                    .and_then(|profile| self.subs.minted_at(t.sid, profile))
                    .is_some()
                {
                    continue;
                }
                // In a branch that stood, a miss enters as the template's miss says; in a branch
                // the daemon could not read until now, it is an enumeration whatever the template
                // holds — its matches may predate the watch.
                let entry = match terminus.branch {
                    Branch::Standing => t.miss,
                    Branch::Opened => MintEntry::Enumerated,
                };
                self.mint_sub(t.sid, slot, entry, now, out);
            }
        }

        // Removal pass — `M ∖ T`, where a listing answers for the match: a minted Sub whose anchor
        // slot the certified walk did not enumerate has left the match set, unless a chain
        // directory above its terminus is one the snapshot holds denied — beneath it the walk
        // certified nothing, present or absent, so the Sub stays as it is until that directory is
        // read. A *replaced* terminus never lands here: its slot survives the replace (`(parent,
        // segment)` identity), so it sits in `M ∩ T` and its own recovery descent drives the fire.
        // Each template's victims are collected before the first detach mutates its set; a victim's
        // anchor slot is alive even from a Pending Profile (back-ref retention), so `path_of`
        // resolves for the narration, and the slot cascades once `detach_sub_inner` reaps the
        // Profile. Narration precedes the detach so the template-keyed `DiscoverySubReaped` reads
        // causally before its per-Sub `SubDetached(MatchVanished)`.
        for t in &templates {
            let mut vanished: SmallVec<[(ResourceId, SubId); 4]> = self
                .subs
                .minted(t.sid)
                .filter_map(|sub| {
                    let slot = self.profiles.get(self.subs.get(sub)?.profile())?.resource();
                    (!live_slots.contains(&slot)).then_some((slot, sub))
                })
                .collect();
            vanished.sort_unstable();
            for (slot, victim) in vanished {
                let path = self.tree.path_of(slot).unwrap_or_else(|| {
                    debug_assert!(
                        false,
                        "reconcile removal: a live minted Sub's anchor slot is back-ref-held \
                         (sub = {victim:?}, resource = {slot:?})",
                    );
                    empty_path()
                });
                if path
                    .strip_prefix(&anchor_path)
                    .is_ok_and(|rel| denied_above(&root, rel))
                {
                    continue;
                }
                out.diagnostics.push(Diagnostic::DiscoverySubReaped {
                    source: t.sid,
                    sub: victim,
                    path,
                });
                self.detach_sub_inner(victim, DetachReason::MatchVanished, out);
            }
        }

        // Each template's pass closes on its minted set as the pass leaves it: its length is the
        // fan-out the threshold is held to — the live total, a Sub kept beneath a denied directory
        // included — compared before the line is spent, so a pass below the threshold leaves it
        // unmade. A completed pass *is* the enumeration, so it closes here, idempotently (a
        // template detached by an interleaved cascade is a silent miss at the registry edge), and
        // nowhere earlier: the early returns above (no Dir `current`, empty template set,
        // unresolvable anchor path) walked nothing, so a template leaving such a pass still
        // classifies its first *real* walk as an enumeration, not as a storm of appearances.
        for t in &templates {
            let count = self.subs.minted(t.sid).len();
            if count > FANOUT_WARNING_THRESHOLD
                && self
                    .subs
                    .narrate_once(t.sid, MintNarration::FanoutThreshold)
            {
                out.diagnostics.push(Diagnostic::DiscoveryFanoutThreshold {
                    source: t.sid,
                    count,
                });
            }
            self.subs.close_enumeration(t.sid);
        }
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
    //! Unit pins for the discovery reconcile building blocks: the pure terminus collector, the
    //! template ⟺ `MatchChain` attach boundary, and the reconcile's non-Dir-anchor totality arm.
    //! The end-to-end reconcile lifecycle lives in `tests/discovery_lifecycle.rs`.

    use super::{Branch, ChainTerminus, collect_chain_termini};
    use crate::Engine;
    use crate::testkit::{MAX_SETTLE, SETTLE};
    use crate::testkit::{attach_discovery, discovery_subs_of, mint_spec, pre_place_dir};
    use compact_str::CompactString;
    use specter_core::testkit::{
        covered, dir_snap, dir_snap_nested, empty_program, leaf, uncovered,
    };
    use specter_core::{
        ClassSet, EffectScope, EntryKind, Input, MintEntry, PatternSpec, PriorDir, ProfileIdentity,
        ReactionSpec, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubParams,
    };
    use std::sync::Arc;
    use std::time::Instant;

    /// A terminus in a branch that stood: the collector pins below walk with no baseline.
    fn terminus(segments: &[&str], kind: EntryKind) -> ChainTerminus {
        ChainTerminus {
            segments: segments.iter().map(|s| CompactString::new(s)).collect(),
            kind,
            branch: Branch::Standing,
        }
    }

    /// td = 1: every root entry is a terminus, whatever its kind — Dir, File, and Symlink all mint
    /// (`EntryKind → ResourceKind` folds non-dirs to `File` downstream, but the collector reports
    /// the snapshot's own kind). Order is the `BTreeMap`'s lexicographic walk.
    #[test]
    fn termini_at_depth_one_collect_every_entry_kind_in_lexicographic_order() {
        let root = dir_snap(&[
            ("c", EntryKind::Symlink, 3),
            ("a", EntryKind::Dir, 1),
            ("b.log", EntryKind::File, 2),
        ]);
        assert_eq!(
            collect_chain_termini(&root, PriorDir::Unseen, 1),
            vec![
                terminus(&["a"], EntryKind::Dir),
                terminus(&["b.log"], EntryKind::File),
                terminus(&["c"], EntryKind::Symlink),
            ],
        );
    }

    /// td = 3: the collector descends `Covered` chain dirs only, and a terminus-level `Covered` dir
    /// (a shape the walker never emits — `descends_into` refuses at td) still collects as a Dir
    /// terminus rather than being descended past the chain bound.
    #[test]
    fn termini_at_depth_three_walk_covered_chains_to_the_bound() {
        let root = dir_snap_nested(&[(
            "x",
            covered(dir_snap_nested(&[(
                "y",
                covered(dir_snap_nested(&[
                    ("log", uncovered(10)),
                    ("w", covered(dir_snap(&[("deep", EntryKind::File, 99)]))),
                    ("z.txt", leaf(EntryKind::File, 11)),
                ])),
            )])),
        )]);
        assert_eq!(
            collect_chain_termini(&root, PriorDir::Unseen, 3),
            vec![
                terminus(&["x", "y", "log"], EntryKind::Dir),
                terminus(&["x", "y", "w"], EntryKind::Dir),
                terminus(&["x", "y", "z.txt"], EntryKind::File),
            ],
        );
    }

    /// Adversarial snapshot: a `Leaf` and an `Uncovered` Dir strictly above the terminus depth are
    /// skipped (totality, not policy — the pruned walk never emits them); only the `Covered`
    /// chain's entries at the bound collect. An empty root collects nothing.
    #[test]
    fn entries_above_the_terminus_that_cannot_recurse_are_skipped() {
        let root = dir_snap_nested(&[
            ("early.txt", leaf(EntryKind::File, 1)),
            ("sealed", uncovered(2)),
            ("chain", covered(dir_snap_nested(&[("log", uncovered(3))]))),
        ]);
        assert_eq!(
            collect_chain_termini(&root, PriorDir::Unseen, 2),
            vec![terminus(&["chain", "log"], EntryKind::Dir)],
        );
        assert_eq!(
            collect_chain_termini(&dir_snap(&[]), PriorDir::Unseen, 1),
            [] as [ChainTerminus; 0]
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
