//! Discovery reconcile — the `MatchChain` Profile's stable-verdict consequence.
//!
//! A discovery Profile's scan shape is [`specter_core::ScanConfig::MatchChain`] and its Subs are
//! discovery templates ([`specter_core::MintTemplate`]-bearing — the attach boundary asserts the
//! coupling in both directions). A stable verdict on such a Profile *reconciles the match set* in
//! both directions: mint a dynamic Sub per (chain terminus × template) the registry doesn't know,
//! reap every minted Sub whose terminus left the certified set. The burst then exits through the
//! ordinary silent seal (`seal_baseline_silently`) — discovery fires attachments and detachments,
//! never Effects, so nothing here touches burst state or crosses the Draining gate.
//!
//! Reconcile is the **single lifecycle authority for minted Subs** and is idempotent: one walk of
//! `current` yields the certified terminus set `T`, one registry projection yields the minted set
//! `M`, and the pass mints `T ∖ M` and reaps `M ∖ T` ([`specter_core::DetachReason::MatchVanished`])
//! — so cold-Seed first enumeration, Standard re-reconcile, post-recovery reconcile, and
//! forced-ceiling reconcile are the same set reconciliation (a diff-based fast path would see nothing
//! on the Seed pass, where `baseline == current`). A mint classifies by its template's `enumerated`
//! latch (closed at the end of the template's first completed pass, never re-opened): un-enumerated ⇒
//! cold (restart parity — re-enumeration must not fire), enumerated ⇒ a witnessed appearance whose
//! triggered Seed owes the first fire. The classifier keys on the latch, never on the reconciling
//! burst's intent, so an overflow or recovery reseed still fires for termini that appeared inside the
//! blind window — the minted set `M` is the durable prior-match-set witness (it survives anchor loss,
//! overflow, and baseline clears; `baseline` does not). Membership is anchor-*slot* identity:
//! `(parent, segment)` survives delete-and-recreate, so an atomically replaced terminus stays in `M ∩
//! T` — the minted Sub keeps its `SubId`, fire history, and B1 dedup identity across the replace, and
//! its own anchor-loss descent (not discovery) drives the recovery fire. Removal consumes only
//! certified post-graft snapshots — reconcile is reached from `Stable` verdicts alone, and an
//! unenumerable root returns early rather than reading as "all matches vanished" — so a degraded read
//! can never reap a live match.
//!
//! Determinism: termini surface in `BTreeMap` (lexicographic) order, templates in sorted-`SubId`
//! order, victims in sorted `(source, anchor slot)` order, so mint and reap order — and therefore
//! minted `SubId`s and the `StepOutput` — is deterministic across identically-driven engines.

use crate::Engine;
use crate::engine::SeedWitness;
use crate::path::empty_path;
use compact_str::{CompactString, format_compact};
use smallvec::SmallVec;
use specter_core::{
    ChildEntry, DetachReason, Diagnostic, DirChild, DirSnapshot, EntryKind, MintTemplate, Profile,
    ProfileId, ReactionSpec, ResourceId, ResourceRole, StepOutput, SubAttachAnchor,
    SubAttachRequest, SubId, SubParams,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Threshold beyond which the engine emits a one-shot [`Diagnostic::DiscoveryFanoutThreshold`] for
/// a discovery template. Operator signal that the pattern is matching more targets than typical —
/// likely a too-broad pattern. The registry-side check-and-latch
/// (`SubRegistry::latch_fanout_warning`) is atomic, so a steady-state busy source warns once per
/// lifetime by construction.
pub(crate) const FANOUT_WARNING_THRESHOLD: usize = 1000;

/// One matched chain terminus: the anchor-relative path as root-first snapshot entry names, plus
/// the snapshot's kind for the matched entry.
///
/// Segments stay the snapshot's own `CompactString` keys end to end — the slot walk keys
/// `Tree::ensure_child` per segment and the absolute path is built by joining them onto the anchor
/// path, so no intermediate `PathBuf` is parsed back into components.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ChainTerminus {
    pub(crate) segments: SmallVec<[CompactString; 4]>,
    pub(crate) kind: EntryKind,
}

/// Collect every chain terminus reachable from `root` — the entries at anchor-relative depth
/// `terminus_depth` under `Covered` directories only.
///
/// Pure free function over the pruned snapshot a `MatchChain` walk produces: chain directories
/// strictly above the terminus are `Covered` (the shape's `descends_into` recursed), terminus
/// directories are `Uncovered`, terminus files ordinary leaves. The `Covered`-only descent is
/// totality, not policy — a `Leaf` or `Uncovered` Dir above the terminus is skipped because the
/// walker never emits one (mid-chain non-dirs are filter-dropped at the kinded gate), so the skip
/// only absorbs adversarial hand-built snapshots.
///
/// Per-level `BTreeMap` iteration ⇒ lexicographic terminus order ⇒ deterministic mint order.
pub(crate) fn collect_chain_termini(root: &DirSnapshot, terminus_depth: u32) -> Vec<ChainTerminus> {
    let mut out = Vec::new();
    let mut prefix: SmallVec<[CompactString; 4]> = SmallVec::new();
    collect_into(root, terminus_depth, 1, &mut prefix, &mut out);
    out
}

/// DFS body of [`collect_chain_termini`]. `entry_depth` is the depth of `dir`'s *entries* (anchor
/// children = 1); recursion depth is bounded by the chain length, so the stack stays shallow.
fn collect_into(
    dir: &DirSnapshot,
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
            });
        } else if let ChildEntry::Dir(DirChild::Covered(sub)) = child {
            prefix.push(name.clone());
            collect_into(sub, terminus_depth, entry_depth + 1, prefix, out);
            prefix.pop();
        }
    }
}

/// Owned capture of one template Sub's mint inputs, collected before the reconcile loop takes `&mut
/// self` — an Arc refcount bump per pass instead of re-borrowing the registry per mint. The
/// template carries everything a mint needs (identity with its sealed hash, debounce, the minted
/// reaction); only the Sub's `name` (the synthesized-name prefix) rides alongside.
struct TemplateCapture {
    sid: SubId,
    tpl: Arc<MintTemplate>,
    name: CompactString,
    /// The template's enumeration latch, captured **pre-pass**: every dedup miss in one pass
    /// classifies uniformly by the state the template entered with — within a minting pass, later
    /// termini for a *new* template stay cold even though earlier termini in the same pass already
    /// minted for it. `false` ⇒ this pass is the template's first enumeration (cold mints, restart
    /// parity); `true` ⇒ a miss is a witnessed appearance (triggered mint, owes its first fire).
    enumerated: bool,
    /// Subs minted for this template by this pass — the end-of-pass fan-out sweep's gate and its
    /// contribution to the live count. `0` at capture; the mint arm increments it.
    minted: usize,
}

impl Engine {
    /// Reconcile the discovery Profile's match set against the post-graft `current` — the
    /// [`Consequence::Reconcile`](crate::transitions) body. Mints a dynamic Sub per (chain terminus
    /// × template) the registry projection doesn't know, and reaps every projected minted Sub whose
    /// terminus the certified walk no longer enumerates ([`DetachReason::MatchVanished`]).
    ///
    /// Registry/tree/attach work only — no burst-state writer: the caller (`fire_or_seal`) runs the
    /// silent seal *after* this returns, so the burst exits through the existing category-(a)
    /// terminus. The template set is derived from the live registry at entry, which makes the
    /// zombie case self-correcting: a template detached mid-burst (its cascade already reaped the
    /// minted set) is simply absent here, so the in-flight burst's reconcile mints nothing, reaps
    /// nothing, and the seal reaps the Profile.
    ///
    /// Each mint runs the ordinary `attach_sub_inner` pipeline with a witness classified per
    /// template ([`SeedWitness`], read off the pre-pass `enumerated` capture): a first-enumeration
    /// mint enters its own cold Seed (probe emitted within the same `StepOutput`); a mint against
    /// an established enumeration is a witnessed appearance — its Seed opens triggered
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

        let mut templates: Vec<TemplateCapture> = self
            .subs
            .at(profile_id)
            .iter()
            .filter_map(|&sid| {
                let s = self.subs.get(sid)?;
                let t = s.discovery_template()?;
                Some(TemplateCapture {
                    sid,
                    tpl: Arc::clone(t.spec()),
                    name: s.name.clone(),
                    enumerated: t.enumerated(),
                    minted: 0,
                })
            })
            .collect();
        // `subs.at` yields attach order, which slot reuse can decouple from id order; the explicit
        // sort pins the per-terminus mint order to sorted SubIds for cross-engine determinism.
        templates.sort_unstable_by_key(|t| t.sid);
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

        // Per-pass registry projection: every live Sub minted by one of this Profile's templates,
        // keyed `(source template, anchor slot)`. One scan serves the pass's three set consumers —
        // the mint dedup (`T ∖ M`), the removal pass (`M ∖ T`), and the fan-out live count. The key
        // determines the Sub because a minted Sub for `(template, anchor)` lives exactly on the
        // Profile at `(anchor, template.identity.config_hash())` (`attach_sub_inner`'s
        // find-or-create partition) and template identity changes are wholesale replaces (the
        // cascade reaps the minted set), never in-place rebinds. The anchor slot resolves even for
        // a victim parked in a recovery descent: the `Resource.profiles` back-ref holds the slot
        // alive while its Profile lives, whatever the state. The membership test runs once per
        // registry Sub, so a linear `templates` scan would make the projection O(S·T); the
        // pass-local id set drops it to O(S·log T).
        let template_ids: BTreeSet<SubId> = templates.iter().map(|t| t.sid).collect();
        let minted_index: BTreeMap<(SubId, ResourceId), SubId> = self
            .subs
            .iter()
            .filter_map(|(sid, s)| {
                let src = s.minted_by()?;
                if !template_ids.contains(&src) {
                    return None;
                }
                let slot = self.profiles.get(s.profile()).map(Profile::resource)?;
                Some(((src, slot), sid))
            })
            .collect();

        // The certified terminus-slot set `T`, collected alongside the mint walk. `Symlink` /
        // `Other` termini are deliberately absent — see the skip arm — so a terminus degraded to an
        // unmintable kind reaps its minted Sub in the removal pass below rather than
        // thrash-recovering against a kind that can never anchor a watch.
        let mut live_slots: BTreeSet<ResourceId> = BTreeSet::new();

        for terminus in collect_chain_termini(&root, spec.terminus_depth()) {
            // The absolute path (name suffix + diag payload) materialises lazily — on the first
            // dedup miss, or on a newly-latched unsupported-kind warning: a steady-state pass where
            // every template already minted (or already warned) allocates no paths — O(termini)
            // allocations would otherwise recur on every no-op reconcile.
            let build_abs = |segments: &[CompactString]| -> Arc<Path> {
                let mut p = anchor_path.to_path_buf();
                for seg in segments {
                    p.push(seg.as_str());
                }
                Arc::from(p)
            };

            // A `Symlink` / `Other` (fifo / socket / device) terminus skips the mint wholesale —
            // the slot dance never runs (the post-graft reconciler's own diff bookkeeping is
            // independent of this arm), no kind is stamped, no Sub attaches, and the slot stays out
            // of `live_slots`. Minting one would be a lie the state machine immediately unwinds:
            // the kind projects to a File slot, the minted Profile's first anchor probe `lstat`s
            // and folds `Vanished` on `!is_file()`, parking it in a recovery descent the next
            // reconcile's removal pass then reaps — a mint→park→reap round-trip per chain event,
            // forever, exactly on the patterns symlink farms match (`/srv/*/current`). Narrated
            // once per template lifetime through the registry's check-and-latch; the latch gates
            // only the diagnostic — kind is read fresh off the snapshot each pass, so a real file
            // replacing the symlink at the same path mints normally below.
            if !matches!(terminus.kind, EntryKind::File | EntryKind::Dir) {
                let mut abs: Option<Arc<Path>> = None;
                for t in &templates {
                    if !self.subs.latch_unsupported_kind_warning(t.sid) {
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

            // Slot dance, get-or-create per segment: `ensure_child` is idempotent over the chain-dir
            // slots the post-graft reconciler already created (role `User`) and creates only the
            // terminus slots. A terminus is the discovery shape's boundary — its interior is no part
            // of the proof object — so the reconciler installs no descendant watch there
            // (`wants_descendant_watch` is interior-gated) and leaves anchor demand at the slot
            // entirely to the minted Profile. Stamping the observed kind — only `File | Dir` reaches
            // here, so the `EntryKind → ResourceKind` projection is faithful — lets `Profile.kind`
            // cache at attach instead of waiting for the minted Profile's first Seed probe.
            let mut slot = anchor;
            for seg in &terminus.segments {
                slot = self
                    .tree
                    .ensure_child(slot, seg, ResourceRole::User)
                    .expect("chain slots held alive by the discovery Profile's anchor claim");
            }
            self.tree.set_kind(slot, terminus.kind.into());
            live_slots.insert(slot);

            let mut abs: Option<Arc<Path>> = None;
            for t in &mut templates {
                if minted_index.contains_key(&(t.sid, slot)) {
                    continue;
                }
                let abs = abs.get_or_insert_with(|| build_abs(&terminus.segments));
                // The mint-arm classifier: a dedup miss while the template's enumeration is
                // established means the terminus *appeared* — the discovery Profile observed the
                // driving event and certified the set delta, so the mint carries that witness and
                // its Seed owes the first fire. A miss on an un-enumerated template is first
                // enumeration: cold, restart parity — re-enumeration after a daemon restart (or a
                // mid-life template join) must not fire.
                let witness = if t.enumerated {
                    SeedWitness::Appeared
                } else {
                    SeedWitness::Cold
                };
                // `format_compact!` writes straight into the `CompactString` that becomes
                // `SubParams.name`; the `@` byte is reserved at config validation, so synthesised
                // names never collide with operator names in the registry's `by_name` index. The
                // template name (the prefix) is itself `@`-free for the same reason, so the *first*
                // `@` delimits prefix from path deterministically: `(template, terminus-path) →
                // name` is injective, and no two distinct mints can synthesise one name.
                let synthesized = format_compact!("{}@{}", t.name, abs.display());
                self.attach_sub_inner(
                    SubAttachRequest::from_parts(
                        SubAttachAnchor::Resource(slot),
                        t.tpl.identity.clone(),
                        SubParams {
                            name: synthesized,
                            settle: t.tpl.settle,
                            // The template's sealed spawn clones straight onto the mint — an Arc
                            // bump plus `Copy` fields, with `needs_diff` derived once at lowering
                            // rather than once per mint.
                            reaction: ReactionSpec::Spawn {
                                spec: t.tpl.spawn.clone(),
                                minted_by: Some(t.sid),
                            },
                        },
                    ),
                    witness,
                    now,
                    out,
                )
                .expect(
                    "discovery mint anchored at a freshly ensured live User slot; \
                     the engine's Resource-arm liveness check cannot trip",
                );
                out.diagnostics.push(Diagnostic::DiscoveryMinted {
                    source: t.sid,
                    path: Arc::clone(abs),
                    // Faithful here, unlike on the skip arm's diagnostic: only `File | Dir` termini
                    // reach the mint, so the `ResourceKind` projection loses nothing and names
                    // exactly the kind the Tree stamped.
                    kind: terminus.kind.into(),
                    appeared: t.enumerated,
                });
                t.minted += 1;
            }
        }

        // Removal pass — `M ∖ T`: every projected minted Sub whose anchor slot the certified walk
        // did not enumerate has genuinely left the match set. A *replaced* terminus never lands
        // here: its slot survives the replace (`(parent, segment)` identity), so it sits in `M ∩ T`
        // and its own recovery descent drives the fire. Iteration is over the local projection, so
        // the registry mutation inside `detach_sub_inner` cannot invalidate it; the victim's anchor
        // slot is alive even from a Pending Profile (back-ref retention), so `path_of` resolves for
        // the narration, and the slot cascades once `detach_sub_inner` reaps the Profile. Narration
        // precedes the detach so the source-keyed `DiscoverySubReaped` reads causally before its
        // per-Sub `SubDetached(MatchVanished)`.
        for (&(source, slot), &victim) in &minted_index {
            if live_slots.contains(&slot) {
                continue;
            }
            let path = self.tree.path_of(slot).unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "reconcile removal: a live minted Sub's anchor slot is back-ref-held \
                     (sub = {victim:?}, resource = {slot:?})",
                );
                empty_path()
            });
            out.diagnostics.push(Diagnostic::DiscoverySubReaped {
                source,
                sub: victim,
                path,
            });
            self.detach_sub_inner(victim, DetachReason::MatchVanished, out);
        }

        // End-of-pass fan-out sweep, gated on minted-this-pass. The live count crosses the
        // threshold upward only via a mint and mints happen only in the loop above, so a pass that
        // minted nothing cannot be the crossing pass — quiet steady-state reconciles never touch
        // the latch. The count derives from the pass's own sets — projected survivors plus this
        // pass's mints — so no extra registry scan runs; `SubRegistry::latch_fanout_warning`'s
        // atomic check-and-latch keeps the once-per-template-lifetime property structural.
        for t in templates.iter().filter(|t| t.minted > 0) {
            let survivors = minted_index
                .keys()
                .filter(|(src, slot)| *src == t.sid && live_slots.contains(slot))
                .count();
            if let Some(count) = self.subs.latch_fanout_warning(
                t.sid,
                FANOUT_WARNING_THRESHOLD,
                survivors + t.minted,
            ) {
                out.diagnostics.push(Diagnostic::DiscoveryFanoutThreshold {
                    source: t.sid,
                    count,
                });
            }
        }

        // A completed pass *is* the enumeration: latch every captured template, unconditionally and
        // idempotently (a stale id — template detached by an interleaved cascade — is a silent miss
        // at the registry edge). This write must stay at the end of the completed walk: the early
        // returns above (no Dir `current`, empty template set, unresolvable anchor path) walked
        // nothing, so a template leaving such a pass un-latched is the point — its first *real*
        // walk still classifies as enumeration, not as a storm of appearances.
        for t in &templates {
            self.subs.mark_enumerated(t.sid);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit pins for the discovery reconcile building blocks: the pure terminus collector, the
    //! template ⟺ `MatchChain` attach boundary, and the reconcile's non-Dir-anchor totality arm.
    //! The end-to-end reconcile lifecycle lives in `tests/discovery_lifecycle.rs`.

    use super::{ChainTerminus, collect_chain_termini};
    use crate::Engine;
    use crate::testkit::{MAX_SETTLE, SETTLE};
    use crate::testkit::{attach_discovery, discovery_subs_of, mint_template, pre_place_dir};
    use compact_str::CompactString;
    use specter_core::testkit::{
        covered, dir_snap, dir_snap_nested, empty_program, leaf, uncovered,
    };
    use specter_core::{
        ClassSet, EffectScope, EntryKind, Input, PatternSpec, ProfileIdentity, ReactionSpec,
        ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubParams,
    };
    use std::sync::Arc;
    use std::time::Instant;

    fn terminus(segments: &[&str], kind: EntryKind) -> ChainTerminus {
        ChainTerminus {
            segments: segments.iter().map(|s| CompactString::new(s)).collect(),
            kind,
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
            collect_chain_termini(&root, 1),
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
            collect_chain_termini(&root, 3),
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
            collect_chain_termini(&root, 2),
            vec![terminus(&["chain", "log"], EntryKind::Dir)],
        );
        assert!(collect_chain_termini(&dir_snap(&[]), 1).is_empty());
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
                    reaction: ReactionSpec::Mint(mint_template()),
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
    /// early-returned pass is **not** an enumeration — the template's latch stays open, so its
    /// first real walk still classifies as first enumeration, not as a storm of appearances.
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
            mint_template(),
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
        assert!(
            !e.subs()
                .get(sid)
                .unwrap()
                .discovery_template()
                .unwrap()
                .enumerated(),
            "an early-returned pass walked nothing — the enumeration latch stays open",
        );
        let _ = e.cancel_all_in_flight_probes();
    }
}
