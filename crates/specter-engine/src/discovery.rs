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
//! `M`, and the pass mints `T ∖ M` and reaps `M ∖ T`
//! ([`specter_core::DetachReason::MatchVanished`]) — so cold-Seed first enumeration, Standard
//! re-reconcile, post-recovery reconcile, and forced-ceiling reconcile are the same set
//! reconciliation (a diff-based fast path would see nothing on the Seed pass, where `baseline ==
//! current`). Membership is anchor-*slot* identity: `(parent, segment)` survives
//! delete-and-recreate, so an atomically replaced terminus stays in `M ∩ T` — the minted Sub
//! keeps its `SubId`, fire history, and B1 dedup identity across the replace, and its own
//! anchor-loss descent (not discovery) drives the recovery fire. Removal consumes only certified
//! post-graft snapshots — reconcile is reached from `Stable` verdicts alone, and an unenumerable
//! root returns early rather than reading as "all matches vanished" — so a degraded read can never
//! reap a live match.
//!
//! Determinism: termini surface in `BTreeMap` (lexicographic) order, templates in sorted-`SubId`
//! order, victims in sorted `(source, anchor slot)` order, so mint and reap order — and therefore
//! minted `SubId`s and the `StepOutput` — is deterministic across identically-driven engines.

use crate::Engine;
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

/// Owned capture of one template Sub's mint inputs, collected before the reconcile loop takes
/// `&mut self` — an Arc refcount bump per pass instead of re-borrowing the registry per mint. The
/// template carries everything a mint needs (identity with its sealed hash, debounce, the minted
/// reaction); only the Sub's `name` (the synthesized-name prefix) rides alongside.
struct TemplateCapture {
    sid: SubId,
    tpl: Arc<MintTemplate>,
    name: CompactString,
    /// Subs minted for this template by this pass — the end-of-pass fan-out sweep's gate and its
    /// contribution to the live count. `0` at capture; the mint arm increments it.
    minted: usize,
}

impl Engine {
    /// Reconcile the discovery Profile's match set against the post-graft `current` — the
    /// [`Consequence::Reconcile`](crate::transitions) body. Mints a dynamic Sub per (chain terminus
    /// × template) the registry projection doesn't know, and reaps every projected minted Sub
    /// whose terminus the certified walk no longer enumerates
    /// ([`DetachReason::MatchVanished`]).
    ///
    /// Registry/tree/attach work only — no burst-state writer: the caller (`fire_or_seal`) runs the
    /// silent seal *after* this returns, so the burst exits through the existing category-(a)
    /// terminus. The template set is derived from the live registry at entry, which makes the
    /// zombie case self-correcting: a template detached mid-burst (its cascade already reaped the
    /// minted set) is simply absent here, so the in-flight burst's reconcile mints nothing, reaps
    /// nothing, and the seal reaps the Profile.
    ///
    /// Each mint runs the ordinary `attach_sub_inner` pipeline, so a minted Profile enters its own
    /// cold Seed burst (probe emitted) within the same `StepOutput` — discovery's "fire" is a batch
    /// of attachments. Each reap runs the ordinary `detach_sub_inner` pipeline, so a victim's
    /// Profile follows the standard detach lifecycle (`ReapNow` from Idle / Pending — common,
    /// since a vanished terminus usually parked its minted Profile in a recovery descent —
    /// deferred to burst end from Active).
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
                    tpl: Arc::clone(&t.spec),
                    name: s.name.clone(),
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
        // the mint dedup (`T ∖ M`), the removal pass (`M ∖ T`), and the fan-out live count. The
        // key determines the Sub because a minted Sub for `(template, anchor)` lives exactly on
        // the Profile at `(anchor, template.identity.config_hash())` (`attach_sub_inner`'s
        // find-or-create partition) and template identity changes are wholesale replaces (the
        // cascade reaps the minted set), never in-place rebinds. The anchor slot resolves even for
        // a victim parked in a recovery descent: the `Resource.profiles` back-ref holds the slot
        // alive while its Profile lives, whatever the state.
        let minted_index: BTreeMap<(SubId, ResourceId), SubId> = self
            .subs
            .iter()
            .filter_map(|(sid, s)| {
                let src = s.minted_by()?;
                if !templates.iter().any(|t| t.sid == src) {
                    return None;
                }
                let slot = self.profiles.get(s.profile()).map(Profile::resource)?;
                Some(((src, slot), sid))
            })
            .collect();

        // The certified terminus-slot set `T`, collected alongside the mint walk. `Symlink` /
        // `Other` termini are deliberately absent — see the skip arm — so a terminus degraded to
        // an unmintable kind reaps its minted Sub in the removal pass below rather than
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
            // independent of this arm), no kind is stamped, no Sub attaches, and the slot stays
            // out of `live_slots`. Minting one would be a lie the state machine immediately
            // unwinds: the kind projects to a File slot, the minted Profile's first anchor probe
            // `lstat`s and folds `Vanished` on `!is_file()`, parking it in a recovery descent the
            // next reconcile's removal pass then reaps — a mint→park→reap round-trip per chain
            // event, forever, exactly on the patterns symlink farms match (`/srv/*/current`).
            // Narrated once per template lifetime through the registry's check-and-latch; the
            // latch gates only the diagnostic — kind is read fresh off the snapshot each pass, so
            // a real file replacing the symlink at the same path mints normally below.
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

            // Slot dance, get-or-create per segment: `ensure_child` is idempotent over the
            // chain-dir slots the post-graft reconciler already created (role `User`) and creates
            // only the terminus slots (Uncovered dirs / leaves get no reconciler contribution).
            // Stamping the observed kind — only `File | Dir` reaches here, so the `EntryKind →
            // ResourceKind` projection is faithful — lets `Profile.kind` cache at attach instead of
            // waiting for the minted Profile's first Seed probe.
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
                // `format_compact!` writes straight into the `CompactString` that becomes
                // `SubParams.name`; the `@` byte is reserved at config validation, so synthesised
                // names never collide with operator names in the registry's `by_name` index.
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
                });
                t.minted += 1;
            }
        }

        // Removal pass — `M ∖ T`: every projected minted Sub whose anchor slot the certified walk
        // did not enumerate has genuinely left the match set. A *replaced* terminus never lands
        // here: its slot survives the replace (`(parent, segment)` identity), so it sits in
        // `M ∩ T` and its own recovery descent drives the fire. Iteration is over the local
        // projection, so the registry mutation inside `detach_sub_inner` cannot invalidate it; the
        // victim's anchor slot is alive even from a Pending Profile (back-ref retention), so
        // `path_of` resolves for the narration, and the slot cascades once `detach_sub_inner`
        // reaps the Profile. Narration precedes the detach so the source-keyed
        // `DiscoverySubReaped` reads causally before its per-Sub
        // `SubDetached(MatchVanished)`.
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
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod discovery_tests;
