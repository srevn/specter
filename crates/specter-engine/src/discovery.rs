//! Discovery reconcile — the `MatchChain` Profile's stable-verdict consequence.
//!
//! A discovery Profile's scan shape is [`specter_core::ScanConfig::MatchChain`] and its Subs are
//! discovery templates ([`specter_core::MintTemplate`]-bearing — the attach boundary asserts the
//! coupling in both directions). A stable verdict on such a Profile *reconciles the match set*: for
//! every chain terminus in the post-graft snapshot × every template on the Profile, mint a dynamic
//! Sub unless one already exists. The burst then exits through the ordinary silent seal
//! (`seal_baseline_silently`) — discovery fires attachments, never Effects, so nothing here touches
//! burst state or crosses the Draining gate.
//!
//! Reconcile is **add-only and idempotent**: a full walk of `current` gated by the registry-derived
//! dedup query, so cold-Seed first enumeration, Standard re-reconcile, post-recovery re-mint, and
//! forced-ceiling reconcile are the same operation (a diff-based fast path would see nothing on the
//! Seed pass, where `baseline == current`). Removal stays anchor-terminal: a vanished match's
//! minted Sub reaps via *its own* Profile's anchor-loss path, decoupled from discovery — Resource
//! identity (`(parent, segment)`) makes the two compose without double-mint or gap across a
//! vanish/reappear race inside one settle window.
//!
//! Determinism: termini surface in `BTreeMap` (lexicographic) order, templates in sorted-`SubId`
//! order, so mint order — and therefore minted `SubId`s and the `StepOutput` — is deterministic
//! across identically-driven engines.

use crate::Engine;
use compact_str::{CompactString, format_compact};
use smallvec::SmallVec;
use specter_core::{
    ActionProgram, ChildEntry, Diagnostic, DirChild, DirSnapshot, EffectScope, EntryKind,
    MintTemplate, ProfileId, ResourceId, ResourceRole, StepOutput, SubAttachAnchor,
    SubAttachRequest, SubId, SubParams,
};
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

/// Owned capture of one template-bearing Sub's mint inputs, collected before the reconcile loop
/// takes `&mut self` — Arc refcount bumps per pass instead of re-borrowing the registry per mint.
struct TemplateCapture {
    sid: SubId,
    spec: Arc<MintTemplate>,
    /// `spec.identity.config_hash()` precomputed once per pass — the Profile-partition half of the
    /// dedup key, shared by every terminus this pass visits.
    cfg_hash: u64,
    name: CompactString,
    program: Arc<ActionProgram>,
    scope: EffectScope,
    log_output: bool,
    /// Whether this pass minted at least one Sub for this template — the end-of-pass fan-out
    /// sweep's gate. `false` at capture; the mint arm sets it.
    minted: bool,
}

impl Engine {
    /// Reconcile the discovery Profile's match set against the post-graft `current` — the
    /// [`Consequence::Reconcile`](crate::transitions) body. Mints a dynamic Sub per (chain terminus
    /// × template) that the dedup query doesn't already know; never removes anything.
    ///
    /// Registry/tree/attach work only — no burst-state writer: the caller (`fire_or_seal`) runs the
    /// silent seal *after* this returns, so the burst exits through the existing category-(a)
    /// terminus. The template set is derived from the live registry at entry, which makes the
    /// zombie case self-correcting: a template detached mid-burst (its cascade already reaped the
    /// minted set) is simply absent here, so the in-flight burst's reconcile mints nothing and the
    /// seal reaps the Profile.
    ///
    /// Each mint runs the ordinary `attach_sub_inner` pipeline, so a minted Profile enters its own
    /// cold Seed burst (probe emitted) within the same `StepOutput` — discovery's "fire" is a batch
    /// of attachments.
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
                    spec: Arc::clone(&t.spec),
                    cfg_hash: t.spec.identity.config_hash(),
                    name: s.name.clone(),
                    program: Arc::clone(&s.program),
                    scope: s.scope,
                    log_output: s.log_output,
                    minted: false,
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
            // independent of this arm), no kind is stamped, no Sub attaches. Minting one would be a
            // lie the state machine immediately unwinds: the kind projects to a File slot, the
            // minted Profile's first anchor probe `lstat`s and folds `Vanished` on `!is_file()`,
            // and the all-dynamic teardown reaps it — a mint→reap round-trip per chain event,
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

            let mut abs: Option<Arc<Path>> = None;
            for t in &mut templates {
                if self.discovery_already_minted(t.sid, slot, t.cfg_hash) {
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
                        t.spec.identity.clone(),
                        SubParams::minted(
                            synthesized,
                            Arc::clone(&t.program),
                            t.scope,
                            t.spec.settle,
                            t.log_output,
                            t.sid,
                        ),
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
                t.minted = true;
            }
        }

        // End-of-pass fan-out sweep, gated on minted-this-pass. The live count crosses the
        // threshold upward only via a mint and mints happen only in the loop above, so a pass that
        // minted nothing cannot be the crossing pass — quiet steady-state reconciles are scan-free
        // — while the check on the crossing pass itself still latches, and a mint-heavy pass still
        // scans once per template rather than once per mint.
        for t in templates.iter().filter(|t| t.minted) {
            self.maybe_warn_discovery_fanout(t.sid, out);
        }
    }

    /// Whether template `source` already has a live minted Sub anchored at `anchor` — the mint
    /// dedup gate, derived from `SubRegistry` truth (no cached map to drift).
    ///
    /// Resolves the same `(resource, config_hash)` partition `find_or_create_profile` keys on: a
    /// minted Sub for this `(template, anchor)` pair, if one exists, lives on the Profile at
    /// `(anchor, template.identity.config_hash())` tagged `source_discovery == Some(source)`. Cost
    /// is O(Subs on that one Profile) — single-digit in practice.
    fn discovery_already_minted(&self, source: SubId, anchor: ResourceId, cfg_hash: u64) -> bool {
        let Some(profile) = self.profiles.find(anchor, cfg_hash) else {
            return false;
        };
        self.subs.at(profile).iter().any(|&sid| {
            self.subs
                .get(sid)
                .is_some_and(|s| s.minted_by() == Some(source))
        })
    }

    /// Emit the one-shot [`Diagnostic::DiscoveryFanoutThreshold`] iff template `source`'s *live*
    /// minted-Sub count first crosses [`FANOUT_WARNING_THRESHOLD`].
    ///
    /// Two cheap gates bound the O(total Subs) count scan to passes that could actually cross: the
    /// call site runs this only for templates that minted this pass (the count moves upward only
    /// via a mint), and the template carrier's `fanout_warned` pre-gate below skips an
    /// already-warned (pathological) template on later minting passes. So the scan runs only on
    /// minting passes within the pre-warning prefix of each template's life.
    /// `SubRegistry::latch_fanout_warning`'s atomic check-and-latch remains the structural
    /// one-shot; both gates are additive, not its replacement.
    fn maybe_warn_discovery_fanout(&mut self, source: SubId, out: &mut StepOutput) {
        if self
            .subs
            .get(source)
            .is_none_or(|s| s.discovery_template().is_none_or(|t| t.fanout_warned))
        {
            return;
        }
        let count = self
            .subs
            .iter()
            .filter(|(_, s)| s.minted_by() == Some(source))
            .count();
        if let Some(count) = self
            .subs
            .latch_fanout_warning(source, FANOUT_WARNING_THRESHOLD, count)
        {
            out.diagnostics
                .push(Diagnostic::DiscoveryFanoutThreshold { source, count });
        }
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod discovery_tests;
