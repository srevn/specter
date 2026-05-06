//! Translator: `(ClassSet, ResourceKind) → kqueue NOTE_* fflags`.
//!
//! Single platform-specific translation point. Backend-agnostic
//! [`ClassSet`] enters; kqueue-specific `u32` fflags leave. The identity
//! floor is OR-ed unconditionally onto every registration so the
//! engine's reconciler always sees terminal events (delete / rename /
//! revoke), independent of the user's class mask.
//!
//! Inotify's analogue would live in a sibling module
//! `sensor::inotify::translate`; the engine surface is unchanged.
//!
//! ## Mapping table
//!
//! | Class      | Dir bits                                  | File bits                          |
//! |------------|-------------------------------------------|------------------------------------|
//! | `STRUCTURE`| `NOTE_WRITE \| NOTE_EXTEND \| NOTE_LINK`  | ∅ (file-irrelevant)                |
//! | `CONTENT`  | ∅ (dir-irrelevant)                        | `NOTE_WRITE \| NOTE_EXTEND`        |
//! | `METADATA` | `NOTE_ATTRIB`                             | `NOTE_ATTRIB \| NOTE_LINK`         |
//!
//! `NOTE_LINK` placement is kind-aware: on a Dir it's a structural
//! signal (subdirectory `..` backref count change); on a File it's a
//! metadata signal (hardlink count change).
//!
//! `ResourceKind::Unknown` (defensive path; the sensor's own `fstat`
//! couldn't classify the inode as Dir or Reg) is treated as File via
//! [`ResourceKind::effective`] — single source of truth for that
//! convention. The engine's reconciler stamps the correct kind on its
//! next probe; if that flips Dir↔File the engine emits Unwatch + Watch
//! and the FD is reopened with the corrected mask. This branch is
//! effectively dead code in v1's flow but kept as forward-compatible
//! defense.

use libc::{
    NOTE_ATTRIB, NOTE_DELETE, NOTE_EXTEND, NOTE_LINK, NOTE_RENAME, NOTE_REVOKE, NOTE_WRITE,
};
use specter_core::{ClassSet, ResourceKind};

/// Identity floor — OR-ed onto every kqueue vnode registration regardless
/// of the user's `events` mask. These three NOTE bits drive **Tree
/// integrity** (slot vacate on delete) and **watch lifecycle**
/// (re-register on rename) and must always reach the engine even when the
/// user opted out of CONTENT / STRUCTURE / METADATA. The engine's entry
/// filter then folds terminal events into the appropriate class for
/// per-Profile filtering (CONTENT for files, STRUCTURE for dirs); see
/// `engine::transitions::fs_event_to_class`.
pub(super) const IDENTITY_FLOOR: u32 = NOTE_DELETE | NOTE_RENAME | NOTE_REVOKE;

/// Compute the kqueue fflags mask for a vnode registration given the
/// user's [`ClassSet`] and the resource's kind. The result always
/// contains [`IDENTITY_FLOOR`].
///
/// Pure function: no I/O, no allocation, branches only on its inputs.
/// `const` so the call in `KqueueWatcher::watch` can be inlined and
/// constant-folded for fixture cases.
#[must_use]
pub(super) const fn class_set_to_fflags(events: ClassSet, kind: ResourceKind) -> u32 {
    let mut fflags = IDENTITY_FLOOR;
    // Resolve the Unknown→File collapse once at the entry point; the
    // branches below decide on `effective` so the convention isn't
    // re-encoded per arm.
    let effective = kind.effective();

    // STRUCTURE — Dir-only. NOTE_LINK lands here on Dirs.
    if events.intersects(ClassSet::STRUCTURE) && matches!(effective, ResourceKind::Dir) {
        fflags |= NOTE_WRITE | NOTE_EXTEND | NOTE_LINK;
    }

    // CONTENT — File-only (Unknown collapses to File via `.effective()`).
    if events.intersects(ClassSet::CONTENT) && matches!(effective, ResourceKind::File) {
        fflags |= NOTE_WRITE | NOTE_EXTEND;
    }

    // METADATA — both kinds. NOTE_LINK lands here on Files;
    // Dir's NOTE_LINK is in the STRUCTURE branch above and we don't
    // re-add it here even when STRUCTURE isn't set, because LINK on a
    // Dir is not a metadata signal.
    if events.intersects(ClassSet::METADATA) {
        fflags |= NOTE_ATTRIB;
        if matches!(effective, ResourceKind::File) {
            fflags |= NOTE_LINK;
        }
    }

    fflags
}

#[cfg(test)]
mod tests {
    use super::{IDENTITY_FLOOR, class_set_to_fflags};
    use libc::{
        NOTE_ATTRIB, NOTE_DELETE, NOTE_EXTEND, NOTE_LINK, NOTE_RENAME, NOTE_REVOKE, NOTE_WRITE,
    };
    use specter_core::{ClassSet, ResourceKind};

    /// `IDENTITY_FLOOR` membership is a load-bearing invariant; pin it
    /// explicitly so a refactor that drops a bit is a test failure.
    #[test]
    fn identity_floor_includes_delete_rename_revoke() {
        assert_ne!(IDENTITY_FLOOR & NOTE_DELETE, 0, "DELETE missing from floor");
        assert_ne!(IDENTITY_FLOOR & NOTE_RENAME, 0, "RENAME missing from floor");
        assert_ne!(IDENTITY_FLOOR & NOTE_REVOKE, 0, "REVOKE missing from floor");
        // Floor does NOT include WRITE/EXTEND/ATTRIB/LINK — those are
        // class-gated.
        assert_eq!(IDENTITY_FLOOR & NOTE_WRITE, 0, "floor leaks WRITE");
        assert_eq!(IDENTITY_FLOOR & NOTE_EXTEND, 0, "floor leaks EXTEND");
        assert_eq!(IDENTITY_FLOOR & NOTE_ATTRIB, 0, "floor leaks ATTRIB");
        assert_eq!(IDENTITY_FLOOR & NOTE_LINK, 0, "floor leaks LINK");
    }

    /// EMPTY × any-kind always degrades to identity-floor only. This is
    /// the v1 fallback for fixture-defaulted `ClassSet::EMPTY` watches.
    #[test]
    fn empty_class_set_yields_identity_floor_only() {
        for kind in [ResourceKind::Dir, ResourceKind::File, ResourceKind::Unknown] {
            assert_eq!(
                class_set_to_fflags(ClassSet::EMPTY, kind),
                IDENTITY_FLOOR,
                "EMPTY × {kind:?} leaked non-floor bits"
            );
        }
    }

    // ── STRUCTURE ────────────────────────────────────────────────────

    #[test]
    fn structure_on_dir_adds_write_extend_link() {
        let f = class_set_to_fflags(ClassSet::STRUCTURE, ResourceKind::Dir);
        assert_eq!(
            f,
            IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND | NOTE_LINK,
            "STRUCTURE × Dir wrong"
        );
    }

    #[test]
    fn structure_on_file_is_noop() {
        // STRUCTURE is dir-only; on File it contributes no extra bits.
        assert_eq!(
            class_set_to_fflags(ClassSet::STRUCTURE, ResourceKind::File),
            IDENTITY_FLOOR
        );
    }

    #[test]
    fn structure_on_unknown_is_noop() {
        // Unknown defaults to File-shape; STRUCTURE is dir-only ⇒ no bits.
        assert_eq!(
            class_set_to_fflags(ClassSet::STRUCTURE, ResourceKind::Unknown),
            IDENTITY_FLOOR
        );
    }

    // ── CONTENT ──────────────────────────────────────────────────────

    #[test]
    fn content_on_file_adds_write_extend() {
        let f = class_set_to_fflags(ClassSet::CONTENT, ResourceKind::File);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND);
        assert_eq!(f & NOTE_LINK, 0, "CONTENT must not register LINK on File");
        assert_eq!(f & NOTE_ATTRIB, 0, "CONTENT must not register ATTRIB");
    }

    #[test]
    fn content_on_unknown_treated_as_file() {
        // Defensive: Unknown ⇒ File. CONTENT contributes the file bits.
        let f = class_set_to_fflags(ClassSet::CONTENT, ResourceKind::Unknown);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND);
    }

    #[test]
    fn content_on_dir_is_noop() {
        // CONTENT is file-only; on Dir it contributes no extra bits.
        assert_eq!(
            class_set_to_fflags(ClassSet::CONTENT, ResourceKind::Dir),
            IDENTITY_FLOOR
        );
    }

    // ── METADATA ─────────────────────────────────────────────────────

    #[test]
    fn metadata_on_dir_adds_attrib_only() {
        // NOTE_LINK on a Dir is STRUCTURE, not METADATA. So a
        // METADATA-only Dir does NOT register LINK.
        let f = class_set_to_fflags(ClassSet::METADATA, ResourceKind::Dir);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_ATTRIB);
        assert_eq!(f & NOTE_LINK, 0, "METADATA on Dir must not register LINK");
    }

    #[test]
    fn metadata_on_file_adds_attrib_and_link() {
        // NOTE_LINK on a File is METADATA (hardlink count change).
        let f = class_set_to_fflags(ClassSet::METADATA, ResourceKind::File);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_ATTRIB | NOTE_LINK);
    }

    #[test]
    fn metadata_on_unknown_treated_as_file() {
        let f = class_set_to_fflags(ClassSet::METADATA, ResourceKind::Unknown);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_ATTRIB | NOTE_LINK);
    }

    // ── Combinations ─────────────────────────────────────────────────

    #[test]
    fn structure_and_content_on_dir_only_structure_applies() {
        // STRUCTURE | CONTENT on Dir: STRUCTURE wins; CONTENT is no-op.
        let f = class_set_to_fflags(ClassSet::STRUCTURE | ClassSet::CONTENT, ResourceKind::Dir);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND | NOTE_LINK);
    }

    #[test]
    fn structure_and_content_on_file_only_content_applies() {
        // STRUCTURE | CONTENT on File: CONTENT wins; STRUCTURE is no-op.
        // This is the `DEFAULT_SUBTREE_ROOT` × file-leaf case.
        let f = class_set_to_fflags(ClassSet::DEFAULT_SUBTREE_ROOT, ResourceKind::File);
        assert_eq!(f, IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND);
        assert_eq!(f & NOTE_LINK, 0);
    }

    #[test]
    fn content_and_metadata_on_file_yields_full_file_mask() {
        // The `DEFAULT_PER_FILE` mask × File anchor.
        let f = class_set_to_fflags(ClassSet::DEFAULT_PER_FILE, ResourceKind::File);
        assert_eq!(
            f,
            IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK
        );
    }

    #[test]
    fn all_classes_on_dir_yields_full_dir_mask() {
        let all = ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA;
        let f = class_set_to_fflags(all, ResourceKind::Dir);
        // Dir gets STRUCTURE (W|E|L) + METADATA (A); CONTENT is no-op.
        assert_eq!(
            f,
            IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND | NOTE_LINK | NOTE_ATTRIB
        );
    }

    #[test]
    fn all_classes_on_file_yields_full_file_mask() {
        let all = ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA;
        let f = class_set_to_fflags(all, ResourceKind::File);
        // File gets CONTENT (W|E) + METADATA (A|L); STRUCTURE is no-op.
        // NOTE_LINK lands once via METADATA — same result.
        assert_eq!(
            f,
            IDENTITY_FLOOR | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK
        );
    }

    /// Re-translation determinism: the function is pure, so the same
    /// inputs always yield the same fflags. This protects the diff-skip
    /// optimization in `KqueueWatcher::watch`: two equal `(events, kind)`
    /// pairs must produce identical u32s.
    #[test]
    fn translate_is_deterministic() {
        for events in [
            ClassSet::EMPTY,
            ClassSet::STRUCTURE,
            ClassSet::CONTENT,
            ClassSet::METADATA,
            ClassSet::DEFAULT_SUBTREE_ROOT,
            ClassSet::DEFAULT_PER_FILE,
            ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA,
        ] {
            for kind in [ResourceKind::Dir, ResourceKind::File, ResourceKind::Unknown] {
                let a = class_set_to_fflags(events, kind);
                let b = class_set_to_fflags(events, kind);
                assert_eq!(a, b, "{events:?} × {kind:?} not deterministic");
            }
        }
    }
}
