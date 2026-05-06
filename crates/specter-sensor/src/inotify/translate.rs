//! L4 translator: `(ClassSet, ResourceKind) → inotify mask`.
//!
//! Single platform-specific translation point for Linux. Mirror of
//! [`crate::kqueue::translate`]. The identity floor (D7) is OR-ed
//! unconditionally onto every registration so the engine's reconciler
//! always sees terminal events (delete / rename / unmount) regardless of
//! the user's class mask.
//!
//! ## Mapping table (D8 / §13 of the design doc)
//!
//! | Class       | Dir bits                                                  | File bits                          |
//! |-------------|-----------------------------------------------------------|------------------------------------|
//! | `STRUCTURE` | `IN_CREATE \| IN_DELETE \| IN_MOVED_FROM \| IN_MOVED_TO`  | ∅                                  |
//! | `CONTENT`   | ∅                                                         | `IN_MODIFY \| IN_CLOSE_WRITE`      |
//! | `METADATA`  | `IN_ATTRIB`                                               | `IN_ATTRIB`                        |
//!
//! Identity floor (always OR-ed): `IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT`.
//! Defensive flags (always OR-ed): `IN_DONT_FOLLOW | IN_EXCL_UNLINK`.
//!
//! Dir-anchored watches additionally OR `IN_ONLYDIR` at install time;
//! that lives in the watcher (Phase B6), not in the translator — it's an
//! `inotify_add_watch` directional flag, not part of the event mask.
//!
//! ## D10 parity
//!
//! kqueue's `NOTE_LINK` is kind-aware: STRUCTURE on Dir, METADATA on File.
//! inotify has no `NOTE_LINK` analogue — hardlink count changes on a File
//! surface as `IN_ATTRIB` (per `inotify(7)`: "link count since Linux
//! 2.6.25"), which the translator already routes through METADATA. The
//! kind-disambiguating branch the kqueue translator carries is therefore
//! collapsed: `IN_ATTRIB` is set whenever METADATA is requested,
//! regardless of kind.
//!
//! ## Unknown kind
//!
//! [`ResourceKind::Unknown`] (defensive path; the engine has not yet
//! classified the slot) collapses to `File` via [`ResourceKind::effective`]
//! — single source of truth shared with the kqueue translator and the
//! engine's L5 entry filter. STRUCTURE on Unknown therefore registers no
//! extra bits (Unknown ≡ File-shape; STRUCTURE is dir-only); CONTENT on
//! Unknown registers the file bits. This branch is effectively dead in
//! v1's flow (the watcher caches the observed kind from `fstat`), but
//! kept as forward-compatible defence for the descent placeholder edge
//! case.

use libc::{
    IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE, IN_DELETE_SELF, IN_DONT_FOLLOW,
    IN_EXCL_UNLINK, IN_MODIFY, IN_MOVE_SELF, IN_MOVED_FROM, IN_MOVED_TO, IN_UNMOUNT,
};
use specter_core::{ClassSet, ResourceKind};

/// Identity floor — OR-ed onto every inotify registration regardless of
/// the user's `events` mask. Per design D7, these three bits drive Tree
/// integrity (slot vacate on `IN_DELETE_SELF`) and watch lifecycle
/// (re-resolve on `IN_MOVE_SELF` / `IN_UNMOUNT`); they must always reach
/// the engine even when the user opted out of CONTENT / STRUCTURE /
/// METADATA. The engine's L5 entry filter folds terminal events into the
/// appropriate class for per-Profile filtering (CONTENT for files,
/// STRUCTURE for dirs); see `engine::transitions::fs_event_to_class`.
///
/// `IN_IGNORED` is *not* in this floor: it is not part of the user-
/// visible class surface. The kernel emits it unconditionally as a
/// cleanup signal when a watch descriptor is reaped; the watcher
/// (Phase B7) consumes it before normalization.
pub(super) const IDENTITY_FLOOR: u32 = IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT;

/// Defensive flags applied on every `inotify_add_watch`:
///
/// - `IN_DONT_FOLLOW` — never follow symlinks at the watched path.
///   Parity with kqueue's `O_NOFOLLOW` discipline; v1 has no
///   follow-symlinks opt-in.
/// - `IN_EXCL_UNLINK` — stop firing events on an unlinked-but-still-open
///   inode. kqueue auto-removes the registration when the fd closes;
///   inotify's path-based registration would otherwise keep delivering
///   `IN_MODIFY` on a deleted inode held open by a writer. Setting
///   `IN_EXCL_UNLINK` aligns inotify's semantics with kqueue's
///   "registration ends when the inode is unreachable."
pub(super) const ADD_WATCH_DEFENCE: u32 = IN_DONT_FOLLOW | IN_EXCL_UNLINK;

/// Compute the inotify mask for an `inotify_add_watch` call given the
/// user's [`ClassSet`] and the resource's kind. The result always
/// contains `IDENTITY_FLOOR | ADD_WATCH_DEFENCE`.
///
/// Pure / `const fn`: no I/O, no allocation, branches only on its
/// inputs. Excludes `IN_ONLYDIR` — that is an `add_watch` directional
/// flag the watcher OR-s post-fstat (Phase B6) once the kind has been
/// verified, not part of the event-mask translation.
#[must_use]
pub(super) const fn class_set_to_mask(events: ClassSet, kind: ResourceKind) -> u32 {
    let mut mask = IDENTITY_FLOOR | ADD_WATCH_DEFENCE;
    let effective = kind.effective();

    // STRUCTURE — Dir-only; child create/delete/move surface here.
    if events.intersects(ClassSet::STRUCTURE) && matches!(effective, ResourceKind::Dir) {
        mask |= IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO;
    }

    // CONTENT — File-only (Unknown collapses to File via `.effective()`).
    if events.intersects(ClassSet::CONTENT) && matches!(effective, ResourceKind::File) {
        mask |= IN_MODIFY | IN_CLOSE_WRITE;
    }

    // METADATA — both kinds. inotify has no NOTE_LINK analogue (hardlink
    // count changes surface as IN_ATTRIB on Linux), so the kind-aware
    // branching kqueue's translator carries collapses here.
    if events.intersects(ClassSet::METADATA) {
        mask |= IN_ATTRIB;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::{ADD_WATCH_DEFENCE, IDENTITY_FLOOR, class_set_to_mask};
    use libc::{
        IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE, IN_DELETE_SELF, IN_DONT_FOLLOW,
        IN_EXCL_UNLINK, IN_MODIFY, IN_MOVE_SELF, IN_MOVED_FROM, IN_MOVED_TO, IN_UNMOUNT,
    };
    use specter_core::{ClassSet, ResourceKind};

    /// `IDENTITY_FLOOR` membership is a load-bearing invariant for D7;
    /// pin it explicitly so a refactor that drops a bit fails this test.
    #[test]
    fn identity_floor_includes_delete_move_unmount() {
        assert_ne!(IDENTITY_FLOOR & IN_DELETE_SELF, 0, "DELETE_SELF missing");
        assert_ne!(IDENTITY_FLOOR & IN_MOVE_SELF, 0, "MOVE_SELF missing");
        assert_ne!(IDENTITY_FLOOR & IN_UNMOUNT, 0, "UNMOUNT missing");
        // The floor must NOT include any class-gated bits.
        assert_eq!(IDENTITY_FLOOR & IN_CREATE, 0, "floor leaks CREATE");
        assert_eq!(IDENTITY_FLOOR & IN_MODIFY, 0, "floor leaks MODIFY");
        assert_eq!(IDENTITY_FLOOR & IN_ATTRIB, 0, "floor leaks ATTRIB");
    }

    /// `ADD_WATCH_DEFENCE` membership is the kqueue-parity invariant
    /// (`IN_DONT_FOLLOW` mirrors `O_NOFOLLOW`, `IN_EXCL_UNLINK` mirrors
    /// kqueue's auto-detach on unlink-then-fd-close).
    #[test]
    fn add_watch_defence_pins_dont_follow_and_excl_unlink() {
        assert_ne!(ADD_WATCH_DEFENCE & IN_DONT_FOLLOW, 0, "DONT_FOLLOW missing");
        assert_ne!(ADD_WATCH_DEFENCE & IN_EXCL_UNLINK, 0, "EXCL_UNLINK missing");
    }

    /// EMPTY × any-kind always degrades to identity-floor + defence
    /// only. v1 fallback for fixture-defaulted `ClassSet::EMPTY`
    /// watches.
    #[test]
    fn empty_class_set_yields_floor_and_defence_only() {
        for kind in [ResourceKind::Dir, ResourceKind::File, ResourceKind::Unknown] {
            assert_eq!(
                class_set_to_mask(ClassSet::EMPTY, kind),
                IDENTITY_FLOOR | ADD_WATCH_DEFENCE,
                "EMPTY × {kind:?} leaked non-floor / non-defence bits"
            );
        }
    }

    // ── STRUCTURE ─────────────────────────────────────────────────────

    #[test]
    fn structure_on_dir_adds_create_delete_moved_from_to() {
        let m = class_set_to_mask(ClassSet::STRUCTURE, ResourceKind::Dir);
        assert_eq!(
            m,
            IDENTITY_FLOOR
                | ADD_WATCH_DEFENCE
                | IN_CREATE
                | IN_DELETE
                | IN_MOVED_FROM
                | IN_MOVED_TO,
            "STRUCTURE × Dir wrong"
        );
    }

    #[test]
    fn structure_on_file_is_noop() {
        // STRUCTURE is dir-only; on File it contributes no extra bits.
        assert_eq!(
            class_set_to_mask(ClassSet::STRUCTURE, ResourceKind::File),
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE,
        );
    }

    #[test]
    fn structure_on_unknown_is_noop() {
        // Unknown.effective() == File; STRUCTURE is dir-only ⇒ no extras.
        assert_eq!(
            class_set_to_mask(ClassSet::STRUCTURE, ResourceKind::Unknown),
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE,
        );
    }

    // ── CONTENT ───────────────────────────────────────────────────────

    #[test]
    fn content_on_file_adds_modify_close_write() {
        let m = class_set_to_mask(ClassSet::CONTENT, ResourceKind::File);
        assert_eq!(
            m,
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_MODIFY | IN_CLOSE_WRITE
        );
    }

    #[test]
    fn content_on_unknown_treated_as_file() {
        // Defensive: Unknown ⇒ File-shape; CONTENT contributes file bits.
        assert_eq!(
            class_set_to_mask(ClassSet::CONTENT, ResourceKind::Unknown),
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_MODIFY | IN_CLOSE_WRITE,
        );
    }

    #[test]
    fn content_on_dir_is_noop() {
        // CONTENT is file-only; on Dir it contributes no extra bits.
        assert_eq!(
            class_set_to_mask(ClassSet::CONTENT, ResourceKind::Dir),
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE,
        );
    }

    // ── METADATA ──────────────────────────────────────────────────────

    #[test]
    fn metadata_on_dir_adds_attrib() {
        let m = class_set_to_mask(ClassSet::METADATA, ResourceKind::Dir);
        assert_eq!(m, IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_ATTRIB);
    }

    #[test]
    fn metadata_on_file_adds_attrib() {
        // No NOTE_LINK analogue under inotify — IN_ATTRIB covers
        // hardlink count changes on its own (per inotify(7)). The kind-
        // aware branch the kqueue translator carries is collapsed here.
        let m = class_set_to_mask(ClassSet::METADATA, ResourceKind::File);
        assert_eq!(m, IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_ATTRIB);
    }

    #[test]
    fn metadata_on_unknown_treated_as_file() {
        let m = class_set_to_mask(ClassSet::METADATA, ResourceKind::Unknown);
        assert_eq!(m, IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_ATTRIB);
    }

    // ── Combinations ──────────────────────────────────────────────────

    #[test]
    fn structure_and_content_on_dir_only_structure_applies() {
        let m = class_set_to_mask(ClassSet::STRUCTURE | ClassSet::CONTENT, ResourceKind::Dir);
        assert_eq!(
            m,
            IDENTITY_FLOOR
                | ADD_WATCH_DEFENCE
                | IN_CREATE
                | IN_DELETE
                | IN_MOVED_FROM
                | IN_MOVED_TO,
        );
    }

    #[test]
    fn default_subtree_root_on_file_yields_content_only() {
        // STRUCTURE | CONTENT × File = CONTENT (STRUCTURE no-op on File).
        let m = class_set_to_mask(ClassSet::DEFAULT_SUBTREE_ROOT, ResourceKind::File);
        assert_eq!(
            m,
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_MODIFY | IN_CLOSE_WRITE
        );
    }

    #[test]
    fn default_per_file_yields_modify_close_write_attrib() {
        let m = class_set_to_mask(ClassSet::DEFAULT_PER_FILE, ResourceKind::File);
        assert_eq!(
            m,
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_MODIFY | IN_CLOSE_WRITE | IN_ATTRIB,
        );
    }

    #[test]
    fn all_classes_on_dir_yields_full_dir_mask() {
        let all = ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA;
        let m = class_set_to_mask(all, ResourceKind::Dir);
        // Dir gets STRUCTURE (CREATE | DELETE | MOVED_*) + METADATA
        // (ATTRIB); CONTENT is no-op on Dir.
        assert_eq!(
            m,
            IDENTITY_FLOOR
                | ADD_WATCH_DEFENCE
                | IN_CREATE
                | IN_DELETE
                | IN_MOVED_FROM
                | IN_MOVED_TO
                | IN_ATTRIB,
        );
    }

    #[test]
    fn all_classes_on_file_yields_full_file_mask() {
        let all = ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA;
        let m = class_set_to_mask(all, ResourceKind::File);
        // File gets CONTENT (MODIFY | CLOSE_WRITE) + METADATA (ATTRIB);
        // STRUCTURE is no-op on File.
        assert_eq!(
            m,
            IDENTITY_FLOOR | ADD_WATCH_DEFENCE | IN_MODIFY | IN_CLOSE_WRITE | IN_ATTRIB,
        );
    }

    /// Re-translation determinism: pure function ⇒ same input always
    /// yields the same mask. Protects the diff-skip optimization the
    /// watcher's `watch` (Phase B6) relies on.
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
                let a = class_set_to_mask(events, kind);
                let b = class_set_to_mask(events, kind);
                assert_eq!(a, b, "{events:?} × {kind:?} not deterministic");
            }
        }
    }
}
