//! inotify mask + kind → [`FsEvent`]. Mirror of
//! [`crate::kqueue::normalize`].
//!
//! A single inotify event may carry multiple bits (the kernel coalesces
//! `IN_MODIFY | IN_CLOSE_WRITE` on close, for instance). We emit at most
//! one [`FsEvent`] per record by priority order:
//!
//! ```text
//! Revoked > Removed > Renamed > StructureChanged > Modified > MetadataChanged
//! ```
//!
//! The terminal flags (`IN_UNMOUNT` / `IN_DELETE_SELF` / `IN_MOVE_SELF`)
//! are slot-final — once one fires, no further events arrive on the
//! watch descriptor. Pairing a non-terminal flag with a terminal one
//! reports the terminal: emitting both would be noise the engine doesn't
//! act on. The non-terminal `Modified` / `MetadataChanged` coalescing is
//! acceptable because the engine's `Settling` state debounces either as
//! "something changed; reschedule."
//!
//! ## `IN_IGNORED`
//!
//! `IN_IGNORED` is sensor-internal — the watcher's
//! [`super::watcher::InotifyWatcher::poll_until`] consumes it before
//! invoking this translator. The defensive guard returning `None`
//! is kept so a slipped record (impossible under healthy invariants) is
//! silently dropped rather than mis-routed.
//!
//! ## kqueue parity
//!
//! kqueue's `NOTE_LINK` is kind-aware (Dir → StructureChanged, File →
//! MetadataChanged). inotify has no `NOTE_LINK` analogue; hardlink
//! count changes surface as `IN_ATTRIB` regardless of kind, so the
//! kind-disambiguating branch the kqueue normalizer carries is
//! collapsed: `IN_ATTRIB` always yields `MetadataChanged`. The
//! `Dir`-vs-`File` branch in the `IN_MODIFY | IN_CLOSE_WRITE` arm is
//! defensive — the translator never registers those bits on a Dir, but
//! a kernel anomaly that delivered them would still land on a sensible
//! `StructureChanged`.

use libc::{
    IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE, IN_DELETE_SELF, IN_IGNORED, IN_MODIFY,
    IN_MOVE_SELF, IN_MOVED_FROM, IN_MOVED_TO, IN_UNMOUNT,
};
use specter_core::{FsEvent, ResourceKind};

/// Map a single inotify record's mask + the watcher's cached kind to
/// at most one [`FsEvent`]. Returns `None` when:
///
/// - `IN_IGNORED` is set (sensor-internal cleanup signal),
/// - the mask carries no actionable bit (registration ack on a record
///   the kernel emitted with only orientation flags like `IN_ISDIR`).
///
/// Pure / `const fn`: branches only on its inputs.
#[must_use]
pub(super) const fn mask_to_fs_event(mask: u32, kind: ResourceKind) -> Option<FsEvent> {
    // IN_IGNORED is sensor-internal — the watcher consumes it before
    // calling here. Defensive guard: a slipped IN_IGNORED returns None
    // so it can't masquerade as a class-bearing event.
    if mask & IN_IGNORED != 0 {
        return None;
    }

    // Terminal flags first — slot-final, dominate any co-set bit.
    if mask & IN_UNMOUNT != 0 {
        return Some(FsEvent::Revoked);
    }
    if mask & IN_DELETE_SELF != 0 {
        return Some(FsEvent::Removed);
    }
    if mask & IN_MOVE_SELF != 0 {
        return Some(FsEvent::Renamed);
    }

    // Name-bearing structure events: parent's child set changed.
    // v1 collapses the cookie + name; the engine probes the parent on
    // every StructureChanged to discover what changed by name.
    if mask & (IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO) != 0 {
        return Some(FsEvent::StructureChanged);
    }

    // Resolve `Unknown` once via the canonical collapse — single source
    // of truth shared with the translator and the engine's class filter.
    let effective = kind.effective();

    // Non-terminal CONTENT bits collapse on kind. The translator
    // requests these only on File watches, so the Dir arm is defensive
    // (kernel-anomaly safety net): a stray IN_MODIFY on a Dir folds
    // into StructureChanged, matching kqueue's NOTE_WRITE-on-Dir
    // policy.
    if mask & (IN_MODIFY | IN_CLOSE_WRITE) != 0 {
        return Some(if matches!(effective, ResourceKind::Dir) {
            FsEvent::StructureChanged
        } else {
            FsEvent::Modified
        });
    }

    if mask & IN_ATTRIB != 0 {
        return Some(FsEvent::MetadataChanged);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::mask_to_fs_event;
    use libc::{
        IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE, IN_DELETE_SELF, IN_IGNORED, IN_MODIFY,
        IN_MOVE_SELF, IN_MOVED_FROM, IN_MOVED_TO, IN_UNMOUNT,
    };
    use specter_core::{FsEvent, ResourceKind};

    // ── Terminal priority ─────────────────────────────────────────────

    #[test]
    fn unmount_yields_revoked() {
        assert_eq!(
            mask_to_fs_event(IN_UNMOUNT, ResourceKind::File),
            Some(FsEvent::Revoked),
        );
    }

    #[test]
    fn delete_self_yields_removed() {
        assert_eq!(
            mask_to_fs_event(IN_DELETE_SELF, ResourceKind::Dir),
            Some(FsEvent::Removed),
        );
    }

    #[test]
    fn move_self_yields_renamed() {
        assert_eq!(
            mask_to_fs_event(IN_MOVE_SELF, ResourceKind::File),
            Some(FsEvent::Renamed),
        );
    }

    #[test]
    fn unmount_dominates_co_set_bits() {
        // IN_UNMOUNT > IN_DELETE_SELF > IN_MOVE_SELF > non-terminal.
        let mask = IN_UNMOUNT | IN_DELETE_SELF | IN_MOVE_SELF | IN_MODIFY;
        assert_eq!(
            mask_to_fs_event(mask, ResourceKind::File),
            Some(FsEvent::Revoked),
        );
    }

    #[test]
    fn delete_self_dominates_move_self() {
        let mask = IN_DELETE_SELF | IN_MOVE_SELF;
        assert_eq!(
            mask_to_fs_event(mask, ResourceKind::File),
            Some(FsEvent::Removed),
        );
    }

    #[test]
    fn delete_self_dominates_modify() {
        let mask = IN_DELETE_SELF | IN_MODIFY;
        assert_eq!(
            mask_to_fs_event(mask, ResourceKind::File),
            Some(FsEvent::Removed),
        );
    }

    // ── Structure events ──────────────────────────────────────────────

    #[test]
    fn create_delete_moved_yield_structure_changed() {
        for m in [IN_CREATE, IN_DELETE, IN_MOVED_FROM, IN_MOVED_TO] {
            assert_eq!(
                mask_to_fs_event(m, ResourceKind::Dir),
                Some(FsEvent::StructureChanged),
                "{m:#x} should yield StructureChanged"
            );
        }
    }

    #[test]
    fn structure_dominates_content_and_metadata() {
        let mask = IN_CREATE | IN_MODIFY | IN_ATTRIB;
        assert_eq!(
            mask_to_fs_event(mask, ResourceKind::Dir),
            Some(FsEvent::StructureChanged),
        );
    }

    // ── Content events ────────────────────────────────────────────────

    #[test]
    fn modify_close_write_yield_modified_on_file() {
        for m in [IN_MODIFY, IN_CLOSE_WRITE, IN_MODIFY | IN_CLOSE_WRITE] {
            assert_eq!(
                mask_to_fs_event(m, ResourceKind::File),
                Some(FsEvent::Modified),
                "{m:#x} on File should yield Modified"
            );
        }
    }

    #[test]
    fn modify_close_write_on_unknown_yields_modified() {
        // Unknown.effective() == File ⇒ Modified.
        for m in [IN_MODIFY, IN_CLOSE_WRITE] {
            assert_eq!(
                mask_to_fs_event(m, ResourceKind::Unknown),
                Some(FsEvent::Modified),
            );
        }
    }

    #[test]
    fn modify_close_write_on_dir_yields_structure_changed_defensively() {
        // The translator never registers MODIFY/CLOSE_WRITE on Dir, but
        // a kernel anomaly that delivered one should still land on a
        // sensible signal. Mirrors kqueue's NOTE_WRITE-on-Dir rule.
        for m in [IN_MODIFY, IN_CLOSE_WRITE] {
            assert_eq!(
                mask_to_fs_event(m, ResourceKind::Dir),
                Some(FsEvent::StructureChanged),
            );
        }
    }

    // ── Metadata events ───────────────────────────────────────────────

    #[test]
    fn attrib_yields_metadata_changed() {
        for kind in [ResourceKind::Dir, ResourceKind::File, ResourceKind::Unknown] {
            assert_eq!(
                mask_to_fs_event(IN_ATTRIB, kind),
                Some(FsEvent::MetadataChanged),
                "IN_ATTRIB × {kind:?} should yield MetadataChanged"
            );
        }
    }

    #[test]
    fn modify_dominates_attrib_on_file() {
        let mask = IN_MODIFY | IN_ATTRIB;
        assert_eq!(
            mask_to_fs_event(mask, ResourceKind::File),
            Some(FsEvent::Modified),
        );
    }

    // ── IN_IGNORED + empty ────────────────────────────────────────────

    #[test]
    fn ignored_returns_none() {
        // IN_IGNORED dominates even paired with a class-bearing bit;
        // the watcher consumes it pre-normalize, but the defensive
        // guard catches any slipped record.
        for paired in [0, IN_MODIFY, IN_CREATE, IN_ATTRIB] {
            assert_eq!(
                mask_to_fs_event(IN_IGNORED | paired, ResourceKind::File),
                None,
                "IN_IGNORED | {paired:#x} should suppress event"
            );
        }
    }

    #[test]
    fn empty_mask_returns_none() {
        // No actionable bits ⇒ no event.
        assert_eq!(mask_to_fs_event(0, ResourceKind::File), None);
        assert_eq!(mask_to_fs_event(0, ResourceKind::Dir), None);
        assert_eq!(mask_to_fs_event(0, ResourceKind::Unknown), None);
    }
}
