//! Re-registration on mask change — the per-resource `(wd, mask)` cache
//! + the kernel's "replace mask" semantics (per `inotify(7)`: "If the
//!   pathname referred to by pathname is already being watched, then the
//!   existing watch is updated").
//!
//! These tests exercise [`InotifyWatcher::watch`]'s re-watch path: a
//! second `watch()` call on a resource that already holds an entry. The
//! watcher diffs the cached mask against the translator's output for the
//! new `(events, kind)` and re-registers via `inotify_add_watch` when
//! they differ; the unchanged-mask fast path skips the syscall entirely.
//! Mirror of `kqueue_rewatch.rs`. Linux only.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a `mio::Poll` registered on the watcher's inotify fd. The
/// helper centralises the "register `as_fd()` for READABLE" boilerplate
/// the drain loops below share — `drain_ready` is non-blocking by
/// trait, so the caller blocks via the reactor.
fn poll_for(w: &InotifyWatcher) -> Poll {
    let poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register inotify fd");
    poll
}

/// Push every [`WatcherEvent::Fs`] in `buf` into `out`. Inotify can emit
/// [`WatcherEvent::Overflow`]; ignore it for these tests (we never push
/// the kernel beyond `max_queued_events`).
fn collect_fs(buf: &mut Vec<WatcherEvent>, out: &mut Vec<(ResourceId, FsEvent)>) {
    for ev in buf.drain(..) {
        if let WatcherEvent::Fs { resource, event } = ev {
            out.push((resource, event));
        }
    }
}

/// Drain at least one event matching `pred` or hit `overall` deadline.
fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut InotifyWatcher,
    pred: F,
    overall: Duration,
) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + overall;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        buf.clear();
        if w.drain_ready(&mut buf).is_err() {
            break;
        }
        collect_fs(&mut buf, &mut out);
        if out.iter().any(&pred) {
            return out;
        }
    }
    out
}

fn drain_for(w: &mut InotifyWatcher, dur: Duration) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + dur;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        buf.clear();
        if w.drain_ready(&mut buf).is_err() {
            break;
        }
        collect_fs(&mut buf, &mut out);
    }
    out
}

/// Re-watch with a widened mask should make new event classes deliverable.
/// Specifically: a CONTENT-only registration filters out
/// `MetadataChanged`; widening to `CONTENT | METADATA` (a fresh `Watch`
/// op the engine emits when `Resource.events_union` changes) must
/// re-register the wd with `IN_ATTRIB`, and a subsequent chmod then
/// fires `MetadataChanged`.
#[test]
fn rewatch_with_widened_mask_delivers_new_classes() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "initial").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First registration: CONTENT only. `IN_ATTRIB` is not installed,
    // so chmod must not fire `MetadataChanged`.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    // Drain any pending registration acks / spurious events.
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod — should NOT fire `MetadataChanged` since `IN_ATTRIB` isn't
    // registered.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms).unwrap();
    let early = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !early
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "CONTENT-only mask must not deliver MetadataChanged; got {early:?}"
    );

    // Re-register with widened mask. Same path, same resource — the
    // watcher takes the re-watch path, diffs cached mask vs new, and
    // re-registers via `inotify_add_watch` with `IN_ATTRIB` now in the
    // mask. The kernel's "replace mask" semantics return the same wd
    // for the same inode, so `wd == prior.wd` and the inode-swap branch
    // does not fire.
    w.watch(
        r,
        &path,
        ResourceKind::File,
        ClassSet::CONTENT | ClassSet::METADATA,
    )
    .unwrap();

    // Now chmod fires `MetadataChanged`.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o644);
    std::fs::set_permissions(&path, perms).unwrap();
    let post = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        post.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "after re-register with METADATA, chmod should fire MetadataChanged; got {post:?}"
    );

    drop(w);
}

/// Re-watch with a narrowed mask should drop classes from the
/// registration. Specifically: an initial `STRUCTURE | METADATA`
/// registration on a Dir delivers both `StructureChanged` on child write
/// and `MetadataChanged` on chmod. Re-registering with `STRUCTURE` only
/// must remove `IN_ATTRIB`, so chmod no longer fires.
#[test]
fn rewatch_with_narrowed_mask_drops_classes() {
    let tmp = TempDir::new().unwrap();
    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First: STRUCTURE | METADATA on the dir.
    w.watch(
        r,
        tmp.path(),
        ResourceKind::Dir,
        ClassSet::STRUCTURE | ClassSet::METADATA,
    )
    .unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // Verify METADATA is currently registered: chmod fires.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let pre = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        pre.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "pre-narrow: chmod should fire MetadataChanged; got {pre:?}"
    );

    // Narrow to STRUCTURE only. Re-watch path: re-registers without
    // `IN_ATTRIB`.
    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod again — must NOT fire MetadataChanged anymore.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let post = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !post
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "post-narrow: METADATA dropped, chmod must not fire MetadataChanged; got {post:?}"
    );

    // STRUCTURE still works — child write fires StructureChanged.
    std::fs::write(tmp.path().join("child.txt"), "x").unwrap();
    let post_struct = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::StructureChanged,
        Duration::from_secs(2),
    );
    assert!(
        post_struct
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::StructureChanged),
        "post-narrow: STRUCTURE still active, child write should fire StructureChanged; got {post_struct:?}"
    );

    drop(w);
}

/// Re-watch with the same mask is a no-op — the watcher's cache catches
/// it and skips the syscall. Observationally: the inode's behavior is
/// unchanged. We exercise this by checking that a normal write still
/// fires `Modified` after a same-mask re-watch (i.e., re-watching didn't
/// accidentally clear the mask).
#[test]
fn rewatch_with_same_mask_preserves_registration() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    // Same mask twice: hits the cache-diff `noop` branch.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post no-op rewatch, write should still fire Modified; got {out:?}"
    );

    drop(w);
}

/// Slow-path rewatch over a `cached_kind == Unknown` entry must derive
/// the install mask from the *observed* inode shape, not from the
/// stale Unknown cache. F-HIGH-1 regression guard.
///
/// Setup mirrors the audit's lifecycle: fresh-watch a Unix domain
/// socket (`fstat` collapses non-Dir / non-regular kinds to `Unknown`),
/// unlink + recreate a Dir at the same path *without* an intervening
/// drain so `by_resource[r]` keeps its Unknown cache, then rewatch
/// with events that force `compute_install_mask` to differ under
/// `cached_kind = Unknown`. The broken behaviour collapses
/// `STRUCTURE | METADATA × Unknown` to File-shape (`effective = File`
/// drops STRUCTURE; only `IN_ATTRIB` is added) and installs that on
/// the Dir — silently dropping `IN_CREATE | IN_DELETE | IN_MOVED_*`.
/// The fix reopens + fstats first, derives the mask from `observed =
/// Dir`, and installs `IN_CREATE` etc., so a child file create fires
/// `StructureChanged` on `r`.
#[test]
fn rewatch_after_unknown_cache_with_kind_flip_installs_observed_mask() {
    use std::os::unix::net::UnixListener;

    let tmp = TempDir::new().unwrap();
    let anchor = tmp.path().join("anchor");

    // Bind a Unix domain socket at `anchor`. The fstat of an
    // `O_PATH` fd over a socket inode returns `Unknown` (S_IFSOCK is
    // neither S_IFDIR nor S_IFREG), so the watcher caches
    // `kind = Unknown` on commit.
    let listener = UnixListener::bind(&anchor).expect("bind unix socket");

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Engine kind is also `Unknown` (descent-placeholder default for
    // an unprobed slot); `ClassSet::EMPTY` keeps the prior mask at
    // identity-floor + defence so the rewatch step is guaranteed to
    // change the mask.
    w.watch(r, &anchor, ResourceKind::Unknown, ClassSet::EMPTY)
        .expect("fresh-watch on socket should succeed under Unknown wildcard");

    // Replace the socket with a directory at the same path. Dropping
    // the listener closes its socket fd; `remove_file` unlinks the
    // inode (kernel queues IN_DELETE_SELF + IN_IGNORED on the prior
    // wd); `create_dir` installs a fresh Dir inode at `anchor`.
    //
    // No `drain_ready` between these steps and the rewatch below —
    // draining would consume the IN_IGNORED, the `IN_IGNORED` arm in
    // `drain_ready` would clear `by_resource[r]`, and the next watch
    // would route through fresh-watch (not rewatch). The bug only
    // bites when rewatch re-enters with the Unknown cache intact.
    drop(listener);
    std::fs::remove_file(&anchor).unwrap();
    std::fs::create_dir(&anchor).unwrap();

    // Rewatch with `STRUCTURE | METADATA`. Under `cached_kind =
    // Unknown` (broken path), `STRUCTURE` collapses to no-op
    // (effective File is structure-blind) and `METADATA` adds
    // `IN_ATTRIB` — install mask differs from the prior, forcing the
    // pre-fix code to reopen and install the File-shape mask on the
    // Dir. Under the fix, the slow path observes `Dir`, computes the
    // Dir-shape mask (`IN_CREATE | IN_DELETE | IN_MOVED_* | IN_ATTRIB
    // | IN_ONLYDIR` over the identity floor), and refreshes the cached
    // kind to Dir.
    w.watch(
        r,
        &anchor,
        ResourceKind::Unknown,
        ClassSet::STRUCTURE | ClassSet::METADATA,
    )
    .expect("rewatch over Unknown→Dir swap should install Dir-shape mask");

    // Child file create — only fires `StructureChanged` on `r` if
    // the install carried `IN_CREATE` on the Dir wd.
    std::fs::write(anchor.join("child.txt"), "x").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::StructureChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::StructureChanged),
        "post Unknown→Dir rewatch, child create should fire StructureChanged \
         (proves the install mask carried IN_CREATE — i.e., the observed kind \
         drove `compute_install_mask`, not the stale Unknown cache); got {out:?}"
    );

    drop(w);
}

/// `unwatch` clears the per-resource cache so a subsequent fresh `watch`
/// re-installs from scratch. Cache-lifecycle invariant.
#[test]
fn unwatch_then_watch_starts_fresh() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    w.unwatch(r);
    // Fresh watch (per-resource cache repopulated). Observable check: a
    // subsequent write fires `Modified` normally.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post unwatch+watch, write should fire Modified; got {out:?}"
    );

    drop(w);
}
