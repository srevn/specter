//! inotify-backed `FsWatcher` + `ConfigWatcher` — Linux only.
//!
//! Layered (mirror of [`crate::kqueue`]):
//! - [`ffi`]: thin libc wrappers (the only `unsafe` surface).
//! - [`record`]: parser for the `inotify_event` record format.
//! - [`translate`]: `(ClassSet, ResourceKind) → mask`.
//! - [`normalize`]: `(mask, kind) → FsEvent`.
//! - [`watcher`]: state-bearing `FsWatcher` impl (engine-side).
//! - [`config_watch`]: state-bearing `ConfigWatcher` impl (auto-reload).
//!
//! The two watchers share the FFI surface but own *separate* inotify fds — independent kernel
//! queues, no cross-talk. They are deliberately decoupled so the engine watcher's per-resource
//! bookkeeping (slotmap, wd-routing table) stays out of the config watcher's drain path and vice
//! versa. Each inotify fd is exposed through the watcher's [`std::os::fd::AsFd`] supertrait so the
//! reactor can multiplex both with a single `mio::Poll` (or equivalent).
//!
//! 32-bit Linux is gated out at compile time. The inotify watcher itself does not require
//! pointer-width parity — `wd → ResourceId` routing pays a `BTreeMap` cell rather than stuffing the
//! id into a kernel-supplied slot. The gate exists to keep the rule uniform with the kqueue sibling
//! (which packs `ResourceId` into `kevent.udata`, a `*mut c_void`) so the test fixtures stay
//! 64-bit-only across both backends.

#[cfg(target_pointer_width = "32")]
compile_error!(
    "specter-sensor: 32-bit Linux is unsupported in v1 — kept uniform \
     with the kqueue branch; see the module docs."
);

mod config_watch;
mod ffi;
mod normalize;
mod record;
mod translate;
mod watcher;

pub use config_watch::InotifyConfigWatcher;
pub use watcher::InotifyWatcher;
