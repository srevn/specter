//! kqueue-backed `FsWatcher` + `ConfigWatcher` — macOS / FreeBSD only.
//!
//! Layered (mirror of [`crate::inotify`]):
//! - [`ffi`]: thin libc wrappers (the only `unsafe` surface). Holds
//!   both the `kevent`-side primitives and the path → `OwnedFd` /
//!   `fstat`-kind helpers.
//! - [`normalize`]: kqueue flags → `FsEvent`.
//! - [`wake`]: cross-thread interruption via `EVFILT_USER`.
//! - [`watcher`]: state-bearing `FsWatcher` impl (engine-side).
//! - [`config_watch`]: state-bearing `ConfigWatcher` impl (auto-reload).
//!
//! The two watchers share the FFI surface but own *separate* kqueue
//! fds — independent kernel queues, independent wake idents, no
//! cross-talk. They are deliberately decoupled so the engine watcher's
//! per-resource bookkeeping (slotmap, kind cache, drain window) stays
//! out of the config watcher's tight blocking loop and vice versa.
//!
//! 32-bit BSD support is gated out at compile time — the engine
//! watcher's `udata` round-trip from `ResourceId.as_ffi() : u64` to
//! `void *` loses bits on 32-bit systems. v1 targets 64-bit only.
//! (The config watcher's `udata` constants `1` / `2` would survive a
//! 32-bit truncation, but keeping the gate uniform avoids a per-module
//! exception.)
#[cfg(target_pointer_width = "32")]
compile_error!(
    "specter-sensor: 32-bit targets are unsupported in v1 — `kqueue::udata` \
     would lose 32 bits of `ResourceId.as_ffi()`."
);

mod config_watch;
mod ffi;
mod normalize;
mod translate;
mod wake;
mod watcher;

pub use config_watch::KqueueConfigWatcher;
pub use watcher::KqueueWatcher;
