//! kqueue-backed `FsWatcher` — macOS / FreeBSD only.
//!
//! Layered:
//! - [`ffi`]: thin libc wrappers (the only `unsafe` surface).
//! - [`fd`]: path → `OwnedFd` open + `fstat` kind detection.
//! - [`normalize`]: kqueue flags → `FsEvent`.
//! - [`wake`]: cross-thread interruption via `EVFILT_USER`.
//! - [`watcher`]: state-bearing `FsWatcher` impl.
//!
//! 32-bit BSD support is gated out at compile time — the `udata`
//! round-trip from `ResourceId.as_ffi() : u64` to `void *` loses bits on
//! 32-bit systems. v1 targets 64-bit only.
#[cfg(target_pointer_width = "32")]
compile_error!(
    "specter-sensor: 32-bit targets are unsupported in v1 — `kqueue::udata` \
     would lose 32 bits of `ResourceId.as_ffi()`."
);

mod fd;
mod ffi;
mod normalize;
mod translate;
mod wake;
mod watcher;

#[cfg(test)]
mod tests;

pub use wake::KqueueWakeHandle;
pub use watcher::KqueueWatcher;
