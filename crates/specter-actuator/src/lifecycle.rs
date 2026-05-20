//! One-shot lifecycle ratchet shared between a paired
//! ([`crate::spawner::ChildWaiter`], [`crate::spawner::ChildSignaler`]).
//!
//! The flag transitions `false → true` exactly once, when the child
//! has been reaped. Producer is either:
//!
//! - the wait thread — [`crate::spawner::ChildWaiter::wait`] after
//!   `child.wait()` returns (the normal path), or
//! - the recovery path — [`crate::spawner::ChildSignaler::reap_blocking`]
//!   on wait-thread-spawn-failure recovery.
//!
//! Consumers are [`crate::spawner::ChildSignaler::signal_term`] /
//! `signal_kill` / `is_dead`, the post-reap fast-path inside
//! `reap_blocking`, and the per-step timer thread
//! ([`crate::timer::run_timer`]).
//!
//! `Release`-on-store / `Acquire`-on-load matches the publish-subscribe
//! shape exactly: the writer publishes "child reaped; do not syscall
//! against this pid"; every reader synchronises-with that publish. The
//! prior code used `SeqCst` across every site — overkill for a flag
//! whose only contract is "reader sees `true` ⇒ producer has finished
//! the reap-side work".
//!
//! Production pairs are minted by [`crate::os::OsSpawner`]; the test
//! mock by [`crate::testkit::MockSpawner`]; the in-crate test fixtures
//! that exercise [`crate::pipe`] and [`crate::pool::state`] aggregation
//! mirror the same shape.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Paired-lifecycle dead-state ratchet. See module docs for the
/// producer/consumer contract.
///
/// Cloning shares the underlying cell — that is the *only* way to
/// hand a flag to the paired half of a [`crate::spawner::ChildWaiter`]
/// / [`crate::spawner::ChildSignaler`] pair. The newtype hides the
/// `Arc<AtomicBool>` and the memory ordering from callers so no site
/// can pick a weaker / stronger ordering by accident.
#[derive(Debug, Clone)]
pub(crate) struct DeadFlag(Arc<AtomicBool>);

impl DeadFlag {
    /// Mint a fresh, not-yet-dead flag.
    ///
    /// Deliberately no `Default` impl: every pair *must* share a single
    /// flag, and `DeadFlag::default()` would silently mint a fresh
    /// independent cell where a `.clone()` was meant — a protocol bug
    /// manifesting as "signals never short-circuit after reap". Forcing
    /// the explicit mint here, and sharing via [`Clone`], makes pairing
    /// visible at every construction site (`build_pair` in production;
    /// `allocate_spawn` in the mock).
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Mark the paired child as reaped. Idempotent.
    ///
    /// Call *after* the kernel has returned from `waitpid` (success or
    /// failure) on the normal path, or after a recovery `waitpid` on
    /// the [`crate::spawner::ChildSignaler::reap_blocking`] path. Once
    /// this returns, the pid is eligible for reuse and signal syscalls
    /// against it would race a recycled process — every consumer that
    /// observes `is_dead == true` must short-circuit instead of issuing
    /// `kill(2)`.
    pub(crate) fn mark_dead(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// `true` once the paired half has called [`Self::mark_dead`].
    /// `Acquire` synchronises-with the producer's `Release`-store.
    pub(crate) fn is_dead(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::DeadFlag;

    /// A fresh flag is not dead.
    #[test]
    fn new_is_not_dead() {
        assert!(!DeadFlag::new().is_dead());
    }

    /// `mark_dead` flips the flag exactly once and is idempotent.
    #[test]
    fn mark_dead_is_observable_and_idempotent() {
        let f = DeadFlag::new();
        f.mark_dead();
        assert!(f.is_dead());
        f.mark_dead();
        assert!(f.is_dead());
    }

    /// `clone` shares the underlying cell — marking through one handle
    /// is visible through the other. This is the load-bearing pairing
    /// invariant: a `(waiter, signaler)` pair built from a fresh flag
    /// and its clone observes the *same* dead state.
    #[test]
    fn clone_shares_state() {
        let producer = DeadFlag::new();
        let consumer = producer.clone();
        assert!(!consumer.is_dead());
        producer.mark_dead();
        assert!(consumer.is_dead());
    }
}
