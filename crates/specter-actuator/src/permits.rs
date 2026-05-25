//! Counting semaphore via a bounded crossbeam channel.
//!
//! Construction pre-fills the channel with N tokens; [`Permits::try_acquire`]
//! is non-blocking (`try_recv`); [`Permit::drop`] sends one token back
//! (`send`). RAII discipline means a permit is reliably released even on
//! panic.
//!
//! Non-blocking acquire is load-bearing for the controller: it must
//! never block on a permit, otherwise shutdown signals couldn't be
//! processed mid-block. The pump defers slots whose acquire failed back
//! to a transient buffer, restoring FIFO at end-of-pump.

use crossbeam::channel::{Receiver, Sender, TrySendError, bounded};
use std::num::NonZeroUsize;

/// Counting semaphore.
///
/// The `n: NonZeroUsize` constructor argument encodes the "at least one
/// permit" invariant in the type system; the public boundary
/// ([`crate::SubprocessActuator::new`]) takes [`NonZeroUsize`] directly,
/// so the invariant flows in by typing rather than by runtime sentinel.
#[derive(Debug)]
pub struct Permits {
    /// Receiver side: acquiring a token consumes one.
    rx: Receiver<()>,
    /// Sender side: releasing a token (via [`Permit::drop`]) returns one.
    /// Cloned into each [`Permit`] to keep RAII Drop simple.
    tx: Sender<()>,
}

impl Permits {
    /// Construct with `n` tokens.
    #[must_use]
    pub fn new(n: NonZeroUsize) -> Self {
        let n = n.get();
        let (tx, rx) = bounded::<()>(n);
        for _ in 0..n {
            tx.send(())
                .expect("bounded channel just constructed; send must succeed");
        }
        Self { rx, tx }
    }

    /// Non-blocking acquire. Returns `Some(Permit)` if a token was
    /// available; `None` if the semaphore is at capacity.
    #[must_use]
    pub fn try_acquire(&self) -> Option<Permit> {
        match self.rx.try_recv() {
            Ok(()) => Some(Permit {
                tx: self.tx.clone(),
            }),
            Err(_) => None,
        }
    }
}

/// RAII permit guard. Releases on drop.
#[must_use]
#[derive(Debug)]
pub struct Permit {
    tx: Sender<()>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        // Under our invariant we hold one of N tokens, so the channel
        // can never be at capacity here. `Full` ⇒ double-drop or
        // accounting bug — `debug_assert` in dev, silent in release
        // (discard the token rather than deadlock). `Disconnected` ⇒
        // [`Permits`] dropped first (actuator teardown); the token
        // vanishes with the already-gone semaphore.
        if self.tx.try_send(()) == Err(TrySendError::Full(())) {
            debug_assert!(
                false,
                "Permit::drop: channel at capacity; double-drop or accounting bug",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! Sibling unit tests for [`crate::permits`].

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    const fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test setup: n must be non-zero")
    }

    #[test]
    fn permits_with_n_tokens_allows_n_acquisitions() {
        let p = Permits::new(nz(3));
        let _a = p.try_acquire().expect("first acquire");
        let _b = p.try_acquire().expect("second acquire");
        let _c = p.try_acquire().expect("third acquire");
        assert!(p.try_acquire().is_none(), "fourth acquire fails");
    }

    #[test]
    fn permit_drop_releases_for_subsequent_acquire() {
        let p = Permits::new(nz(1));
        {
            let _g = p.try_acquire().expect("acquire");
            assert!(p.try_acquire().is_none(), "second acquire fails while held");
        }
        assert!(p.try_acquire().is_some(), "released token reusable");
    }

    #[test]
    fn permits_concurrent_acquire_release_safe() {
        let p = Arc::new(Permits::new(nz(4)));
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let p = Arc::clone(&p);
            let counter = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    loop {
                        if let Some(g) = p.try_acquire() {
                            counter.fetch_add(1, Ordering::SeqCst);
                            thread::sleep(Duration::from_micros(10));
                            drop(g);
                            break;
                        }
                        thread::yield_now();
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 8 * 50);
    }

    #[test]
    fn permits_drop_with_held_permits_does_not_panic() {
        let p = Permits::new(nz(2));
        let g = p.try_acquire().expect("acquire");
        drop(p);
        drop(g); // permit's send to a dropped Permits silently fails — no panic
    }

    #[test]
    fn permits_debug_does_not_drain_tokens() {
        let p = Permits::new(nz(2));
        let _ = format!("{p:?}");
        let _a = p.try_acquire().expect("acquire post-Debug");
        let _b = p.try_acquire().expect("acquire post-Debug");
    }
}
