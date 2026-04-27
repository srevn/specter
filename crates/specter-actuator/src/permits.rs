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

use crossbeam::channel::{Receiver, Sender, bounded};

/// Counting semaphore.
///
/// Construct with `n.max(1)` tokens — a configured `0` would prevent
/// any spawns and is treated as a misconfiguration, not a feature.
#[derive(Debug)]
pub struct Permits {
    /// Receiver side: acquiring a token consumes one.
    rx: Receiver<()>,
    /// Sender side: releasing a token (via [`Permit::drop`]) returns one.
    /// Cloned into each [`Permit`] to keep RAII Drop simple.
    tx: Sender<()>,
}

impl Permits {
    /// Construct with `n` tokens (clamped to `>= 1`).
    #[must_use]
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
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
#[derive(Debug)]
pub struct Permit {
    tx: Sender<()>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        // If the [`Permits`] has been dropped (channel disconnected),
        // the send fails silently — the token vanishes with the
        // already-gone semaphore. Acceptable: actuator is being torn
        // down.
        let _ = self.tx.send(());
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

    #[test]
    fn permits_with_n_tokens_allows_n_acquisitions() {
        let p = Permits::new(3);
        let _a = p.try_acquire().expect("first acquire");
        let _b = p.try_acquire().expect("second acquire");
        let _c = p.try_acquire().expect("third acquire");
        assert!(p.try_acquire().is_none(), "fourth acquire fails");
    }

    #[test]
    fn permit_drop_releases_for_subsequent_acquire() {
        let p = Permits::new(1);
        {
            let _g = p.try_acquire().expect("acquire");
            assert!(p.try_acquire().is_none(), "second acquire fails while held");
        }
        assert!(p.try_acquire().is_some(), "released token reusable");
    }

    #[test]
    fn permits_with_zero_clamps_to_one() {
        let p = Permits::new(0);
        let _g = p.try_acquire().expect("clamped to one token");
        assert!(p.try_acquire().is_none(), "only one token");
    }

    #[test]
    fn permits_concurrent_acquire_release_safe() {
        let p = Arc::new(Permits::new(4));
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
        let p = Permits::new(2);
        let g = p.try_acquire().expect("acquire");
        drop(p);
        drop(g); // permit's send to a dropped Permits silently fails — no panic
    }

    #[test]
    fn permits_debug_does_not_drain_tokens() {
        let p = Permits::new(2);
        let _ = format!("{p:?}");
        let _a = p.try_acquire().expect("acquire post-Debug");
        let _b = p.try_acquire().expect("acquire post-Debug");
    }
}
