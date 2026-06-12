//! The engine-obligation × walker-skip seam, end-to-end on a real filesystem.
//!
//! For an events-incomplete Standard burst the engine emits anchor-rooted
//! `ProofObligation::WholeSubtree` probes (`Profile::event_chains_prove_quiescence` fails), and
//! the verdict floor proves quiescence through the N=2 hash channel: two consecutive
//! Authoritative samples must agree. That proof is sound only if the walker delivers *full fresh
//! reads* under `WholeSubtree` — every frame refuses the mtime-skip against the stale baseline
//! (`Retry` never commits, so both samples carry the same pre-burst baseline). An in-place
//! rewrite of an existing file bumps no parent-dir mtime, so any frame cloned from the baseline
//! would hide it; under `Chains` the off-chain `data/` frame is exactly such a clone, and two
//! samples would agree on a tree that changed.
//!
//! Real `WorkerProber`, real `quiescence_verdict`: the off-chain in-place write landing between
//! two fixed-shape samples makes them disagree (fold = `Retry { observed_motion: true }` — no
//! false `Stable`), and once the writer stops, two agreeing samples fold `Stable(Natural)`.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{
    Input, ProbeCorrelation, ProbeOutcome, ProbeRequest, ProfileId, ProofAuthority,
    ProofObligation, QuiescenceVerdict, QuiescenceWitness, ScanConfig, StableReason, TreeSnapshot,
    quiescence_verdict,
};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
}

struct TestProberSink {
    tx: Sender<Input>,
}

impl ProberResponseSender for TestProberSink {
    fn send(&self, response: ProbeResponse) -> Result<(), SendError> {
        self.tx
            .send(Input::ProbeResponse(response))
            .map_err(|_| SendError::Disconnected)
    }
}

fn recv_snapshot(
    rx: &crossbeam::channel::Receiver<Input>,
) -> (Arc<specter_core::DirSnapshot>, ProofAuthority) {
    match rx
        .recv_timeout(Duration::from_secs(2))
        .expect("response within timeout")
    {
        Input::ProbeResponse(r) => match r.outcome {
            ProbeOutcome::SubtreeProven {
                snapshot,
                authority,
            } => (snapshot, authority),
            other => panic!("expected SubtreeProven, got {other:?}"),
        },
        other => panic!("unexpected input: {other:?}"),
    }
}

#[test]
fn whole_subtree_samples_witness_offchain_inplace_write() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    std::fs::create_dir(root.join("data")).unwrap();
    std::fs::write(root.join("data/blob.bin"), b"AAAA").unwrap();
    std::fs::create_dir(root.join("incoming")).unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(Arc::new(TestProberSink { tx }), nz(1)).unwrap();
    let p = fresh_profile_id();
    let scan_config = Arc::new(ScanConfig::builder().recursive(true).build());
    let target: Arc<Path> = Arc::from(root.to_path_buf());

    // The shape the engine emits for both the Seed walk and every events-incomplete Standard
    // sample: anchor-rooted, WholeSubtree-obligated, unforced. Only the baseline varies — the
    // Seed walk has none; the burst samples carry the stale pre-burst baseline (Retry never
    // commits, so it stays stale across the whole loop).
    let whole =
        |corr: u64, baseline: Option<Arc<specter_core::DirSnapshot>>| ProbeRequest::Subtree {
            owner: p,
            correlation: ProbeCorrelation::from(corr),
            target_path: Arc::clone(&target),
            anchor_path: Arc::clone(&target),
            scan_config: Arc::clone(&scan_config),
            captured_with: 0,
            baseline_subtree: baseline,
            obligation: ProofObligation::WholeSubtree,
            forced: false,
        };

    // Seed baseline over the resting tree.
    prober.submit(whole(1, None));
    let (s0, a0) = recv_snapshot(&rx);
    assert_eq!(a0, ProofAuthority::Authoritative);

    // The unrelated STRUCTURE point event that opens the Standard burst.
    std::fs::write(root.join("incoming/x"), b"x").unwrap();

    // Sample 1: prior = None primes the channel (absence of confirmation, not observed motion).
    prober.submit(whole(2, Some(Arc::clone(&s0))));
    let (s1, a1) = recv_snapshot(&rx);
    assert_eq!(a1, ProofAuthority::Authoritative);
    let h1 = TreeSnapshot::Dir(Arc::clone(&s1)).hash();
    assert!(
        matches!(
            quiescence_verdict(
                ProofAuthority::Authoritative,
                false,
                QuiescenceWitness::HashChannel {
                    prior: None,
                    response: h1,
                },
            ),
            QuiescenceVerdict::Retry {
                observed_motion: false,
            },
        ),
        "the first sample primes the channel without certifying",
    );

    // The mmap-style writer the events-incomplete mask cannot see lands BETWEEN the samples: an
    // in-place rewrite of an existing file. The leaf's own size/mtime move; the parent dir's
    // mtime does not — under a Chains obligation this frame would be cloned from the stale
    // baseline and the write would hide. The size change keeps the leaf hash distinct even
    // inside one mtime-granularity window.
    std::fs::write(root.join("data/blob.bin"), b"BBBBBBBB").unwrap();

    // Sample 2: the full fresh read sees the write — the samples disagree, and the channel
    // refuses to certify (`observed_motion: true` counts toward the mask-blindspot streak).
    prober.submit(whole(3, Some(Arc::clone(&s0))));
    let (s2, a2) = recv_snapshot(&rx);
    assert_eq!(a2, ProofAuthority::Authoritative);
    let h2 = TreeSnapshot::Dir(Arc::clone(&s2)).hash();
    assert_ne!(
        h1, h2,
        "WholeSubtree samples must observe the off-chain in-place write",
    );
    assert!(
        matches!(
            quiescence_verdict(
                ProofAuthority::Authoritative,
                false,
                QuiescenceWitness::HashChannel {
                    prior: Some(h1),
                    response: h2,
                },
            ),
            QuiescenceVerdict::Retry {
                observed_motion: true,
            },
        ),
        "the channel folds Retry on the disagreement — no false Stable",
    );

    // Writer stopped: the next sample agrees with the last, and the channel certifies.
    prober.submit(whole(4, Some(Arc::clone(&s0))));
    let (s3, a3) = recv_snapshot(&rx);
    assert_eq!(a3, ProofAuthority::Authoritative);
    let h3 = TreeSnapshot::Dir(s3).hash();
    assert_eq!(h2, h3, "a quiet tree yields agreeing samples");
    assert!(
        matches!(
            quiescence_verdict(
                ProofAuthority::Authoritative,
                false,
                QuiescenceWitness::HashChannel {
                    prior: Some(h2),
                    response: h3,
                },
            ),
            QuiescenceVerdict::Stable(StableReason::Natural),
        ),
        "two agreeing full fresh reads certify quiescence",
    );

    let _ = prober.shutdown();
}

const fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("non-zero literal in test fixture")
}
