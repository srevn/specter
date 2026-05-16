//! The stable-digest seam — the *only* fingerprinting in `core`/`engine`
//! (I7).
//!
//! Every digest (`config_hash`, `dir_hash`, `leaf_hash`, future stable
//! digests) is built on SipHash-2-4 with the algorithm's default
//! (unkeyed) initialisation, through [`StableHasher`].
//!
//! ## Why a seam, not `std::hash::Hash`
//!
//! `std::hash::Hash` is explicitly *not* a wire format: `Hash`
//! impls and the `Hasher::write_uXX` defaults are platform- and
//! version-defined (e.g. `write_u128` defaults to native-endian
//! `to_ne_bytes`; `siphasher` 1.0.3 specialises `write_u8/16/32/64`
//! little-endian but leaves `write_u128`/`write_str` on the
//! native-endian/std defaults). A digest built through blanket `Hash`
//! is therefore *not* reproducible across processes or architectures —
//! the property every Specter digest comparison depends on.
//!
//! [`StableHasher`] owns the bytes. Every primitive is the value's
//! **explicit little-endian image** fed to the SipHash-2-4
//! `write(&[u8])` core; the seam never calls a width-specialised or
//! native-endian `write_uXX`. Stability therefore rests solely on the
//! SipHash-2-4 compression core and finalisation — pinned by the
//! `SipHasher24` type aliases (algorithm) and by the production golden
//! tests (`dir_hash`/`leaf_hash`/`config_hash`) against crate-upgrade
//! drift. On a little-endian target the seam is byte-identical to the
//! pre-seam blanket-`Hash` fold (so the goldens do not shift); on
//! big-endian it is *correct* where the old native-endian `write_u128`
//! was not. There is no blanket `Hash` route and no native-endian
//! escape: the inner hasher is only ever driven by `write(&[u8])`.
//!
//! These are *fingerprints*, not MACs. There are no secret keys and no
//! adversarial-collision guarantees: every digest is computed over
//! operator-controlled inputs (config files, kernel-observed filesystem
//! state) and consumed in-process for equivalence checks and `BTreeMap`
//! lookups. SipHash-2-4 is chosen for its prefix-freeness, well-studied
//! diffusion, and small state — properties the hierarchical `dir_hash`
//! fold leans on — not for keyed authentication.
//!
//! ## When to pick which width
//!
//! - **64-bit ([`hasher`] → [`StableHasher::finish_u64`]):**
//!   `config_hash`. One digest per Profile lifetime; collisions are a
//!   once-per-process event, well below 2⁻³² per-pair risk.
//! - **128-bit ([`hasher_128`] → [`StableHasher::finish_u128`]):**
//!   `dir_hash`, `leaf_hash`. Computed at every level of the
//!   hierarchical snapshot, on every burst, for every Profile. The
//!   pair-comparison space is `O(levels × bursts × profiles)`; 64-bit
//!   collisions become probable over a long-running session and would
//!   mask real changes.
//!
//! ## Composite encoders
//!
//! The seam is a primitive-only byte canonicaliser with zero domain
//! knowledge. Composite types encode through a single named function
//! sited beside the type, composed only of seam primitives:
//! [`put_systemtime_into`] here (a `std` type), and
//! [`crate::fs_id::encode_into`] beside `FsIdentity`. Callers fold
//! everything explicitly — there is no blanket `Hash` shortcut, which
//! is what makes a native-endian width unconstructable.

use siphasher::sip::SipHasher24 as Sip64;
use siphasher::sip128::{Hasher128, SipHasher24 as Sip128};
use std::fmt;
use std::hash::Hasher;
use std::time::{SystemTime, UNIX_EPOCH};

/// The stable-digest seam: a byte-canonical encoder over SipHash-2-4.
///
/// `H` is the pinned algorithm — [`Sip64`] (64-bit) or [`Sip128`]
/// (128-bit). The inner hasher is private and is only ever driven by
/// `write(&[u8])`, so a width-specialised or native-endian path is
/// *unconstructable*. Build one via [`hasher`] / [`hasher_128`];
/// finalise via [`StableHasher::finish_u64`] / [`finish_u128`].
///
/// Every `put_*` writes the value's explicit little-endian byte image.
/// On a little-endian target this is byte-identical to the historical
/// blanket-`Hash` fold (`write_u8` / little-endian `write_uXX` /
/// `str`'s `write(bytes)`+`0xff`); the difference is structural — the
/// encoding is now endian-explicit and reproducible across processes
/// and architectures by construction.
pub struct StableHasher<H>(H);

impl<H> fmt::Debug for StableHasher<H> {
    /// Opaque: a stable-digest accumulator's mid-fold SipHash state
    /// carries no caller-meaningful information and is deliberately not
    /// rendered (also keeps the impl free of an `H: Debug` bound).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StableHasher").finish_non_exhaustive()
    }
}

impl StableHasher<Sip64> {
    /// Finalise the 64-bit fingerprint.
    #[must_use]
    pub fn finish_u64(&self) -> u64 {
        self.0.finish()
    }
}

impl StableHasher<Sip128> {
    /// Finalise the 128-bit fingerprint.
    ///
    /// Collapses `siphasher`'s two-word `Hash128` into a single `u128`
    /// as `(h1 as u128) | ((h2 as u128) << 64)` — the upstream
    /// `From<Hash128> for u128`.
    #[must_use]
    pub fn finish_u128(&self) -> u128 {
        u128::from(self.0.finish128())
    }
}

impl<H: Hasher> StableHasher<H> {
    /// Fold one byte.
    pub fn put_u8(&mut self, v: u8) {
        self.0.write(&[v]);
    }

    /// Fold a `u32` as its little-endian byte image.
    pub fn put_u32(&mut self, v: u32) {
        self.0.write(&v.to_le_bytes());
    }

    /// Fold a `u64` as its little-endian byte image.
    pub fn put_u64(&mut self, v: u64) {
        self.0.write(&v.to_le_bytes());
    }

    /// Fold a `u128` as its little-endian byte image.
    ///
    /// This is the load-bearing fix: the byte image equals
    /// `put_u64(v as u64); put_u64((v >> 64) as u64)` and, on a
    /// little-endian target, the historical native-endian `write_u128`
    /// — but it is endian-explicit, so it is also correct on
    /// big-endian, where the old path silently diverged.
    pub fn put_u128(&mut self, v: u128) {
        self.0.write(&v.to_le_bytes());
    }

    /// Fold a string: its UTF-8 bytes followed by the `0xff`
    /// prefix-free terminator — byte-identical to `std`'s
    /// `Hash for str` / `write_str` default, made explicit so the
    /// encoding cannot drift with a `std`/`siphasher` change.
    pub fn put_str(&mut self, s: &str) {
        self.0.write(s.as_bytes());
        self.0.write(&[0xFF]);
    }
}

/// A fresh 64-bit stable-digest hasher.
#[must_use]
pub fn hasher() -> StableHasher<Sip64> {
    StableHasher(Sip64::new())
}

/// A fresh 128-bit stable-digest hasher.
#[must_use]
pub fn hasher_128() -> StableHasher<Sip128> {
    StableHasher(Sip128::new())
}

/// Cross-process-stable `SystemTime` encoder — the single canonical
/// route (`std`'s `SystemTime: Hash` is platform-defined and not stable
/// across macOS / Linux / FreeBSD).
///
/// Decomposes to `(sign, secs, subsec_nanos)` relative to `UNIX_EPOCH`,
/// folded as `put_u8` / `put_u64` / `put_u32`. Pre-epoch instants take
/// the `0u8` sign byte and encode the magnitude returned by
/// `SystemTimeError::duration`; post-epoch take `1u8`.
pub fn put_systemtime_into<H: Hasher>(t: SystemTime, h: &mut StableHasher<H>) {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => {
            h.put_u8(1);
            h.put_u64(d.as_secs());
            h.put_u32(d.subsec_nanos());
        }
        Err(e) => {
            let d = e.duration();
            h.put_u8(0);
            h.put_u64(d.as_secs());
            h.put_u32(d.subsec_nanos());
        }
    }
}

#[cfg(test)]
mod seam_tests {
    use super::{hasher, hasher_128, put_systemtime_into};
    use siphasher::sip::SipHasher24 as RawSip64;
    use siphasher::sip128::Hasher128;
    use siphasher::sip128::SipHasher24 as RawSip128;
    use std::hash::{Hash, Hasher};
    use std::time::{Duration, UNIX_EPOCH};

    /// Reference: a *raw* SipHash-2-4 hasher fed an explicit
    /// little-endian byte image through its `write(&[u8])` core. The
    /// seam's contract is to be byte-identical to this — it owns the
    /// bytes, it never delegates to a width-specialized or
    /// native-endian `write_uXX`.
    fn raw128_le(image: &[u8]) -> u128 {
        let mut h = RawSip128::new();
        h.write(image);
        u128::from(h.finish128())
    }

    /// Every primitive is the value's explicit little-endian image fed
    /// to the SipHash-2-4 `write([u8])` core — proven against a raw
    /// hasher consuming the hand-built byte image.
    #[test]
    fn primitives_are_explicit_little_endian() {
        let mut s = hasher_128();
        s.put_u8(0xAB);
        s.put_u32(0x1122_3344);
        s.put_u64(0x0102_0304_0506_0708);
        s.put_u128(0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
        s.put_str("name");

        let mut image = vec![0xABu8];
        image.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        image.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        image.extend_from_slice(&0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100u128.to_le_bytes());
        image.extend_from_slice(b"name");
        image.push(0xFF); // str prefix-free terminator

        assert_eq!(s.finish_u128(), raw128_le(&image));
    }

    /// Load-bearing: `put_u128` is exactly `put_u64(lo); put_u64(hi)`.
    /// On a little-endian target this reproduces today's native
    /// `write_u128` byte-for-byte while being correct on big-endian.
    #[test]
    fn put_u128_equals_split_low_then_high() {
        let v: u128 = 0xdead_beef_0bad_f00d_1357_9bdf_2468_ace0;
        let lo = u64::try_from(v & u128::from(u64::MAX)).expect("low 64 bits fit u64");
        let hi = u64::try_from(v >> 64).expect("high 64 bits fit u64 after shift");
        let mut a = hasher_128();
        a.put_u128(v);
        let mut b = hasher_128();
        b.put_u64(lo);
        b.put_u64(hi);
        assert_eq!(a.finish_u128(), b.finish_u128());
    }

    /// `put_str` reproduces std `Hash for str` (the bytes followed by
    /// the `0xff` prefix-free terminator) so directory-name and
    /// glob-source folds are byte-identical to the pre-seam path.
    #[test]
    fn put_str_matches_std_str_hash() {
        let mut s = hasher();
        s.put_str("foo/bar.rs");
        let mut r = RawSip64::new();
        "foo/bar.rs".hash(&mut r);
        assert_eq!(s.finish_u64(), r.finish());
    }

    #[test]
    fn put_str_is_prefix_free() {
        let mut ab = hasher();
        ab.put_str("ab");
        let mut a_b = hasher();
        a_b.put_str("a");
        a_b.put_str("b");
        assert_ne!(ab.finish_u64(), a_b.finish_u64());
    }

    /// `put_systemtime_into` is byte-identical to the legacy
    /// `(sign, secs, subsec)` decomposition folded via the primitive
    /// `u8`/`u64`/`u32` writes.
    #[test]
    fn put_systemtime_post_epoch_matches_legacy() {
        let t = UNIX_EPOCH + Duration::new(1_700_000_000, 123_456_789);
        let mut s = hasher_128();
        put_systemtime_into(t, &mut s);

        let mut r = RawSip128::new();
        1u8.hash(&mut r);
        1_700_000_000u64.hash(&mut r);
        123_456_789u32.hash(&mut r);
        assert_eq!(s.finish_u128(), u128::from(r.finish128()));
    }

    #[test]
    fn put_systemtime_pre_epoch_uses_zero_sign() {
        let t = UNIX_EPOCH - Duration::new(5, 250);
        let mut s = hasher_128();
        put_systemtime_into(t, &mut s);

        let mut r = RawSip128::new();
        0u8.hash(&mut r);
        5u64.hash(&mut r);
        250u32.hash(&mut r);
        assert_eq!(s.finish_u128(), u128::from(r.finish128()));
    }

    #[test]
    fn finish_is_deterministic_both_widths() {
        let mut a = hasher();
        let mut b = hasher();
        a.put_str("specter");
        b.put_str("specter");
        assert_eq!(a.finish_u64(), b.finish_u64());

        let mut c = hasher_128();
        let mut d = hasher_128();
        c.put_str("specter");
        d.put_str("specter");
        assert_eq!(c.finish_u128(), d.finish_u128());
    }

    #[test]
    fn distinct_inputs_distinct_digest() {
        let mut a = hasher_128();
        a.put_str("alpha");
        let mut b = hasher_128();
        b.put_str("beta");
        assert_ne!(a.finish_u128(), b.finish_u128());
    }

    /// The 64-bit and 128-bit widths are independent finalisations: the
    /// low 64 bits of the 128-bit digest do not coincide with the
    /// 64-bit digest over the same fold.
    #[test]
    fn widths_have_independent_finalisation() {
        let mut wide = hasher_128();
        wide.put_str("foo");
        let lo =
            u64::try_from(wide.finish_u128() & u128::from(u64::MAX)).expect("low 64 bits fit u64");
        let mut narrow = hasher();
        narrow.put_str("foo");
        assert_ne!(lo, narrow.finish_u64());
    }
}
