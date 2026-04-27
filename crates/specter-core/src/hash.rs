//! Deterministic hash entry-points. SipHash-2-4 with fixed keys `(0, 0)`,
//! 64-bit and 128-bit variants.
//!
//! This is the *only* hashing in `core`/`engine` (I7). All call
//! sites — `config_hash`, `Snapshot::content_hash`, `dir_hash`, `leaf_hash`,
//! future stable digests — must use [`hasher`]/[`hash_one`] (64-bit) or
//! [`hasher_128`]/[`hash_one_128`] (128-bit) so output is reproducible
//! across processes and Rust versions.
//!
//! Rotating the keys changes every hash and is therefore a breaking change.
//!
//! ## When to pick which width
//!
//! - **64-bit (`hasher`/`hash_one`):** `config_hash`, `content_hash`.
//!   One hash per snapshot/profile lifetime; collisions are a once-per-
//!   process event, well below 2⁻³² per-pair risk.
//! - **128-bit (`hasher_128`/`hash_one_128`):** `dir_hash`, `leaf_hash`.
//!   These are computed at every level of the hierarchical snapshot,
//!   on every burst, for every Profile. The pair-comparison space is
//!   `O(levels × bursts × profiles)`; 64-bit collisions become probable
//!   over a long-running session and would mask real changes.
//!
//! ## `SystemTime` encoding
//!
//! [`hash_systemtime_into`] is the single canonical `SystemTime` encoder.
//! `std`'s `SystemTime: Hash` is platform-defined and not stable across
//! macOS / Linux / FreeBSD; this routes the same instant to the same bytes
//! everywhere via `(sign, secs, subsec_nanos)` decomposition relative to
//! `UNIX_EPOCH`.

use siphasher::sip::SipHasher24 as Sip64;
use siphasher::sip128::{Hasher128, SipHasher24 as Sip128};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

const KEY_0: u64 = 0;
const KEY_1: u64 = 0;

/// Fresh 64-bit SipHash-2-4 hasher with the project's pinned keys.
#[must_use]
pub fn hasher() -> Sip64 {
    Sip64::new_with_keys(KEY_0, KEY_1)
}

/// Single-shot 64-bit hash of any `Hash` value.
#[must_use]
pub fn hash_one<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut h = hasher();
    value.hash(&mut h);
    h.finish()
}

/// Fresh 128-bit SipHash-2-4 hasher with the project's pinned keys.
///
/// The returned `Sip128` (`siphasher::sip128::SipHasher24`) implements
/// `std::hash::Hasher`; pair it with [`Hasher128Ext::finish_128_u128`]
/// at finalisation to recover the full `u128`.
#[must_use]
pub fn hasher_128() -> Sip128 {
    Sip128::new_with_keys(KEY_0, KEY_1)
}

/// Single-shot 128-bit hash of any `Hash` value.
#[must_use]
pub fn hash_one_128<T: Hash + ?Sized>(value: &T) -> u128 {
    let mut h = hasher_128();
    value.hash(&mut h);
    h.finish_128_u128()
}

/// Extension trait for siphasher's 128-bit hashers: collapse the two-word
/// `Hash128` into a single `u128`.
///
/// The encoding mirrors `siphasher`'s upstream `From<Hash128> for u128`:
/// the low 64 bits are `Hash128.h1`, the high 64 bits are `Hash128.h2`.
/// Pinned via the `pinned_key_128_for_foo` golden — drift in the encoding
/// is a breaking change and must be accompanied by a golden refresh.
pub trait Hasher128Ext {
    /// Finalize and return the canonical 128-bit hash value.
    fn finish_128_u128(&self) -> u128;
}

impl Hasher128Ext for Sip128 {
    fn finish_128_u128(&self) -> u128 {
        u128::from(self.finish128())
    }
}

/// Cross-process-stable `SystemTime` encoding.
///
/// Decomposes to `(sign, secs, subsec_nanos)` relative to `UNIX_EPOCH`.
/// Pre-epoch instants take the `0u8` sign byte and encode the magnitude
/// returned by `SystemTimeError::duration`. Width-agnostic: works with
/// any `Hasher` (64-bit, 128-bit, std, custom) because it only uses the
/// trait's primitive `u8`/`u32`/`u64` writes.
pub fn hash_systemtime_into<H: Hasher>(t: SystemTime, h: &mut H) {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => {
            1u8.hash(h);
            d.as_secs().hash(h);
            d.subsec_nanos().hash(h);
        }
        Err(e) => {
            let d = e.duration();
            0u8.hash(h);
            d.as_secs().hash(h);
            d.subsec_nanos().hash(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Hasher128Ext, hash_one, hash_one_128, hasher, hasher_128};
    use std::hash::Hasher;

    /// Golden test: locks the SipHash-2-4 key choice `(0, 0)` against the
    /// stable `Hash for str` implementation. If this assertion ever fires,
    /// either the keys rotated (breaking change — bump every persisted
    /// `config_hash`) or `Hash for str` changed semantics (Rust ABI break).
    ///
    /// The pinned value was computed by running the test once with a
    /// placeholder; the failure prints the actual hash, which is then pasted
    /// here. Do not regenerate casually.
    #[test]
    fn pinned_key_for_foo() {
        let h = hash_one("foo");
        assert_eq!(
            h, 0xe1b1_9adf_b2e3_48a2,
            "hash_one(&\"foo\") drifted; key rotation or Hash impl change. got={h:#018x}",
        );
    }

    #[test]
    fn hasher_is_deterministic() {
        let mut a = hasher();
        let mut b = hasher();
        a.write(b"specter");
        b.write(b"specter");
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn distinct_inputs_distinct_hash() {
        assert_ne!(hash_one("alpha"), hash_one("beta"));
    }

    // ---------------------------------------------------------------------------
    // 128-bit
    // ---------------------------------------------------------------------------

    /// Mirrors `pinned_key_for_foo` for the 128-bit hasher: pins the keyed
    /// SipHash128-2-4 output for the input `"foo"`. Drift is a breaking
    /// change for every persisted `dir_hash`/`leaf_hash`.
    #[test]
    fn pinned_key_128_for_foo() {
        let h = hash_one_128("foo");
        assert_eq!(
            h, 0x0bcf_4e56_fefe_511b_8208_f531_8ff6_2bbc,
            "hash_one_128(&\"foo\") drifted; key rotation, encoding change, or \
             siphasher upgrade. got={h:#034x}",
        );
    }

    #[test]
    fn hasher_128_is_deterministic() {
        let mut a = hasher_128();
        let mut b = hasher_128();
        a.write(b"specter");
        b.write(b"specter");
        assert_eq!(a.finish_128_u128(), b.finish_128_u128());
    }

    #[test]
    fn distinct_inputs_distinct_hash_128() {
        assert_ne!(hash_one_128("alpha"), hash_one_128("beta"));
    }

    /// Read the high and low 64-bit halves of a `u128` without tripping
    /// `clippy::cast_possible_truncation`. The mask + `try_from` pattern is
    /// well-defined: the mask guarantees the result fits in `u64`.
    fn split_u128(v: u128) -> (u64, u64) {
        let hi = u64::try_from(v >> 64).expect("upper bits fit in u64 after shift");
        let lo = u64::try_from(v & u128::from(u64::MAX)).expect("masked low bits fit in u64");
        (hi, lo)
    }

    /// Sanity that the 128-bit output gives full 128 bits of entropy: the
    /// high 64 bits and the low 64 bits both vary across distinct inputs.
    /// A degenerate "two copies of the same u64" would only differ in one
    /// half between distinct inputs.
    #[test]
    fn hash_one_128_high_low_independent() {
        let (a_hi, a_lo) = split_u128(hash_one_128("alpha"));
        let (b_hi, b_lo) = split_u128(hash_one_128("beta"));
        assert_ne!(
            a_hi, b_hi,
            "high halves matched across distinct inputs — possible degenerate hasher",
        );
        assert_ne!(
            a_lo, b_lo,
            "low halves matched across distinct inputs — possible degenerate hasher",
        );
    }

    /// Cross-width separation: 128-bit and 64-bit hashers have different
    /// state sizes and finalisation; the low 64 bits of the 128-bit hash
    /// should never equal the 64-bit hash for the same input (other than by
    /// coincidence — pin one specific input as a regression check).
    #[test]
    fn hash_one_128_low_half_differs_from_hash_one() {
        let h64 = hash_one("foo");
        let (_, lo) = split_u128(hash_one_128("foo"));
        assert_ne!(
            h64, lo,
            "128-bit low half coincides with 64-bit hash for \"foo\" — \
             expected divergent state; encoding change?",
        );
    }
}
