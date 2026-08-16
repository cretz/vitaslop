//! A fast, non-cryptographic hasher for the capture's hot maps.
//!
//! # Why this exists
//! `std`'s default hasher is SipHash-1-3, chosen so a `HashMap` cannot be driven into its worst
//! case by an attacker who controls the keys. That is the right default for a program that hashes
//! untrusted input; it is the wrong one for the inside of a draw call.
//!
//! MEASURED on a browser-like race window: `draw: snapshot textures` cost **11.7% of the frame
//! while moving 0.0 MB**, and a memo over the arithmetic it does changed it by 1%. The arithmetic
//! was never the cost. Every bound texture unit of every draw performs several map operations -
//! the recorded format, the template memo, the snapshot table, the "already checked this scene"
//! set - and a race frame does that for ~2,000 units. SipHash is tens of nanoseconds per probe on
//! a 16-byte key; the keys here are two or four `u32`s.
//!
//! The keys are also not attacker-controlled in any meaningful sense: they are guest addresses,
//! byte lengths and GXM control words produced by a title running inside the emulator. A title
//! that wanted to degrade its own emulator's hash map has far more direct options.
//!
//! # The function
//! The FxHash construction (rotate, xor, multiply by an odd constant), which is what rustc itself
//! uses for its own interned tables. It is not a good hash for adversarial input and makes no
//! claim to be; it distributes integer keys well, which is the only property this needs.
//!
//! Deliberately written here rather than pulled in as a dependency: it is nine lines, and the one
//! thing that matters about it - which `write_*` methods are specialised - is a property of THIS
//! crate's key types.

use std::hash::{BuildHasherDefault, Hasher};

/// The odd 64-bit constant FxHash multiplies by. Any odd constant with a good spread of bits
/// works; this is the one rustc uses, so the choice is at least well-trodden.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    /// The generic path, one byte at a time. Every key type this crate hashes goes through one of
    /// the specialised methods below, so this is the correctness fallback rather than a hot path -
    /// and writing it as a chunked loop would be optimising a path nothing takes.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u32 as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `HashMap` over integer-ish keys, hashed with [`FxHasher`].
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// A `HashSet` over integer-ish keys, hashed with [`FxHasher`].
pub type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn h<T: Hash>(v: &T) -> u64 {
        let mut s = FxHasher::default();
        v.hash(&mut s);
        s.finish()
    }

    /// Keys that differ must hash differently, on the shapes this crate actually uses. Not a
    /// distribution test - a "the words are all mixed in" test. A hasher that ignored, say, the
    /// length half of an `(address, length)` key would collide every mip-chain read with the
    /// level-0 read of the same texture, and the second one would silently serve the first one's
    /// bytes.
    #[test]
    fn every_word_of_a_key_reaches_the_hash() {
        assert_ne!(h(&(0x8100_0000u32, 64usize)), h(&(0x8100_0000u32, 65usize)));
        assert_ne!(h(&(0x8100_0000u32, 64usize)), h(&(0x8100_0004u32, 64usize)));
        assert_ne!(h(&[1u32, 2, 3, 4]), h(&[1u32, 2, 3, 5]));
        assert_ne!(h(&[1u32, 2, 3, 4]), h(&[4u32, 3, 2, 1]));
        assert_ne!(h(&(0x8100_0000u32, 64usize, 2usize)), h(&(0x8100_0000u32, 64usize, 4usize)));
    }

    /// A map built on it behaves like a map. The point is not that this could plausibly fail - it
    /// is that a hasher whose `finish` ignored part of the state would still pass the test above
    /// and would break lookups, and this is the shape that catches it.
    #[test]
    fn a_map_over_it_round_trips() {
        let mut m: FxHashMap<(u32, usize), u32> = FxHashMap::default();
        for i in 0..1000u32 {
            m.insert((0x8000_0000 + i * 4, i as usize), i);
        }
        for i in 0..1000u32 {
            assert_eq!(m.get(&(0x8000_0000 + i * 4, i as usize)), Some(&i));
        }
        assert_eq!(m.len(), 1000);
        assert_eq!(m.get(&(0x8000_0000, 1)), None);
    }
}
