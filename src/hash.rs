//! Hashing for integer-keyed tables (packed voxel coordinates): one 64-bit finalizer (murmur3
//! fmix64) per key. std's SipHash is several times slower on the millions of lookups a felling
//! or a flood does, and these keys are not attacker-controlled. (A plain multiply would leave the
//! bucket bits blind to whatever sits in the key's high bits.)

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct VoxHasher(u64);

impl Hasher for VoxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(self.0.rotate_left(8) ^ b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        let mut k = v;
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^= k >> 33;
        self.0 = k;
    }
}

pub type VoxBuild = BuildHasherDefault<VoxHasher>;
pub type VoxMap<V> = HashMap<u64, V, VoxBuild>;
pub type VoxSet = HashSet<u64, VoxBuild>;
