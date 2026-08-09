//! Generation-stamped sampling scratch shared by the neighbor samplers.
//!
//! The homogeneous ([`crate::loader::NeighborSampler`]) and heterogeneous
//! ([`crate::loader::HeteroNeighborSampler`]) samplers reuse the same three
//! pieces of hot-path machinery, defined once here:
//!
//! - [`GenSlots`] / [`GenDedup`] — node dedup that resets across sampling
//!   passes with a generation-counter bump instead of an O(n) clear.
//! - [`FloydStamps`] — the fixed 256-entry stamp block backing Floyd's
//!   sampling without replacement for small degrees.
//! - [`WyRand`] — the sampling PRNG.
//!
//! Everything here is `#[inline(always)]`: these calls sit inside per-edge
//! loops and must compile to the same code as the operations written inline.

use crate::graph::NodeId;
use rustc_hash::FxHashMap;

/// How many frontier entries ahead the sampler hop loops prefetch the CSR
/// offset word (and, in the homogeneous sampler, the first edge line at half
/// this distance). The dedup probes and neighbor-list reads are random-access
/// and serially dependent without these hints.
pub(crate) const FRONTIER_PREFETCH_DIST: usize = 4;

/// Dense generation-stamped dedup table mapping global node IDs to local
/// indices for one sampling pass.
///
/// Each slot packs the generation tag in the high 32 bits and the local
/// index in the low 32, so a dedup probe touches exactly one cache line
/// instead of two parallel arrays. Call [`GenSlots::begin`] before the first
/// probe of each pass: it invalidates every prior entry with a counter bump,
/// paying the O(n) `fill` only when the u32 generation wraps.
pub(crate) struct GenSlots {
    slots: Vec<u64>,
    generation: u32,
}

impl GenSlots {
    /// Table covering global IDs `0..len`, with no pass begun yet.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            slots: vec![0u64; len],
            generation: 0,
        }
    }

    /// Pack a generation tag and local index into one dedup-slot word.
    #[inline(always)]
    const fn pack_slot(generation: u32, idx: u32) -> u64 {
        ((generation as u64) << 32) | idx as u64
    }

    /// Start a new sampling pass, invalidating all entries from prior passes.
    #[inline(always)]
    pub(crate) fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.slots.fill(0);
            self.generation = 1;
        }
    }

    /// Probe for `global`; on a miss, record it with local index
    /// `next_local`. Returns `(local index, inserted)`.
    #[inline(always)]
    pub(crate) fn probe_or_insert(&mut self, global: NodeId, next_local: u32) -> (u32, bool) {
        let i = global as usize;
        let slot = self.slots[i];
        if (slot >> 32) as u32 == self.generation {
            (slot as u32, false)
        } else {
            self.slots[i] = Self::pack_slot(self.generation, next_local);
            (next_local, true)
        }
    }
}

/// Two-mode node dedup: a dense [`GenSlots`] table when the ID space is
/// small enough (one random array load per probe), or an `FxHashMap` when a
/// dense table would waste memory and cache on a sparse sample (the map pays
/// hash + bucket walk, ~3-5x slower per probe). The caller picks the mode at
/// construction; both expose the same probe/reset surface.
pub(crate) enum GenDedup {
    Dense(GenSlots),
    Map(FxHashMap<NodeId, u32>),
}

impl GenDedup {
    /// Start a new sampling pass (dense: generation bump; map: clear,
    /// keeping capacity).
    #[inline(always)]
    pub(crate) fn begin(&mut self) {
        match self {
            Self::Dense(slots) => slots.begin(),
            Self::Map(map) => map.clear(),
        }
    }

    /// Probe for `global`; on a miss, record it with local index
    /// `next_local`. Returns `(local index, inserted)`.
    #[inline(always)]
    pub(crate) fn probe_or_insert(&mut self, global: NodeId, next_local: u32) -> (u32, bool) {
        match self {
            Self::Dense(slots) => slots.probe_or_insert(global, next_local),
            Self::Map(map) => match map.entry(global) {
                std::collections::hash_map::Entry::Occupied(e) => (*e.get(), false),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(next_local);
                    (next_local, true)
                }
            },
        }
    }
}

/// Generation-stamped membership block for Floyd's sampling without
/// replacement over degrees <= 256. Call [`FloydStamps::begin`] once per
/// sampled node: reuse is a counter bump, not an O(n) clear; the `fill` runs
/// only when the u32 generation wraps.
pub(crate) struct FloydStamps {
    stamp: [u32; 256],
    generation: u32,
}

impl FloydStamps {
    pub(crate) fn new() -> Self {
        Self {
            stamp: [0u32; 256],
            generation: 0,
        }
    }

    /// Start a new per-node sample, invalidating all stamps from prior ones.
    #[inline(always)]
    pub(crate) fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.fill(0);
            self.generation = 1;
        }
    }

    /// Mark index `i` as taken. Returns `true` if it was not yet taken in
    /// the current sample.
    #[inline(always)]
    pub(crate) fn test_and_set(&mut self, i: usize) -> bool {
        if self.stamp[i] != self.generation {
            self.stamp[i] = self.generation;
            true
        } else {
            false
        }
    }
}

impl Default for FloydStamps {
    fn default() -> Self {
        Self::new()
    }
}

/// Ultra-fast wyrand PRNG - simpler and faster than xoshiro for our use case.
/// Based on wyhash: https://github.com/wangyi-fudan/wyhash
#[derive(Clone)]
pub(crate) struct WyRand {
    state: u64,
}

impl WyRand {
    #[inline(always)]
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0xa0761d6478bd642f);
        let t = (self.state as u128).wrapping_mul((self.state ^ 0xe7037ed1a0b428db) as u128);
        ((t >> 64) ^ t) as u64
    }

    #[inline(always)]
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_slots_dedup_within_and_across_passes() {
        let mut slots = GenSlots::new(8);
        slots.begin();
        assert_eq!(slots.probe_or_insert(3, 0), (0, true));
        assert_eq!(slots.probe_or_insert(5, 1), (1, true));
        assert_eq!(slots.probe_or_insert(3, 2), (0, false));

        // A new pass forgets everything.
        slots.begin();
        assert_eq!(slots.probe_or_insert(3, 0), (0, true));
    }

    #[test]
    fn gen_dedup_map_mode_matches_dense_mode() {
        let mut dense = GenDedup::Dense(GenSlots::new(16));
        let mut map = GenDedup::Map(FxHashMap::default());
        for d in [&mut dense, &mut map] {
            d.begin();
            assert_eq!(d.probe_or_insert(7, 0), (0, true));
            assert_eq!(d.probe_or_insert(2, 1), (1, true));
            assert_eq!(d.probe_or_insert(7, 2), (0, false));
            d.begin();
            assert_eq!(d.probe_or_insert(2, 0), (0, true));
        }
    }

    #[test]
    fn floyd_stamps_reset_on_begin() {
        let mut stamps = FloydStamps::new();
        stamps.begin();
        assert!(stamps.test_and_set(10));
        assert!(!stamps.test_and_set(10));
        stamps.begin();
        assert!(stamps.test_and_set(10));
    }

    #[test]
    fn floyd_stamps_survive_generation_wrap() {
        let mut stamps = FloydStamps::new();
        stamps.generation = u32::MAX;
        stamps.stamp[4] = u32::MAX;
        stamps.begin(); // wraps: fill(0), generation = 1
        assert_eq!(stamps.generation, 1);
        assert!(stamps.test_and_set(4));
    }

    #[test]
    fn gen_slots_survive_generation_wrap() {
        let mut slots = GenSlots::new(4);
        slots.generation = u32::MAX;
        slots.slots[2] = GenSlots::pack_slot(u32::MAX, 9);
        slots.begin(); // wraps: fill(0), generation = 1
        assert_eq!(slots.generation, 1);
        assert_eq!(slots.probe_or_insert(2, 0), (0, true));
    }

    #[test]
    fn wyrand_is_deterministic_per_seed() {
        let mut a = WyRand::new(42);
        let mut b = WyRand::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_eq!(WyRand::new(7).next_u32(), WyRand::new(7).next_u64() as u32);
    }
}
