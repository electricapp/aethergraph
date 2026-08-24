//! Portable Philox4x32-10 counter-based random-number generator.
//!
//! The GPU sampler uses this exact counter layout: `(node_lo, node_hi, layer,
//! draw_group)` with `(seed_lo, seed_hi)` as the key.  The constants are from
//! Random123: `M0=0xD2511F53`, `M1=0xCD9E8D57`, `W0=0x9E3779B9`, `W1=0xBB67AE85`.

/// Philox4x32-10 with draw_group 0 (legacy sampler keying).
#[must_use]
pub fn philox4x32_10(seed: u64, layer: u32, node: u64) -> [u32; 4] {
    philox4x32_10_draw(seed, layer, node, 0)
}

/// Philox4x32-10 with an explicit draw-group counter lane.
#[must_use]
pub fn philox4x32_10_draw(seed: u64, layer: u32, node: u64, draw_group: u32) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    let mut counter = [node as u32, (node >> 32) as u32, layer, draw_group];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for round in 0..10 {
        let hi0 = ((counter[0] as u64 * M0 as u64) >> 32) as u32;
        let lo0 = counter[0].wrapping_mul(M0);
        let hi1 = ((counter[2] as u64 * M1 as u64) >> 32) as u32;
        let lo1 = counter[2].wrapping_mul(M1);
        counter = [
            hi1 ^ counter[1] ^ key[0],
            lo1,
            hi0 ^ counter[3] ^ key[1],
            lo0,
        ];
        if round != 9 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
    }
    counter
}

/// One u32 from Philox, matching the GPU `aether_philox_u32` helper.
#[must_use]
pub fn philox_u32(seed: u64, layer: u32, node: u64, draw_index: u32) -> u32 {
    let r = philox4x32_10_draw(seed, layer, node, draw_index >> 2);
    match draw_index & 3 {
        1 => r[1],
        2 => r[2],
        3 => r[3],
        _ => r[0],
    }
}

/// Algorithm R reservoir without replacement — CPU oracle for K5.2 (`fanout <= 32`).
///
/// Returns up to `fanout` neighbor ids. If `neighbors.len() <= fanout`, returns
/// a clone of the full neighborhood.
#[must_use]
pub fn reservoir_sample(
    neighbors: &[u32],
    fanout: usize,
    seed: u64,
    layer: u32,
    node: u64,
) -> Vec<u32> {
    let degree = neighbors.len();
    if degree == 0 || fanout == 0 {
        return Vec::new();
    }
    if fanout >= degree {
        return neighbors.to_vec();
    }
    let mut reservoir: Vec<u32> = neighbors[..fanout].to_vec();
    for (t, &candidate) in neighbors.iter().enumerate().skip(fanout) {
        let r = philox_u32(seed, layer, node, t as u32);
        let j = (r as usize) % (t + 1);
        if j < fanout {
            reservoir[j] = candidate;
        }
    }
    reservoir
}

#[cfg(test)]
mod tests {
    use super::{philox_u32, philox4x32_10, reservoir_sample};

    #[test]
    fn random123_known_zero_vector() {
        assert_eq!(
            philox4x32_10(0, 0, 0),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn counter_dimensions_are_independent_and_repeatable() {
        let sample = philox4x32_10(42, 3, 99);
        assert_eq!(sample, philox4x32_10(42, 3, 99));
        assert_ne!(sample, philox4x32_10(42, 4, 99));
        assert_ne!(sample, philox4x32_10(43, 3, 99));
        assert_ne!(sample, philox4x32_10(42, 3, 100));
    }

    #[test]
    fn reservoir_is_deterministic_and_sized() {
        let neighbors: Vec<u32> = (0..20).collect();
        let a = reservoir_sample(&neighbors, 5, 7, 1, 3);
        let b = reservoir_sample(&neighbors, 5, 7, 1, 3);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        for &n in &a {
            assert!(neighbors.contains(&n));
        }
    }

    #[test]
    fn philox_u32_lanes_match_draw_packing() {
        let full = super::philox4x32_10_draw(1, 2, 3, 0);
        assert_eq!(philox_u32(1, 2, 3, 0), full[0]);
        assert_eq!(philox_u32(1, 2, 3, 1), full[1]);
    }
}
