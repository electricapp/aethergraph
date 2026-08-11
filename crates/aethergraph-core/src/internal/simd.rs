//! Bulk numeric conversions with runtime SIMD dispatch.
//!
//! Wheels are built for baseline x86-64, which lacks F16C — so a
//! compile-time-gated path would never run for most users. Detection
//! happens at runtime instead (std caches the CPUID result), and the
//! scalar fallback stays fully portable. x86-64 dispatch prefers the
//! 16-wide AVX-512 convert, then the 8-wide F16C convert, then scalar.
//!
//! aarch64 needs no detection: the f16→f32 vector convert (`fcvtl`) is
//! baseline NEON, present on every aarch64 CPU, so that path dispatches
//! unconditionally.
//!
//! bf16→f32 needs no special ISA: bfloat16 is the top 16 bits of an f32,
//! so widening is a shift. A portable scalar path and an AVX2 8-wide path
//! ([`bf16_le_to_f32`]) cover the on-CPU feature-transform groundwork.

/// Convert little-endian f16 bytes into f32 values.
///
/// `src.len()` must be exactly `2 * dst.len()`.
///
/// On x86-64 with F16C (any CPU from ~2013 on), eight elements convert per
/// `vcvtph2ps` — order-of-magnitude faster than the per-element software
/// conversion the scalar path performs. On aarch64, `fcvtl`/`fcvtl2`
/// convert eight elements per iteration unconditionally.
pub fn f16_le_to_f32(src: &[u8], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len() * 2,
        "f16 source byte length {} != 2 x destination length {}",
        src.len(),
        dst.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: `avx512f` verified at runtime; lengths asserted above.
            unsafe { f16_le_to_f32_avx512(src, dst) };
            return;
        }
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx")
        {
            // SAFETY: the required target features were verified at runtime
            // immediately above; slice lengths were asserted at entry.
            unsafe { f16_le_to_f32_f16c(src, dst) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon` is baseline for every aarch64 target, so the
        // feature requirement holds statically; slice lengths were
        // asserted at entry.
        unsafe { f16_le_to_f32_neon(src, dst) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    f16_le_to_f32_scalar(src, dst);
}

#[inline]
fn f16_le_to_f32_scalar(src: &[u8], dst: &mut [f32]) {
    for (out, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
        *out = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
    }
}

/// # Safety
/// Caller must verify the `f16c` and `avx` target features are available,
/// and that `src.len() == 2 * dst.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn f16_le_to_f32_f16c(src: &[u8], dst: &mut [f32]) {
    use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};

    let n = dst.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `src.len() == 2 * n` keep byte offset
        // `2 * i` plus 16 bytes in range.
        let src_p = unsafe { src.as_ptr().add(i * 2) } as *const __m128i;
        // SAFETY: `src_p` points at 16 in-range bytes; the unaligned load
        // variant tolerates any alignment.
        let h = unsafe { _mm_loadu_si128(src_p) };
        // Safe under target-feature 1.1: the enclosing fn statically
        // enables f16c/avx and the intrinsic takes no pointers.
        let f = _mm256_cvtph_ps(h);
        // SAFETY: `i + 8 <= n` keeps offset `i` plus 8 floats in range.
        let dst_p = unsafe { dst.as_mut_ptr().add(i) };
        // SAFETY: `dst_p` points at 8 in-range floats; the unaligned store
        // variant tolerates any alignment.
        unsafe { _mm256_storeu_ps(dst_p, f) };
        i += 8;
    }
    f16_le_to_f32_scalar(&src[i * 2..], &mut dst[i..]);
}

/// # Safety
/// Caller must verify the `avx512f` target feature is available and that
/// `src.len() == 2 * dst.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn f16_le_to_f32_avx512(src: &[u8], dst: &mut [f32]) {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm512_cvtph_ps, _mm512_storeu_ps};

    let n = dst.len();
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n` and `src.len() == 2 * n` keep byte offset
        // `2 * i` plus 32 bytes in range.
        let src_p = unsafe { src.as_ptr().add(i * 2) } as *const __m256i;
        // SAFETY: `src_p` points at 32 in-range bytes; the unaligned load
        // variant tolerates any alignment.
        let h = unsafe { _mm256_loadu_si256(src_p) };
        // Safe under target-feature 1.1: the enclosing fn statically
        // enables avx512f and the intrinsic takes no pointers.
        let f = _mm512_cvtph_ps(h);
        // SAFETY: `i + 16 <= n` keeps offset `i` plus 16 floats in range.
        let dst_p = unsafe { dst.as_mut_ptr().add(i) };
        // SAFETY: `dst_p` points at 16 in-range floats; the unaligned
        // store variant tolerates any alignment.
        unsafe { _mm512_storeu_ps(dst_p, f) };
        i += 16;
    }
    // Tail (< 16 elements) via the scalar path.
    f16_le_to_f32_scalar(&src[i * 2..], &mut dst[i..]);
}

/// # Safety
/// Caller must guarantee `src.len() == 2 * dst.len()`. The `neon` target
/// feature is baseline on aarch64, so the feature requirement is met by
/// construction.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn f16_le_to_f32_neon(src: &[u8], dst: &mut [f32]) {
    use std::arch::aarch64::{
        float32x4_t, uint16x8_t, vcvt_f32_f16, vcvt_high_f32_f16, vget_low_f16,
        vreinterpretq_f16_u16,
    };

    let n = dst.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `src.len() == 2 * n` keep byte offset
        // `2 * i` plus 16 bytes in range.
        let src_p = unsafe { src.as_ptr().add(i * 2) } as *const uint16x8_t;
        // SAFETY: `src_p` points at 16 in-range bytes; `read_unaligned`
        // tolerates any alignment.
        let raw = unsafe { src_p.read_unaligned() };
        // Safe under target-feature 1.1: the enclosing fn statically
        // enables neon and these intrinsics take no pointers.
        let h = vreinterpretq_f16_u16(raw);
        let lo = vcvt_f32_f16(vget_low_f16(h));
        let hi = vcvt_high_f32_f16(h);
        // SAFETY: `i + 8 <= n` keeps offset `i` plus 8 floats in range.
        let dst_p = unsafe { dst.as_mut_ptr().add(i) } as *mut float32x4_t;
        // SAFETY: `dst_p` points at the first 4 of 8 in-range floats;
        // `write_unaligned` tolerates any alignment.
        unsafe { dst_p.write_unaligned(lo) };
        // SAFETY: one `float32x4_t` past `dst_p` is still within the 8
        // in-range floats.
        let dst_hi = unsafe { dst_p.add(1) };
        // SAFETY: `dst_hi` points at the last 4 of 8 in-range floats.
        unsafe { dst_hi.write_unaligned(hi) };
        i += 8;
    }
    f16_le_to_f32_scalar(&src[i * 2..], &mut dst[i..]);
}

/// Convert little-endian bf16 (bfloat16) bytes into f32 values.
///
/// `src.len()` must be exactly `2 * dst.len()`. bf16 is the top 16 bits of
/// an IEEE f32, so the conversion is `(bits as u32) << 16` reinterpreted —
/// exact, no rounding. On x86-64 with AVX2, eight elements convert per
/// iteration; everywhere else the scalar path is used.
pub fn bf16_le_to_f32(src: &[u8], dst: &mut [f32]) {
    assert_eq!(
        src.len(),
        dst.len() * 2,
        "bf16 source byte length {} != 2 x destination length {}",
        src.len(),
        dst.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` verified at runtime; lengths asserted above.
            unsafe { bf16_le_to_f32_avx2(src, dst) };
            return;
        }
    }

    bf16_le_to_f32_scalar(src, dst);
}

#[inline]
fn bf16_le_to_f32_scalar(src: &[u8], dst: &mut [f32]) {
    for (out, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        *out = f32::from_bits((bits as u32) << 16);
    }
}

/// # Safety
/// Caller must verify the `avx2` target feature is available and that
/// `src.len() == 2 * dst.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_le_to_f32_avx2(src: &[u8], dst: &mut [f32]) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_cvtepu16_epi32, _mm256_slli_epi32,
        _mm256_storeu_si256,
    };

    let n = dst.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `src.len() == 2 * n` keep byte offset
        // `2 * i` plus 16 bytes in range.
        let src_p = unsafe { src.as_ptr().add(i * 2) } as *const __m128i;
        // SAFETY: `src_p` points at 16 in-range bytes; the unaligned load
        // variant tolerates any alignment.
        let raw = unsafe { _mm_loadu_si128(src_p) };
        // Widen eight u16 lanes to u32, then shift each into the high half
        // of an f32. Safe under target-feature 1.1: no pointers involved.
        let widened = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_slli_epi32::<16>(widened);
        // SAFETY: `i + 8 <= n` keeps offset `i` plus 8 floats in range.
        let dst_p = unsafe { dst.as_mut_ptr().add(i) } as *mut __m256i;
        // SAFETY: `dst_p` points at 8 in-range floats (32 bytes); the
        // unaligned store tolerates any alignment. The bit pattern is a
        // valid f32 (bf16 << 16 is exactly an f32 with a zeroed mantissa
        // tail).
        unsafe { _mm256_storeu_si256(dst_p, shifted) };
        i += 8;
    }
    bf16_le_to_f32_scalar(&src[i * 2..], &mut dst[i..]);
}

/// Gather `offsets[node+1] - offsets[node]` for each node in `nodes`.
///
/// This is the CSR degree lookup for a sampling frontier: the node IDs
/// are scattered, so each pair of reads is an independent random access
/// into a large array. Hardware gather (`vpgatherdq`) issues the whole
/// vector's worth of loads as one instruction, letting the memory system
/// overlap the cache misses instead of serializing them behind a scalar
/// loop's dependent address computations.
///
/// `nodes` must be in bounds for `offsets` (`node + 1 < offsets.len()`);
/// out-of-range nodes yield degree 0. Non-monotonic offsets — possible on
/// a file loaded with less than full validation — saturate to 0 rather
/// than wrapping.
pub fn gather_degrees(offsets: &[u64], nodes: &[u32], dst: &mut [u32]) {
    assert_eq!(
        nodes.len(),
        dst.len(),
        "node count {} != destination length {}",
        nodes.len(),
        dst.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` verified at runtime; lengths asserted above,
            // and the kernel bounds-checks every index it gathers.
            unsafe { gather_degrees_avx2(offsets, nodes, dst) };
            return;
        }
    }

    gather_degrees_scalar(offsets, nodes, dst);
}

/// Portable degree gather — the reference the vector paths must match.
fn gather_degrees_scalar(offsets: &[u64], nodes: &[u32], dst: &mut [u32]) {
    for (out, &node) in dst.iter_mut().zip(nodes) {
        let idx = node as usize;
        *out = match (offsets.get(idx), offsets.get(idx + 1)) {
            (Some(&start), Some(&end)) => end.saturating_sub(start) as u32,
            _ => 0,
        };
    }
}

/// AVX2 degree gather: four u64 offsets per `vpgatherdq`.
///
/// Each iteration gathers `offsets[node]` and `offsets[node+1]` for four
/// nodes as two gathers, subtracts, and narrows to u32. Nodes near the
/// end of the array fall to the scalar tail so the gather never reads
/// past `offsets`.
///
/// # Safety
/// The caller must have verified `avx2` support at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_degrees_avx2(offsets: &[u64], nodes: &[u32], dst: &mut [u32]) {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_i32gather_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    };

    // A gathered index must satisfy `idx + 1 < offsets.len()`, so any node
    // at or above this bound takes the scalar path.
    let Some(max_node) = offsets.len().checked_sub(1) else {
        dst.fill(0);
        return;
    };

    let base = offsets.as_ptr() as *const i64;
    let mut i = 0usize;
    while i + 4 <= nodes.len() {
        let chunk = &nodes[i..i + 4];
        // The vector path assumes every lane is in range; a chunk with any
        // out-of-range or offset-inverting node goes scalar instead of
        // needing a masked gather plus a saturating vector subtract.
        if chunk.iter().any(|&n| (n as usize) >= max_node) {
            gather_degrees_scalar(offsets, chunk, &mut dst[i..i + 4]);
            i += 4;
            continue;
        }

        // SAFETY: `chunk` is 4 u32s = 16 bytes; loadu has no alignment
        // requirement.
        let idx = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) };
        // SAFETY: every lane of `idx` is < max_node, so both `base[idx]`
        // and `base[idx + 1]` are within `offsets`. Scale 8 = size_of::<u64>().
        let starts = unsafe { _mm256_i32gather_epi64::<8>(base, idx) };
        // SAFETY: `max_node = offsets.len() - 1` and every lane is below
        // it, so `base + 1` is in bounds and one past it is at worst the
        // final element.
        let ends_base = unsafe { base.add(1) };
        // SAFETY: see above — every gathered address is within `offsets`.
        let ends = unsafe { _mm256_i32gather_epi64::<8>(ends_base, idx) };
        // `_mm256_sub_epi64` is safe under the enclosing target_feature.
        let diff = _mm256_sub_epi64(ends, starts);

        // Narrow four u64 degrees to four u32. Degrees fit u32 by the
        // format's edge-count bound, so the low half of each lane is the
        // whole value. `max(0)` reproduces the scalar path's saturating
        // subtract: non-monotonic offsets can slip past HeaderOnly
        // validation, and a wrapped difference would report a nonsense
        // multi-billion degree instead of 0.
        let mut lanes = [0i64; 4];
        // SAFETY: `lanes` is 4 i64s = 32 bytes, matching the register.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), diff) };
        dst[i..i + 4].copy_from_slice(&[
            lanes[0].max(0) as u32,
            lanes[1].max(0) as u32,
            lanes[2].max(0) as u32,
            lanes[3].max(0) as u32,
        ]);
        i += 4;
    }

    gather_degrees_scalar(offsets, &nodes[i..], &mut dst[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_all_lane_positions_and_tail() {
        // Cross the 8- and 16-lane SIMD boundaries so both the vector body
        // and the scalar tail are exercised, whichever path dispatch picks
        // (AVX-512 strides 16, F16C/NEON stride 8).
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 27, 32, 33] {
            let values: Vec<half::f16> = (0..len)
                .map(|i| half::f16::from_f32(i as f32 * 0.25 - 1.5))
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

            let mut out = vec![0.0f32; len];
            f16_le_to_f32(&bytes, &mut out);

            for (i, (&v, &o)) in values.iter().zip(out.iter()).enumerate() {
                assert_eq!(v.to_f32(), o, "lane {i} of {len}");
            }
        }
    }

    #[test]
    fn scalar_and_dispatch_agree() {
        let values: Vec<half::f16> = (0..1000)
            .map(|i| half::f16::from_f32((i as f32).sin() * 100.0))
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut dispatched = vec![0.0f32; values.len()];
        f16_le_to_f32(&bytes, &mut dispatched);

        let mut scalar = vec![0.0f32; values.len()];
        f16_le_to_f32_scalar(&bytes, &mut scalar);

        assert_eq!(dispatched, scalar);
    }

    #[test]
    fn bf16_converts_all_lane_positions_and_tail() {
        // Cross the 8-lane AVX2 boundary; bf16 is exact (a bit shift), so
        // dispatch must match the reference bit-for-bit.
        for len in [0usize, 1, 7, 8, 9, 16, 17, 31] {
            let values: Vec<half::bf16> = (0..len)
                .map(|i| half::bf16::from_f32(i as f32 * 0.5 - 2.0))
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

            let mut out = vec![0.0f32; len];
            bf16_le_to_f32(&bytes, &mut out);

            for (i, (&v, &o)) in values.iter().zip(out.iter()).enumerate() {
                assert_eq!(v.to_f32(), o, "bf16 lane {i} of {len}");
            }
        }
    }

    /// The dispatched gather must agree with the scalar reference on
    /// every input shape, including the ones that take the in-kernel
    /// scalar fallbacks (short tails, near-end nodes).
    #[test]
    fn gather_degrees_matches_scalar() {
        // Irregular degrees so no lane pattern is accidentally uniform.
        let num_nodes = 500usize;
        let mut offsets = vec![0u64];
        for n in 0..num_nodes {
            let degree = (n * 7 + 3) % 23;
            offsets.push(offsets[n] + degree as u64);
        }

        let cases: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
            (0..num_nodes as u32).collect(),
            (0..num_nodes as u32).rev().collect(),
            (0..num_nodes as u32).step_by(37).collect(),
            // Nodes at and past the end: the last valid node is
            // num_nodes-1, and anything beyond reports 0.
            vec![498, 499, 500, 501, 9999],
            vec![499; 9],
        ];

        for nodes in cases {
            let mut dispatched = vec![0u32; nodes.len()];
            gather_degrees(&offsets, &nodes, &mut dispatched);

            let mut scalar = vec![0u32; nodes.len()];
            gather_degrees_scalar(&offsets, &nodes, &mut scalar);

            assert_eq!(dispatched, scalar, "gather mismatch for {nodes:?}");

            // And both must match the definition.
            for (i, &n) in nodes.iter().enumerate() {
                let want = if (n as usize) < num_nodes {
                    (offsets[n as usize + 1] - offsets[n as usize]) as u32
                } else {
                    0
                };
                assert_eq!(dispatched[i], want, "node {n}");
            }
        }
    }

    #[test]
    fn gather_degrees_handles_degenerate_offsets() {
        // Empty and single-element offsets arrays have no valid node.
        for offsets in [vec![], vec![0u64]] {
            let nodes = [0u32, 1, 2, 3, 4];
            let mut out = vec![9u32; nodes.len()];
            gather_degrees(&offsets, &nodes, &mut out);
            assert_eq!(out, vec![0; nodes.len()], "offsets {offsets:?}");
        }

        // Non-monotonic offsets (possible under partial validation)
        // saturate rather than wrapping to a huge degree.
        let offsets = vec![0u64, 10, 5, 20, 25];
        let nodes = [0u32, 1, 2, 3];
        let mut out = vec![0u32; 4];
        gather_degrees(&offsets, &nodes, &mut out);
        assert_eq!(out, vec![10, 0, 15, 5]);
    }

    #[test]
    fn bf16_scalar_and_dispatch_agree() {
        let values: Vec<half::bf16> = (0..1000)
            .map(|i| half::bf16::from_f32((i as f32).cos() * 50.0))
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut dispatched = vec![0.0f32; values.len()];
        bf16_le_to_f32(&bytes, &mut dispatched);

        let mut scalar = vec![0.0f32; values.len()];
        bf16_le_to_f32_scalar(&bytes, &mut scalar);

        assert_eq!(dispatched, scalar);
    }
}
