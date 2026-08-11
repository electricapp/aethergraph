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
