//! Bulk numeric conversions with runtime SIMD dispatch.
//!
//! Wheels are built for baseline x86-64, which lacks F16C — so a
//! compile-time-gated path would never run for most users. Detection
//! happens at runtime instead (std caches the CPUID result), and the
//! scalar fallback stays fully portable.

/// Convert little-endian f16 bytes into f32 values.
///
/// `src.len()` must be exactly `2 * dst.len()`.
///
/// On x86-64 with F16C (any CPU from ~2013 on), eight elements convert per
/// `vcvtph2ps` — order-of-magnitude faster than the per-element software
/// conversion the scalar path performs.
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
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx")
        {
            // SAFETY: the required target features were verified at runtime
            // immediately above; slice lengths were asserted at entry.
            unsafe { f16_le_to_f32_f16c(src, dst) };
            return;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_all_lane_positions_and_tail() {
        // Cross the 8-lane SIMD boundary so both the vector body and the
        // scalar tail are exercised, whichever path dispatch picks.
        for len in [0usize, 1, 7, 8, 9, 16, 27] {
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
}
