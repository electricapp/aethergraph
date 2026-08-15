//! Bulk numeric conversions with runtime SIMD dispatch.
//!
//! Wheels are built for baseline x86-64, which lacks F16C — so a
//! compile-time-gated path would never run for most users. Detection
//! happens at runtime instead (std caches the CPUID result), and the
//! scalar fallback stays fully portable.
//!
//! aarch64 needs no detection: the f16→f32 vector convert (`fcvtl`) is
//! baseline NEON, present on every aarch64 CPU, so that path dispatches
//! unconditionally.

/// An f16 to f32 converter with its SIMD dispatch already resolved.
///
/// On x86-64 with F16C (any CPU from ~2013 on), eight elements convert per
/// `vcvtph2ps` — order-of-magnitude faster than the per-element software
/// conversion the scalar path performs. On aarch64, `fcvtl`/`fcvtl2`
/// convert eight elements per iteration unconditionally.
///
/// Choosing between those paths on x86-64 is a runtime CPU query. Asking it
/// per row costs a cached lookup and a branch thousands of times per batch
/// and leaves the decision opaque inside the loop; resolving once at the
/// batch boundary reduces it to a register-held flag. On aarch64 the vector
/// path is baseline, so this carries no state and `convert` compiles to a
/// direct call.
///
/// Resolve one per batch — not per row — and pass it down.
#[derive(Clone, Copy, Debug)]
pub struct F16Decoder {
    #[cfg(target_arch = "x86_64")]
    f16c: bool,
}

impl F16Decoder {
    /// Perform the CPU feature query once.
    #[inline]
    pub fn resolve() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            f16c: std::arch::is_x86_feature_detected!("f16c")
                && std::arch::is_x86_feature_detected!("avx"),
        }
    }

    /// Convert little-endian f16 bytes into f32 values.
    ///
    /// `src.len()` must be exactly `2 * dst.len()`.
    #[inline(always)]
    pub fn convert(self, src: &[u8], dst: &mut [f32]) {
        assert_eq!(
            src.len(),
            dst.len() * 2,
            "f16 source byte length {} != 2 x destination length {}",
            src.len(),
            dst.len()
        );

        #[cfg(target_arch = "x86_64")]
        {
            if self.f16c {
                // SAFETY: `resolve` verified f16c and avx are present, and
                // that answer cannot change for the life of the process;
                // slice lengths were asserted at entry.
                unsafe { f16_le_to_f32_f16c(src, dst) };
            } else {
                f16_le_to_f32_scalar(src, dst);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `neon` is baseline for every aarch64 target, so the
            // feature requirement holds statically; slice lengths were
            // asserted at entry.
            unsafe { f16_le_to_f32_neon(src, dst) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        f16_le_to_f32_scalar(src, dst);
    }
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
            F16Decoder::resolve().convert(&bytes, &mut out);

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
        F16Decoder::resolve().convert(&bytes, &mut dispatched);

        let mut scalar = vec![0.0f32; values.len()];
        f16_le_to_f32_scalar(&bytes, &mut scalar);

        assert_eq!(dispatched, scalar);
    }
}
