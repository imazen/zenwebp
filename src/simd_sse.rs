//! SIMD-accelerated Sum of Squared Errors (SSE) computation
//!
//! This module provides optimized SSE computation for block distortion
//! measurement, similar to libwebp's SSE4x4_SSE2.

#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
use std::arch::x86_64::*;

/// Compute Sum of Squared Errors between two 4x4 blocks
///
/// Both blocks are stored in row-major order as 16 bytes.
#[cfg(feature = "unsafe-simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn sse4x4(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    // Scalar fallback for non-x86 or when no SIMD available
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse4x4_scalar(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        sse4x4_sse2(a, b)
    }
}

/// Scalar SSE computation
#[inline]
#[allow(dead_code)]
pub fn sse4x4_scalar(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let diff = i32::from(a[i]) - i32::from(b[i]);
        sum += (diff * diff) as u32;
    }
    sum
}

/// SSE2 implementation of 4x4 block SSE
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
unsafe fn sse4x4_sse2(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let zero = _mm_setzero_si128();

    // Load all 16 bytes at once
    let a_bytes = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let b_bytes = _mm_loadu_si128(b.as_ptr() as *const __m128i);

    // Unpack to 16-bit: low 8 bytes
    let a_lo = _mm_unpacklo_epi8(a_bytes, zero);
    let b_lo = _mm_unpacklo_epi8(b_bytes, zero);

    // Unpack to 16-bit: high 8 bytes
    let a_hi = _mm_unpackhi_epi8(a_bytes, zero);
    let b_hi = _mm_unpackhi_epi8(b_bytes, zero);

    // Subtract
    let d_lo = _mm_sub_epi16(a_lo, b_lo);
    let d_hi = _mm_sub_epi16(a_hi, b_hi);

    // Square and accumulate using madd (pairs of i16 multiplied and summed to i32)
    let sq_lo = _mm_madd_epi16(d_lo, d_lo);
    let sq_hi = _mm_madd_epi16(d_hi, d_hi);

    // Sum all 8 i32 values
    let sum = _mm_add_epi32(sq_lo, sq_hi);

    // Horizontal sum: sum[0]+sum[1]+sum[2]+sum[3]
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b10_11_00_01)); // swap pairs
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10)); // swap halves

    _mm_cvtsi128_si32(sum) as u32
}

/// Compute SSE between a source block and a reconstructed block (pred + residual)
///
/// This is used for RD scoring where we need SSE(src, pred + idct(quantized))
#[cfg(feature = "unsafe-simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn sse4x4_with_residual(src: &[u8; 16], pred: &[u8; 16], residual: &[i32; 16]) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse4x4_with_residual_scalar(src, pred, residual)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        sse4x4_with_residual_sse2(src, pred, residual)
    }
}

/// Scalar implementation of SSE with residual
#[inline]
#[allow(dead_code)]
pub fn sse4x4_with_residual_scalar(src: &[u8; 16], pred: &[u8; 16], residual: &[i32; 16]) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let reconstructed = (i32::from(pred[i]) + residual[i]).clamp(0, 255);
        let diff = i32::from(src[i]) - reconstructed;
        sum += (diff * diff) as u32;
    }
    sum
}

/// SSE2 implementation of SSE with residual
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
unsafe fn sse4x4_with_residual_sse2(src: &[u8; 16], pred: &[u8; 16], residual: &[i32; 16]) -> u32 {
    let zero = _mm_setzero_si128();
    let max_255 = _mm_set1_epi16(255);

    // Load source and prediction
    let src_bytes = _mm_loadu_si128(src.as_ptr() as *const __m128i);
    let pred_bytes = _mm_loadu_si128(pred.as_ptr() as *const __m128i);

    // Unpack to 16-bit
    let src_lo = _mm_unpacklo_epi8(src_bytes, zero);
    let src_hi = _mm_unpackhi_epi8(src_bytes, zero);
    let pred_lo = _mm_unpacklo_epi8(pred_bytes, zero);
    let pred_hi = _mm_unpackhi_epi8(pred_bytes, zero);

    // Load residuals (4 i32 values at a time) and pack to i16
    let res0 = _mm_loadu_si128(residual.as_ptr() as *const __m128i);
    let res1 = _mm_loadu_si128(residual.as_ptr().add(4) as *const __m128i);
    let res2 = _mm_loadu_si128(residual.as_ptr().add(8) as *const __m128i);
    let res3 = _mm_loadu_si128(residual.as_ptr().add(12) as *const __m128i);

    // Pack i32 to i16 (saturating)
    let res_lo = _mm_packs_epi32(res0, res1);
    let res_hi = _mm_packs_epi32(res2, res3);

    // Reconstruct: pred + residual, clamped to [0, 255]
    let rec_lo = _mm_add_epi16(pred_lo, res_lo);
    let rec_hi = _mm_add_epi16(pred_hi, res_hi);

    // Clamp to [0, 255]
    let rec_lo = _mm_max_epi16(rec_lo, zero);
    let rec_lo = _mm_min_epi16(rec_lo, max_255);
    let rec_hi = _mm_max_epi16(rec_hi, zero);
    let rec_hi = _mm_min_epi16(rec_hi, max_255);

    // Compute difference
    let d_lo = _mm_sub_epi16(src_lo, rec_lo);
    let d_hi = _mm_sub_epi16(src_hi, rec_hi);

    // Square and accumulate
    let sq_lo = _mm_madd_epi16(d_lo, d_lo);
    let sq_hi = _mm_madd_epi16(d_hi, d_hi);

    // Sum all values
    let sum = _mm_add_epi32(sq_lo, sq_hi);
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));

    _mm_cvtsi128_si32(sum) as u32
}

use crate::vp8_prediction::{CHROMA_BLOCK_SIZE, CHROMA_STRIDE, LUMA_BLOCK_SIZE, LUMA_STRIDE};

/// Compute SSE for a 16x16 luma block between source and bordered prediction buffer
///
/// Source is in a contiguous row-major array with `src_width` stride.
/// Prediction is in a bordered buffer with LUMA_STRIDE stride and 1-pixel border.
#[cfg(feature = "unsafe-simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse_16x16_luma_scalar(src_y, src_width, mbx, mby, pred)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        sse_16x16_luma_sse2(src_y, src_width, mbx, mby, pred)
    }
}

/// Scalar implementation for 16x16 luma SSE
#[inline]
#[allow(dead_code)]
pub fn sse_16x16_luma_scalar(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 16 * src_width + mbx * 16;

    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;

        for x in 0..16 {
            let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// SSE2 implementation of 16x16 luma SSE
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
unsafe fn sse_16x16_luma_sse2(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let zero = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();

    let src_base = mby * 16 * src_width + mbx * 16;

    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;

        // Load 16 bytes from source and prediction
        let src_bytes = _mm_loadu_si128(src_y.as_ptr().add(src_row) as *const __m128i);
        let pred_bytes = _mm_loadu_si128(pred.as_ptr().add(pred_row) as *const __m128i);

        // Unpack to 16-bit
        let src_lo = _mm_unpacklo_epi8(src_bytes, zero);
        let src_hi = _mm_unpackhi_epi8(src_bytes, zero);
        let pred_lo = _mm_unpacklo_epi8(pred_bytes, zero);
        let pred_hi = _mm_unpackhi_epi8(pred_bytes, zero);

        // Subtract
        let d_lo = _mm_sub_epi16(src_lo, pred_lo);
        let d_hi = _mm_sub_epi16(src_hi, pred_hi);

        // Square and accumulate
        let sq_lo = _mm_madd_epi16(d_lo, d_lo);
        let sq_hi = _mm_madd_epi16(d_hi, d_hi);

        total = _mm_add_epi32(total, sq_lo);
        total = _mm_add_epi32(total, sq_hi);
    }

    // Horizontal sum
    let sum = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
    _mm_cvtsi128_si32(sum) as u32
}

/// Compute SSE for an 8x8 chroma block between source and bordered prediction buffer
#[cfg(feature = "unsafe-simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse_8x8_chroma_scalar(src_uv, src_width, mbx, mby, pred)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        sse_8x8_chroma_sse2(src_uv, src_width, mbx, mby, pred)
    }
}

/// Scalar implementation for 8x8 chroma SSE
#[inline]
#[allow(dead_code)]
pub fn sse_8x8_chroma_scalar(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 8 * src_width + mbx * 8;

    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;

        for x in 0..8 {
            let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// SSE2 implementation of 8x8 chroma SSE
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
#[target_feature(enable = "sse2")]
#[allow(dead_code)]
unsafe fn sse_8x8_chroma_sse2(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let zero = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();

    let src_base = mby * 8 * src_width + mbx * 8;

    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;

        // Load 8 bytes from source and prediction (lower half of xmm register)
        let src_bytes = _mm_loadl_epi64(src_uv.as_ptr().add(src_row) as *const __m128i);
        let pred_bytes = _mm_loadl_epi64(pred.as_ptr().add(pred_row) as *const __m128i);

        // Unpack to 16-bit (only low 8 bytes are valid)
        let src_16 = _mm_unpacklo_epi8(src_bytes, zero);
        let pred_16 = _mm_unpacklo_epi8(pred_bytes, zero);

        // Subtract
        let diff = _mm_sub_epi16(src_16, pred_16);

        // Square and accumulate
        let sq = _mm_madd_epi16(diff, diff);
        total = _mm_add_epi32(total, sq);
    }

    // Horizontal sum
    let sum = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
    _mm_cvtsi128_si32(sum) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse4x4_scalar() {
        let a = [10u8; 16];
        let b = [12u8; 16];
        // Each pixel differs by 2, so SSE = 16 * 4 = 64
        assert_eq!(sse4x4_scalar(&a, &b), 64);
    }

    #[test]
    fn test_sse4x4_identical() {
        let a = [100u8; 16];
        assert_eq!(sse4x4_scalar(&a, &a), 0);
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_sse4x4_simd_matches_scalar() {
        let a: [u8; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let b: [u8; 16] = [
            12, 18, 33, 38, 55, 58, 73, 78, 93, 98, 113, 118, 133, 138, 153, 158,
        ];

        let scalar = sse4x4_scalar(&a, &b);
        let simd = sse4x4(&a, &b);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse4x4_with_residual_scalar() {
        let src = [100u8; 16];
        let pred = [90u8; 16];
        let residual = [10i32; 16]; // pred + residual = 100, so SSE = 0
        assert_eq!(sse4x4_with_residual_scalar(&src, &pred, &residual), 0);
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_sse4x4_with_residual_simd_matches_scalar() {
        let src: [u8; 16] = [
            100, 110, 120, 130, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 1,
        ];
        let pred: [u8; 16] = [
            95, 105, 115, 125, 85, 75, 65, 55, 45, 35, 25, 15, 5, 2, 1, 0,
        ];
        let residual: [i32; 16] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2, 1];

        let scalar = sse4x4_with_residual_scalar(&src, &pred, &residual);
        let simd = sse4x4_with_residual(&src, &pred, &residual);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse_16x16_luma_scalar() {
        // Create a 32x32 source buffer (2x2 macroblocks)
        let src_width = 32;
        let mut src_y = vec![100u8; 32 * 32];
        // Set macroblock (0,0) to values around 100
        for y in 0..16 {
            for x in 0..16 {
                src_y[y * src_width + x] = 100 + (x as u8);
            }
        }

        // Create prediction buffer with border
        let mut pred = [0u8; LUMA_BLOCK_SIZE];
        for y in 0..16 {
            for x in 0..16 {
                pred[(y + 1) * LUMA_STRIDE + 1 + x] = 102 + (x as u8); // diff of 2
            }
        }

        // Each pixel differs by 2, so SSE = 256 * 4 = 1024
        let sse = sse_16x16_luma_scalar(&src_y, src_width, 0, 0, &pred);
        assert_eq!(sse, 1024);
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_sse_16x16_luma_simd_matches_scalar() {
        let src_width = 32;
        let mut src_y = vec![0u8; 32 * 32];
        // Fill with varying values
        for y in 0..16 {
            for x in 0..16 {
                src_y[y * src_width + x] = ((y * 16 + x) % 256) as u8;
            }
        }

        let mut pred = [0u8; LUMA_BLOCK_SIZE];
        for y in 0..16 {
            for x in 0..16 {
                pred[(y + 1) * LUMA_STRIDE + 1 + x] = ((y * 16 + x + 5) % 256) as u8;
            }
        }

        let scalar = sse_16x16_luma_scalar(&src_y, src_width, 0, 0, &pred);
        let simd = sse_16x16_luma(&src_y, src_width, 0, 0, &pred);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse_8x8_chroma_scalar() {
        let src_width = 16;
        let mut src_uv = vec![128u8; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                src_uv[y * src_width + x] = 128 + (x as u8);
            }
        }

        let mut pred = [0u8; CHROMA_BLOCK_SIZE];
        for y in 0..8 {
            for x in 0..8 {
                pred[(y + 1) * CHROMA_STRIDE + 1 + x] = 130 + (x as u8); // diff of 2
            }
        }

        // 64 pixels * 4 = 256
        let sse = sse_8x8_chroma_scalar(&src_uv, src_width, 0, 0, &pred);
        assert_eq!(sse, 256);
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_sse_8x8_chroma_simd_matches_scalar() {
        let src_width = 16;
        let mut src_uv = vec![0u8; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                src_uv[y * src_width + x] = ((y * 8 + x * 3) % 256) as u8;
            }
        }

        let mut pred = [0u8; CHROMA_BLOCK_SIZE];
        for y in 0..8 {
            for x in 0..8 {
                pred[(y + 1) * CHROMA_STRIDE + 1 + x] = ((y * 8 + x * 3 + 7) % 256) as u8;
            }
        }

        let scalar = sse_8x8_chroma_scalar(&src_uv, src_width, 0, 0, &pred);
        let simd = sse_8x8_chroma(&src_uv, src_width, 0, 0, &pred);
        assert_eq!(scalar, simd);
    }
}
