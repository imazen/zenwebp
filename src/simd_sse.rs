//! SIMD-accelerated Sum of Squared Errors (SSE) computation
//!
//! This module provides optimized SSE computation for block distortion
//! measurement, similar to libwebp's SSE4x4_SSE2.
//!
//! Uses archmage for safe SIMD intrinsics with token-based CPU feature verification.

#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
use archmage::{arcane, Has128BitSimd, Sse41Token, SimdToken};
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
use safe_unaligned_simd::x86_64 as simd_mem;
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
use core::arch::x86_64::*;

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
    {
        // SSE4.1 implies SSE2; summon() is now fast (no env var check)
        if let Some(token) = Sse41Token::summon() {
            sse4x4_sse2(token, a, b)
        } else {
            sse4x4_scalar(a, b)
        }
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
#[arcane]
#[allow(dead_code)]
fn sse4x4_sse2(_token: impl Has128BitSimd + Copy, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let zero = _mm_setzero_si128();

    // Load all 16 bytes at once
    let a_bytes = simd_mem::_mm_loadu_si128(a);
    let b_bytes = simd_mem::_mm_loadu_si128(b);

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
    {
        if let Some(token) = Sse41Token::summon() {
            sse4x4_with_residual_sse2(token, src, pred, residual)
        } else {
            sse4x4_with_residual_scalar(src, pred, residual)
        }
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
#[arcane]
#[allow(dead_code)]
fn sse4x4_with_residual_sse2(
    _token: impl Has128BitSimd + Copy,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    let zero = _mm_setzero_si128();
    let max_255 = _mm_set1_epi16(255);

    // Load source and prediction
    let src_bytes = simd_mem::_mm_loadu_si128(src);
    let pred_bytes = simd_mem::_mm_loadu_si128(pred);

    // Unpack to 16-bit
    let src_lo = _mm_unpacklo_epi8(src_bytes, zero);
    let src_hi = _mm_unpackhi_epi8(src_bytes, zero);
    let pred_lo = _mm_unpacklo_epi8(pred_bytes, zero);
    let pred_hi = _mm_unpackhi_epi8(pred_bytes, zero);

    // Load residuals (4 i32 values at a time) and pack to i16
    let res0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[0..4]).unwrap());
    let res1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[4..8]).unwrap());
    let res2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[8..12]).unwrap());
    let res3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[12..16]).unwrap());

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
    {
        if let Some(token) = Sse41Token::summon() {
            sse_16x16_luma_sse2(token, src_y, src_width, mbx, mby, pred)
        } else {
            sse_16x16_luma_scalar(src_y, src_width, mbx, mby, pred)
        }
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
#[arcane]
#[allow(dead_code)]
fn sse_16x16_luma_sse2(
    _token: impl Has128BitSimd + Copy,
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
        let src_bytes = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&src_y[src_row..][..16]).unwrap(),
        );
        let pred_bytes = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&pred[pred_row..][..16]).unwrap(),
        );

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
    {
        if let Some(token) = Sse41Token::summon() {
            sse_8x8_chroma_sse2(token, src_uv, src_width, mbx, mby, pred)
        } else {
            sse_8x8_chroma_scalar(src_uv, src_width, mbx, mby, pred)
        }
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
#[arcane]
#[allow(dead_code)]
fn sse_8x8_chroma_sse2(
    _token: impl Has128BitSimd + Copy,
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
        let src_bytes =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src_uv[src_row..][..8]).unwrap());
        let pred_bytes =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pred[pred_row..][..8]).unwrap());

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

//------------------------------------------------------------------------------
// TTransform - Spectral distortion helper for TDisto calculation
//
// Computes a Hadamard-like transform followed by weighted absolute value sum.
// This is used for perceptual distortion measurement.

/// Compute the TTransform for spectral distortion calculation (scalar version)
/// Returns weighted sum of absolute transform coefficients
#[inline]
#[allow(dead_code)]
pub fn t_transform_scalar(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut tmp = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let row = i * stride;
        let a0 = i32::from(input[row]) + i32::from(input[row + 2]);
        let a1 = i32::from(input[row + 1]) + i32::from(input[row + 3]);
        let a2 = i32::from(input[row + 1]) - i32::from(input[row + 3]);
        let a3 = i32::from(input[row]) - i32::from(input[row + 2]);
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }

    // Vertical pass with weighting
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;

        sum += i32::from(w[i]) * b0.abs();
        sum += i32::from(w[4 + i]) * b1.abs();
        sum += i32::from(w[8 + i]) * b2.abs();
        sum += i32::from(w[12 + i]) * b3.abs();
    }
    sum
}

/// SIMD-accelerated TTransform using SSE2
#[cfg(feature = "unsafe-simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn t_transform(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        t_transform_scalar(input, stride, w)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = Sse41Token::summon() {
            t_transform_sse2(token, input, stride, w)
        } else {
            t_transform_scalar(input, stride, w)
        }
    }
}

/// SSE2 implementation of TTransform
#[cfg(all(target_arch = "x86_64", feature = "unsafe-simd"))]
#[arcane]
#[allow(dead_code)]
fn t_transform_sse2(_token: impl Has128BitSimd + Copy, input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let zero = _mm_setzero_si128();

    // Load 4 rows of 4 bytes each, expand to i16
    // Row 0
    let row0 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[0..4]).unwrap());
    let row0_16 = _mm_unpacklo_epi8(row0, zero);

    // Row 1
    let row1 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride..][..4]).unwrap());
    let row1_16 = _mm_unpacklo_epi8(row1, zero);

    // Row 2
    let row2 =
        simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride * 2..][..4]).unwrap());
    let row2_16 = _mm_unpacklo_epi8(row2, zero);

    // Row 3
    let row3 =
        simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride * 3..][..4]).unwrap());
    let row3_16 = _mm_unpacklo_epi8(row3, zero);

    // Use intermediate array
    let mut tmp = [0i32; 16];

    // Extract rows to i32 for horizontal pass
    let r0 = [
        _mm_extract_epi16(row0_16, 0) as i32,
        _mm_extract_epi16(row0_16, 1) as i32,
        _mm_extract_epi16(row0_16, 2) as i32,
        _mm_extract_epi16(row0_16, 3) as i32,
    ];
    let r1 = [
        _mm_extract_epi16(row1_16, 0) as i32,
        _mm_extract_epi16(row1_16, 1) as i32,
        _mm_extract_epi16(row1_16, 2) as i32,
        _mm_extract_epi16(row1_16, 3) as i32,
    ];
    let r2 = [
        _mm_extract_epi16(row2_16, 0) as i32,
        _mm_extract_epi16(row2_16, 1) as i32,
        _mm_extract_epi16(row2_16, 2) as i32,
        _mm_extract_epi16(row2_16, 3) as i32,
    ];
    let r3 = [
        _mm_extract_epi16(row3_16, 0) as i32,
        _mm_extract_epi16(row3_16, 1) as i32,
        _mm_extract_epi16(row3_16, 2) as i32,
        _mm_extract_epi16(row3_16, 3) as i32,
    ];

    // Horizontal pass
    for (i, row) in [r0, r1, r2, r3].iter().enumerate() {
        let a0 = row[0] + row[2];
        let a1 = row[1] + row[3];
        let a2 = row[1] - row[3];
        let a3 = row[0] - row[2];
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }

    // Vertical pass with weighting
    let mut sum = 0i32;

    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;

        sum += i32::from(w[i]) * b0.abs();
        sum += i32::from(w[4 + i]) * b1.abs();
        sum += i32::from(w[8 + i]) * b2.abs();
        sum += i32::from(w[12 + i]) * b3.abs();
    }

    sum
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

    #[test]
    fn test_t_transform_scalar_basic() {
        // Create a simple 4x4 block with stride 16
        let mut input = [0u8; 64]; // 4 rows * 16 stride
                                   // Fill with a simple gradient
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = ((y * 4 + x) * 10) as u8;
            }
        }

        // Uniform weights
        let weights: [u16; 16] = [1; 16];

        let result = t_transform_scalar(&input, 16, &weights);
        // Just verify it produces a non-zero result for a non-uniform block
        assert!(result > 0);
    }

    #[test]
    fn test_t_transform_scalar_uniform() {
        // A uniform block should produce non-zero only at DC (index 0)
        let mut input = [128u8; 64];
        // Set actual 4x4 block to uniform values
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = 100;
            }
        }

        // Uniform weights
        let weights: [u16; 16] = [1; 16];

        let result = t_transform_scalar(&input, 16, &weights);
        // For uniform input, DC should be 4*100*4 = 1600, others 0
        // DC after both passes: sum of all = 400, weighted by 1 = 400
        assert!(result > 0);
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_t_transform_simd_matches_scalar() {
        // Create a varied 4x4 block
        let mut input = [0u8; 64];
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = ((y * 37 + x * 23 + 50) % 256) as u8;
            }
        }

        // Varied weights
        let weights: [u16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let scalar = t_transform_scalar(&input, 16, &weights);
        let simd = t_transform(&input, 16, &weights);
        assert_eq!(
            scalar, simd,
            "SIMD t_transform should match scalar: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    #[cfg(feature = "unsafe-simd")]
    fn test_t_transform_simd_matches_scalar_varied() {
        // Test with different strides and values
        for stride in [4, 8, 16, 32] {
            let mut input = vec![0u8; 4 * stride];
            for y in 0..4 {
                for x in 0..4 {
                    input[y * stride + x] = ((y * 53 + x * 41 + 17) % 256) as u8;
                }
            }

            let weights: [u16; 16] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1, 1];

            let scalar = t_transform_scalar(&input, stride, &weights);
            let simd = t_transform(&input, stride, &weights);
            assert_eq!(
                scalar, simd,
                "Mismatch at stride {}: scalar={}, simd={}",
                stride, scalar, simd
            );
        }
    }
}
