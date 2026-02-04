//! Spectral distortion functions for VP8 encoding.
//!
//! This module contains Hadamard transform-based distortion functions (TDisto)
//! and flatness detection utilities. These are prime candidates for SIMD
//! optimization as they operate on 4x4/8x8/16x16 pixel blocks.
//!
//! ## Key functions
//!
//! - [`t_transform`]: Weighted 4x4 Hadamard transform (SIMD candidate)
//! - [`tdisto_4x4`], [`tdisto_16x16`]: Spectral distortion measurement
//! - [`is_flat_source_16`]: Flat block detection (SIMD candidate)
//!
//! ## Relationship to psy module
//!
//! The `t_transform` here computes a *weighted* Hadamard for distortion.
//! The [`super::super::psy::satd_4x4`] computes an *unweighted* Hadamard
//! for energy measurement. Both use the same transform kernel.

#![allow(dead_code)]

/// Hadamard transform for a 4x4 block, weighted by w[].
/// Returns the sum of |transformed_coeff| * weight.
///
/// This is a 4x4 Hadamard (Walsh-Hadamard) transform that measures
/// frequency-weighted energy in the block.
///
/// # Arguments
/// * `input` - 4x4 block of pixels (accessed with given stride)
/// * `stride` - Row stride of input buffer
/// * `w` - 16 weights for frequency weighting (CSF table)
#[inline]
pub fn t_transform(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        crate::common::simd_sse::t_transform(input, stride, w)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        t_transform_scalar(input, stride, w)
    }
}

/// Scalar implementation of t_transform.
///
/// This can be optimized with SIMD (AVX2/NEON) for significant speedup.
#[inline]
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
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

/// Compute spectral distortion between two 4x4 blocks.
///
/// Returns |TTransform(a) - TTransform(b)| >> 5
///
/// This measures the perceptual difference between blocks using
/// a frequency-weighted Hadamard transform.
///
/// # Arguments
/// * `a` - First 4x4 block (source)
/// * `b` - Second 4x4 block (prediction/reconstruction)
/// * `stride` - Row stride of both buffers
/// * `w` - 16 weights for frequency weighting (CSF table)
#[inline]
pub fn tdisto_4x4(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let sum1 = t_transform(a, stride, w);
    let sum2 = t_transform(b, stride, w);
    (sum2 - sum1).abs() >> 5
}

/// Compute spectral distortion between two 16x16 blocks.
///
/// Calls tdisto_4x4 16 times for each 4x4 sub-block.
///
/// # Arguments
/// * `a` - First 16x16 block (source)
/// * `b` - Second 16x16 block (prediction/reconstruction)
/// * `stride` - Row stride of both buffers
/// * `w` - 16 weights for frequency weighting (CSF table)
#[inline]
pub fn tdisto_16x16(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut d = 0i32;
    for y in 0..4 {
        for x in 0..4 {
            let offset = y * 4 * stride + x * 4;
            d += tdisto_4x4(&a[offset..], &b[offset..], stride, w);
        }
    }
    d
}

/// Compute spectral distortion between two 8x8 blocks (for chroma).
///
/// Calls tdisto_4x4 4 times for each 4x4 sub-block.
#[inline]
pub fn tdisto_8x8(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut d = 0i32;
    for y in 0..2 {
        for x in 0..2 {
            let offset = y * 4 * stride + x * 4;
            d += tdisto_4x4(&a[offset..], &b[offset..], stride, w);
        }
    }
    d
}

//------------------------------------------------------------------------------
// Flat source detection
//
// Detects if a source block is "flat" (uniform color).
// Used to force DC mode at image edges to avoid prediction artifacts.

/// Check if a 16x16 source block is flat (all pixels same value).
///
/// Returns true if the first pixel value repeats throughout the block.
/// This is used for edge macroblocks where prediction from unavailable
/// neighbors would create artifacts.
///
/// # Arguments
/// * `src` - Source pixels (16x16 block accessed with given stride)
/// * `stride` - Row stride of source buffer
#[inline]
pub fn is_flat_source_16(src: &[u8], stride: usize) -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        is_flat_source_16_dispatch(src, stride)
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        is_flat_source_16_scalar(src, stride)
    }
}

/// Scalar implementation of is_flat_source_16.
#[inline]
pub fn is_flat_source_16_scalar(src: &[u8], stride: usize) -> bool {
    let v = src[0];
    for y in 0..16 {
        let row = y * stride;
        for x in 0..16 {
            if src[row + x] != v {
                return false;
            }
        }
    }
    true
}

/// SIMD dispatch for is_flat_source_16.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn is_flat_source_16_dispatch(src: &[u8], stride: usize) -> bool {
    use archmage::{SimdToken, X64V3Token};
    if let Some(token) = X64V3Token::summon() {
        is_flat_source_16_sse2(token, src, stride)
    } else {
        is_flat_source_16_scalar(src, stride)
    }
}

/// SSE2 implementation: broadcast first pixel, compare 16 bytes per row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[archmage::arcane]
fn is_flat_source_16_sse2(
    _token: impl archmage::Has128BitSimd + Copy,
    src: &[u8],
    stride: usize,
) -> bool {
    use core::arch::x86_64::*;
    use safe_unaligned_simd::x86_64 as simd_mem;

    // Broadcast first pixel value to all 16 bytes
    let v = _mm_set1_epi8(src[0] as i8);

    for y in 0..16 {
        let row_start = y * stride;
        // Load 16 bytes from this row
        let row_arr = <&[u8; 16]>::try_from(&src[row_start..row_start + 16]).unwrap();
        let row_bytes = simd_mem::_mm_loadu_si128(row_arr);
        // Compare for equality: 0xFF where equal, 0x00 where different
        let cmp = _mm_cmpeq_epi8(row_bytes, v);
        // Extract comparison mask: 0xFFFF if all equal
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF {
            return false;
        }
    }
    true
}

/// Check if quantized AC coefficients indicate a flat block.
///
/// Returns true if the number of non-zero AC coefficients is below threshold.
/// Used to detect flat content that should prefer DC mode.
///
/// # Arguments
/// * `levels` - Quantized coefficient levels (16 coeffs per block)
/// * `num_blocks` - Number of 4x4 blocks to check
/// * `thresh` - Maximum allowed non-zero AC coefficients
#[inline]
pub fn is_flat_coeffs(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
    let mut score = 0i32;
    for block in 0..num_blocks {
        // Skip DC (index 0), check AC coefficients (indices 1-15)
        for i in 1..16 {
            if levels[block * 16 + i] != 0 {
                score += 1;
                if score > thresh {
                    return false;
                }
            }
        }
    }
    true
}

/// Flatness threshold for I16 mode (FLATNESS_LIMIT_I16 in libwebp)
/// libwebp uses 0, which means "always check for flatness" in I16 mode
pub const FLATNESS_LIMIT_I16: i32 = 0;

/// Flatness threshold for I4 mode (FLATNESS_LIMIT_I4 in libwebp)
/// libwebp uses 3 (not 2)
pub const FLATNESS_LIMIT_I4: i32 = 3;

/// Flatness threshold for UV mode (FLATNESS_LIMIT_UV in libwebp)
pub const FLATNESS_LIMIT_UV: i32 = 2;

/// Flatness penalty added to rate when UV coefficients are flat (libwebp)
/// This discourages selection of non-DC modes when content is flat.
pub const FLATNESS_PENALTY: u32 = 140;
