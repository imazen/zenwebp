//! SIMD implementations of VP8L inverse transforms.
//!
//! SSE2 implementations ported from libwebp's `src/dsp/lossless_sse2.c`, adapted
//! for our [R, G, B, A] byte layout (libwebp uses ARGB u32, which is [B, G, R, A]
//! in little-endian bytes).
//!
//! Portable implementations use magetypes for cross-platform SIMD (SSE2/NEON/WASM128).
//!
//! Uses archmage's `#[rite]` for inner functions (inlined into the single `#[arcane]`
//! entry point), eliminating per-call target_feature boundary overhead.

#![allow(clippy::too_many_arguments)]

use archmage::prelude::*;
use core::ops::Range;
use magetypes::simd::generic::{u8x16, u8x32};

#[cfg(target_arch = "x86")]
use archmage::intrinsics::x86 as simd_mem;
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem;

/// Helper: get a mutable reference to a 16-byte array from a slice.
#[inline(always)]
fn chunk16(data: &mut [u8], offset: usize) -> &mut [u8; 16] {
    data[offset..].first_chunk_mut::<16>().unwrap()
}

/// Helper: get an immutable reference to a 16-byte array from a slice.
#[inline(always)]
fn chunk16_ref(data: &[u8], offset: usize) -> &[u8; 16] {
    data[offset..].first_chunk::<16>().unwrap()
}

// =============================================================================
// Subtract-green inverse transform (AddGreenToBlueAndRed)
// =============================================================================

/// SSE2 subtract-green inverse: adds green channel to red and blue.
///
/// Our pixel layout: [R, G, B, A] at byte offsets [0, 1, 2, 3].
/// For each pixel: R += G, B += G (wrapping byte addition).
///
/// Entry point with #[arcane] — called from scalar dispatch code.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn add_green_to_blue_and_red_sse2_entry(_token: X64V1Token, image_data: &mut [u8]) {
    add_green_to_blue_and_red_sse2(_token, image_data);
}

/// Inner SSE2 subtract-green inverse.
///
/// 4 pixels in 16 bytes: R0 G0 B0 A0 | R1 G1 B1 A1 | R2 G2 B2 A2 | R3 G3 B3 A3
///
/// As u16 words (little-endian): G0:R0, A0:B0, G1:R1, A1:B1, ...
///
/// Strategy matching libwebp's AddGreenToBlueAndRed_SSE2:
/// 1. Shift right 8 in u16: get green (and alpha) in low bytes
/// 2. Shuffle to replicate green to both 16-bit halves of each pixel
/// 3. Add as bytes: R += G, G += 0, B += G, A += 0
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_green_to_blue_and_red_sse2(_token: X64V1Token, image_data: &mut [u8]) {
    let len = image_data.len();
    let simd_len = len & !15;

    let mut i = 0;
    while i < simd_len {
        let chunk = chunk16(image_data, i);
        let inp = simd_mem::_mm_loadu_si128(chunk);

        // Shift right 8 in u16: [0:G0, 0:A0, 0:G1, 0:A1, ...]
        let shifted = _mm_srli_epi16(inp, 8);

        // Replicate green to both halves: [0:G0, 0:G0, 0:G1, 0:G1, 0:G2, 0:G2, 0:G3, 0:G3]
        let b = _mm_shufflelo_epi16(shifted, 0xA0); // _MM_SHUFFLE(2,2,0,0)
        let c = _mm_shufflehi_epi16(b, 0xA0);

        // R += G, G += 0, B += G, A += 0
        let out = _mm_add_epi8(inp, c);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), out);

        i += 16;
    }

    // Scalar fallback for remaining pixels
    for pixel in image_data[simd_len..].chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_add(pixel[1]);
        pixel[2] = pixel[2].wrapping_add(pixel[1]);
    }
}

// =============================================================================
// Color transform inverse (TransformColorInverse)
// =============================================================================

/// SSE2 color transform inverse — entry point.
///
/// For each pixel, undoes the cross-color prediction:
///   R' = R + ColorTransformDelta(green_to_red, G)
///   B' = B + ColorTransformDelta(green_to_blue, G) + ColorTransformDelta(red_to_blue, R')
///
/// where ColorTransformDelta(t, c) = (t * c) >> 5  (signed fixed-point)
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn transform_color_inverse_sse2_entry(
    _token: X64V1Token,
    image_data: &mut [u8],
    width: usize,
    size_bits: u8,
    transform_data: &[u8],
) {
    let block_xsize = super::lossless_transform::block_xsize(width as u16, size_bits);

    for (y, row) in image_data.chunks_exact_mut(width * 4).enumerate() {
        let row_transform_data_start = (y >> size_bits) * block_xsize * 4;
        let row_tf_data = &transform_data[row_transform_data_start..];

        for (block, transform) in row
            .chunks_mut(4 << size_bits)
            .zip(row_tf_data.chunks_exact(4))
        {
            transform_color_inverse_block_sse2(
                _token,
                block,
                transform[2], // green_to_red
                transform[1], // green_to_blue
                transform[0], // red_to_blue
            );
        }
    }
}

/// SSE2 color transform inverse for a single block of pixels.
///
/// Our pixel layout: [R, G, B, A] at byte offsets [0, 1, 2, 3].
/// As u16 words (little-endian): word0 = G:R (G high, R low), word1 = A:B (A high, B low).
///
/// The _mm_mulhi_epi16 trick for ColorTransformDelta:
///   Given signed 8-bit multiplier `t` and signed 8-bit value `c`:
///     ColorTransformDelta(t, c) = (t * c) >> 5
///
///   If we have `c` in the HIGH byte of a u16 word (i.e., c << 8) and
///   pre-compute `cst = sign_extend(t) << 3`, then:
///     _mm_mulhi_epi16(c << 8, cst) = ((c << 8) * (t << 3)) >> 16 = (c * t) >> 5
///
///   The result lands in the LOW byte of each u16 word — exactly where R and B live.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transform_color_inverse_block_sse2(
    _token: X64V1Token,
    block: &mut [u8],
    green_to_red: u8,
    green_to_blue: u8,
    red_to_blue: u8,
) {
    // Pre-compute sign-extended, left-shifted-by-3 multipliers.
    let cst_g2r = ((green_to_red as i8 as i16) << 3) as u16;
    let cst_g2b = ((green_to_blue as i8 as i16) << 3) as u16;
    let cst_r2b = ((red_to_blue as i8 as i16) << 3) as u16;

    // mults_rb: per 32-bit lane:
    //   low word (word0, G:R position) = cst_g2r -> delta for R
    //   high word (word1, A:B position) = cst_g2b -> delta for B (green contribution)
    let mults_rb = _mm_set1_epi32(((cst_g2b as i32) << 16) | (cst_g2r as u32 as i32 & 0xFFFF));

    // mults_b2: per 32-bit lane:
    //   low word (word0, G:R position) = cst_r2b -> delta_B2 from R'
    //   high word = 0
    // After mulhi, delta_B2 lands in word0; we slli_epi32 by 16 to move to word1 (B position).
    let mults_b2 = _mm_set1_epi32(cst_r2b as u32 as i32);

    // Mask for green and alpha bytes: 0xFF00FF00 per 32-bit lane
    // Selects bytes at offsets 1,3 (G,A) in each pixel.
    let mask_ga = _mm_set1_epi32(0xFF00FF00u32 as i32);

    let len = block.len();
    let simd_len = len & !15;

    let mut i = 0;
    while i < simd_len {
        let inp = simd_mem::_mm_loadu_si128(chunk16_ref(block, i));

        // Extract G and A bytes (high bytes of each u16 word): [0:G0, 0:A0, 0:G1, 0:A1, ...]
        // Wait, mask_ga selects the high bytes, giving [G0:00, A0:00, G1:00, A1:00, ...].
        // That puts G in the HIGH byte of word0, A in the HIGH byte of word1.
        let a = _mm_and_si128(inp, mask_ga);

        // Replicate green to both halves of each pixel's 32-bit lane.
        // shufflelo(2,2,0,0): duplicate word0 and word2 in low 64 bits
        // shufflehi(2,2,0,0): duplicate word4 and word6 in high 64 bits
        // Result: [G0:00, G0:00, G1:00, G1:00, G2:00, G2:00, G3:00, G3:00]
        let b = _mm_shufflelo_epi16(a, 0xA0);
        let c = _mm_shufflehi_epi16(b, 0xA0);

        // mulhi_epi16: ((G << 8) * cst) >> 16 = (G * multiplier) >> 5
        // For word0 (G:R slot): delta_R = (G * g2r) >> 5
        // For word1 (A:B slot): delta_B1 = (G * g2b) >> 5
        let d = _mm_mulhi_epi16(c, mults_rb);

        // Add deltas to input bytes: R += delta_R, B += delta_B1
        // G and A also get junk added from high bytes of delta — fix below.
        let e = _mm_add_epi8(inp, d);

        // Now compute red-to-blue: delta_B2 = (R' * r2b) >> 5
        // R' is in byte 0 of each pixel = low byte of word0 in e.
        // Shift left 8 in u16: puts R' in high byte of word0.
        let f = _mm_slli_epi16(e, 8);

        // mulhi with mults_b2: delta_B2 = (R'<<8 * cst_r2b) >> 16 = (R' * r2b) >> 5
        // Result in word0 (G:R slot) of each 32-bit lane; word1 (A:B slot) = 0.
        let g = _mm_mulhi_epi16(f, mults_b2);

        // Move delta_B2 from word0 (G:R slot) to word1 (A:B slot) within each 32-bit lane.
        // slli_epi32 by 16 shifts the whole 32-bit lane left by 16 bits.
        // This moves the value from bytes 0-1 to bytes 2-3.
        let h = _mm_slli_epi32(g, 16);

        // Add delta_B2 to the partially corrected pixel.
        let corrected = _mm_add_epi8(e, h);

        // Reconstruct: keep R and B from corrected, G and A from original.
        let rb = _mm_andnot_si128(mask_ga, corrected);
        let ga = _mm_and_si128(inp, mask_ga);
        let out = _mm_or_si128(rb, ga);

        simd_mem::_mm_storeu_si128(chunk16(block, i), out);
        i += 16;
    }

    // Scalar fallback
    for pixel in block[simd_len..].chunks_exact_mut(4) {
        let green = pixel[1];
        let mut temp_red = u32::from(pixel[0]);
        let mut temp_blue = u32::from(pixel[2]);

        temp_red +=
            super::lossless_transform::color_transform_delta(green_to_red as i8, green as i8);
        temp_blue +=
            super::lossless_transform::color_transform_delta(green_to_blue as i8, green as i8);
        temp_blue +=
            super::lossless_transform::color_transform_delta(red_to_blue as i8, temp_red as i8);

        pixel[0] = (temp_red & 0xff) as u8;
        pixel[2] = (temp_blue & 0xff) as u8;
    }
}

// =============================================================================
// Predictor transforms with SSE2 batch operations
// =============================================================================

/// SSE2 predictor 1 (left): prefix-sum within 4 pixels.
///
/// Port of libwebp's PredictorAdd1_SSE2. Uses parallel prefix sum:
///   src = [a, b, c, d] (residuals)
///   sum = [a, a+b, a+b+c, a+b+c+d] (prefix sum)
///   result = sum + broadcast(prev_pixel)
///
/// The trick is that wrapping byte addition is associative, so the prefix
/// sum can be computed without serial dependency within the 4-pixel group.
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_1_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    _width: usize,
) {
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    // Load previous pixel (the left neighbor of the first pixel in range).
    // Broadcast to all 4 pixel positions.
    let prev_bytes = image_data[start - 4..].first_chunk::<4>().unwrap();
    let prev_pixel = _mm_cvtsi32_si128(i32::from_le_bytes(*prev_bytes));
    let mut prev = _mm_shuffle_epi32(prev_pixel, 0x00); // broadcast lane 0 to all 4 lanes

    let mut i = start;
    while i < simd_end {
        // Load 4 residual pixels
        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));

        // Parallel prefix sum (byte-wise wrapping add):
        // Step 1: shift left by 1 pixel (4 bytes) and add
        //   [a, b, c, d] + [0, a, b, c] = [a, a+b, b+c, c+d]
        let shift0 = _mm_slli_si128(src, 4);
        let sum0 = _mm_add_epi8(src, shift0);

        // Step 2: shift left by 2 pixels (8 bytes) and add
        //   [a, a+b, b+c, c+d] + [0, 0, a, a+b] = [a, a+b, a+b+c, a+b+c+d]
        let shift1 = _mm_slli_si128(sum0, 8);
        let sum1 = _mm_add_epi8(sum0, shift1);

        // Add previous pixel value (broadcast to all lanes)
        let res = _mm_add_epi8(sum1, prev);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);

        // Broadcast the last pixel (lane 3) for the next iteration
        prev = _mm_shuffle_epi32(res, 0xFF); // _MM_SHUFFLE(3,3,3,3)

        i += 16;
    }

    // Scalar remainder
    while i < end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - 4]);
        i += 1;
    }
}

/// SSE2 predictor 2 (top): out[i] += upper[i], 4 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_2_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    width: usize,
) {
    let stride = width * 4;
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    let mut i = start;
    while i < simd_end {
        let pred = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride));
        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));
        let res = _mm_add_epi8(src, pred);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);
        i += 16;
    }

    while i < end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - stride]);
        i += 1;
    }
}

/// SSE2 predictor 3 (top-right): out[i] += upper[i+1]
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_3_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    width: usize,
) {
    let stride = width * 4;
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    let mut i = start;
    while i < simd_end {
        let pred = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride + 4));
        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));
        let res = _mm_add_epi8(src, pred);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);
        i += 16;
    }

    while i < end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - stride + 4]);
        i += 1;
    }
}

/// SSE2 predictor 4 (top-left): out[i] += upper[i-1]
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_4_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    width: usize,
) {
    let stride = width * 4;
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    let mut i = start;
    while i < simd_end {
        let pred = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride - 4));
        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));
        let res = _mm_add_epi8(src, pred);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);
        i += 16;
    }

    while i < end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - stride - 4]);
        i += 1;
    }
}

/// SSE2 predictor 8 (average TL, T): avg(upper[i-1], upper[i])
///
/// Uses floor average: avg(a,b) = avg_epu8(a,b) - ((a^b)&1)
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_8_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    width: usize,
) {
    let stride = width * 4;
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    let ones = _mm_set1_epi8(1);
    let mut i = start;
    while i < simd_end {
        let t = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride));
        let tl = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride - 4));

        let avg1 = _mm_avg_epu8(t, tl);
        let one = _mm_and_si128(_mm_xor_si128(t, tl), ones);
        let avg = _mm_sub_epi8(avg1, one);

        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));
        let res = _mm_add_epi8(src, avg);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);

        i += 16;
    }

    while i < end {
        image_data[i] = image_data[i].wrapping_add(super::lossless_transform::average2(
            image_data[i - stride - 4],
            image_data[i - stride],
        ));
        i += 1;
    }
}

/// SSE2 predictor 9 (average T, TR): avg(upper[i], upper[i+1])
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_9_sse2(
    _token: X64V1Token,
    image_data: &mut [u8],
    range: core::ops::Range<usize>,
    width: usize,
) {
    let stride = width * 4;
    let start = range.start;
    let end = range.end;
    let simd_end = start + ((end - start) & !15);

    let ones = _mm_set1_epi8(1);
    let mut i = start;
    while i < simd_end {
        let t = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride));
        let tr = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i - stride + 4));

        let avg1 = _mm_avg_epu8(t, tr);
        let one = _mm_and_si128(_mm_xor_si128(t, tr), ones);
        let avg = _mm_sub_epi8(avg1, one);

        let src = simd_mem::_mm_loadu_si128(chunk16_ref(image_data, i));
        let res = _mm_add_epi8(src, avg);
        simd_mem::_mm_storeu_si128(chunk16(image_data, i), res);

        i += 16;
    }

    while i < end {
        image_data[i] = image_data[i].wrapping_add(super::lossless_transform::average2(
            image_data[i - stride],
            image_data[i - stride + 4],
        ));
        i += 1;
    }
}

// =============================================================================
// Dispatch: single #[arcane] entry for all predictor transforms
// =============================================================================

/// Dispatch a single predictor over a byte range, using SSE2 where available.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dispatch_predictor_sse2(
    _token: X64V1Token,
    predictor: u8,
    image_data: &mut [u8],
    start: usize,
    end: usize,
    width: usize,
) {
    let range = start..end;
    if range.is_empty() {
        return;
    }
    match predictor {
        0 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_0(image_data, range, width);
        }
        1 => apply_predictor_1_sse2(_token, image_data, range, width),
        2 => apply_predictor_2_sse2(_token, image_data, range, width),
        3 => apply_predictor_3_sse2(_token, image_data, range, width),
        4 => apply_predictor_4_sse2(_token, image_data, range, width),
        5 => super::lossless_transform::apply_predictor_transform_5(image_data, range, width),
        6 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_6(image_data, range, width);
        }
        7 => super::lossless_transform::apply_predictor_transform_7(image_data, range, width),
        8 => apply_predictor_8_sse2(_token, image_data, range, width),
        9 => apply_predictor_9_sse2(_token, image_data, range, width),
        10 => super::lossless_transform::apply_predictor_transform_10(image_data, range, width),
        11 => super::lossless_transform::apply_predictor_transform_11(image_data, range, width),
        12 => super::lossless_transform::apply_predictor_transform_12(image_data, range, width),
        13 => super::lossless_transform::apply_predictor_transform_13(image_data, range, width),
        _ => {}
    }
}

/// Apply the predictor transform using SSE2 where beneficial.
///
/// Predictors 2, 3, 4, 8, 9 have batch SSE2 implementations (no data dependency
/// on previous output pixel). The rest fall through to scalar.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn apply_predictor_transform_sse2_entry(
    _token: X64V1Token,
    image_data: &mut [u8],
    width: u16,
    height: u16,
    size_bits: u8,
    predictor_data: &[u8],
) {
    let block_xsize = super::lossless_transform::block_xsize(width, size_bits);
    let width = usize::from(width);
    let height = usize::from(height);

    let _ = super::lossless_transform::predictor_transform_borders(image_data, width, height);

    // Coalesce adjacent blocks with the same predictor mode into a single
    // range, reducing per-block dispatch overhead and giving SIMD loops
    // longer runs.
    for y in 1..height {
        let row_block_base = (y >> size_bits) * block_xsize;
        let mut run_start = 0usize;
        let mut run_end = 0usize;
        let mut run_pred = 255u8;

        for block_x in 0..block_xsize {
            let predictor = predictor_data[(row_block_base + block_x) * 4 + 1];
            let start_index = (y * width + (block_x << size_bits).max(1)) * 4;
            let end_index = (y * width + ((block_x + 1) << size_bits).min(width)) * 4;

            if predictor == run_pred && start_index == run_end {
                run_end = end_index;
            } else {
                if run_start < run_end {
                    dispatch_predictor_sse2(
                        _token, run_pred, image_data, run_start, run_end, width,
                    );
                }
                run_pred = predictor;
                run_start = start_index;
                run_end = end_index;
            }
        }
        if run_start < run_end {
            dispatch_predictor_sse2(_token, run_pred, image_data, run_start, run_end, width);
        }
    }
}

// =============================================================================
// Scalar fallbacks for color transforms (used by NEON/WASM stubs until SIMD is done)
// =============================================================================

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
fn color_inverse_scalar_fallback(
    image_data: &mut [u8],
    width: usize,
    size_bits: u8,
    transform_data: &[u8],
) {
    let block_xsize = usize::from(super::lossless::subsample_size(width as u16, size_bits));

    for (y, row) in image_data.chunks_exact_mut(width * 4).enumerate() {
        let row_transform_data_start = (y >> size_bits) * block_xsize * 4;
        let row_tf_data = &transform_data[row_transform_data_start..];

        for (block, transform) in row
            .chunks_mut(4 << size_bits)
            .zip(row_tf_data.chunks_exact(4))
        {
            let red_to_blue = transform[0];
            let green_to_blue = transform[1];
            let green_to_red = transform[2];

            for pixel in block.chunks_exact_mut(4) {
                let green = u32::from(pixel[1]);
                let mut temp_red = u32::from(pixel[0]);
                let mut temp_blue = u32::from(pixel[2]);

                temp_red += super::lossless_transform::color_transform_delta(
                    green_to_red as i8,
                    green as i8,
                );
                temp_blue += super::lossless_transform::color_transform_delta(
                    green_to_blue as i8,
                    green as i8,
                );
                temp_blue += super::lossless_transform::color_transform_delta(
                    red_to_blue as i8,
                    temp_red as i8,
                );

                pixel[0] = (temp_red & 0xff) as u8;
                pixel[2] = (temp_blue & 0xff) as u8;
            }
        }
    }
}

// =============================================================================
// Portable SIMD predictor implementations (magetypes — works on all archs)
// =============================================================================

// --- Portable add-green using magetypes ---

/// Portable add-green inverse: R += G, B += G for each RGBA pixel.
///
/// Strategy: isolate green bytes via AND mask, then use u16 shift + OR
/// to broadcast green to both the R (byte 0) and B (byte 2) positions
/// within each pixel, then add as bytes.
///
/// Layout: [R0,G0,B0,A0, R1,G1,B1,A1, ...] (4 pixels = 16 bytes)
/// Step 1: green_only = input & [0,0xFF,0,0, ...] → [0,G0,0,0, 0,G1,0,0, ...]
/// Step 2: As u16 LE: [G0<<8, 0, G1<<8, 0, ...]. Shift right 8 → [G0, 0, G1, 0, ...]
/// Step 3: As bytes: [G0,0,0,0, G1,0,0,0, ...]. Shift left 16 → [0,0,G0,0, 0,0,G1,0, ...]
/// Step 4: OR steps 2+3: [G0,0,G0,0, G1,0,G1,0, ...]. Add to input.
/// Portable add-green using scalar 4-pixel unrolling.
/// Simple and correct — the compiler autovectorizes this well on all architectures.
fn add_green_portable<T: magetypes::simd::backends::U8x16Backend>(
    _token: T,
    image_data: &mut [u8],
) {
    // Process 4 pixels (16 bytes) at a time for autovectorization
    let (chunks, remainder) = image_data.as_chunks_mut::<16>();
    for chunk in chunks {
        let g0 = chunk[1];
        let g1 = chunk[5];
        let g2 = chunk[9];
        let g3 = chunk[13];
        chunk[0] = chunk[0].wrapping_add(g0);
        chunk[2] = chunk[2].wrapping_add(g0);
        chunk[4] = chunk[4].wrapping_add(g1);
        chunk[6] = chunk[6].wrapping_add(g1);
        chunk[8] = chunk[8].wrapping_add(g2);
        chunk[10] = chunk[10].wrapping_add(g2);
        chunk[12] = chunk[12].wrapping_add(g3);
        chunk[14] = chunk[14].wrapping_add(g3);
    }
    for pixel in remainder.chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_add(pixel[1]);
        pixel[2] = pixel[2].wrapping_add(pixel[1]);
    }
}

// --- Portable predictor helpers using magetypes ---

/// Generic add-from-offset predictor body. Works with any token that supports u8x16.
pub(crate) fn predictor_add_body<T: magetypes::simd::backends::U8x16Backend>(
    token: T,
    image_data: &mut [u8],
    range: &Range<usize>,
    offset: usize,
) {
    let len = range.end - range.start;
    let simd_len = len & !15;
    let mut i = range.start;
    while i < range.start + simd_len {
        let cur = u8x16::load(token, chunk16_ref(image_data, i));
        let src = u8x16::load(token, chunk16_ref(image_data, i - offset));
        (cur + src).store(chunk16(image_data, i));
        i += 16;
    }
    while i < range.end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - offset]);
        i += 1;
    }
}

/// Generic floor-average predictor body: out[i] += floor_avg(a[i], b[i]).
pub(crate) fn predictor_avg_body<T: magetypes::simd::backends::U8x16Backend>(
    token: T,
    image_data: &mut [u8],
    range: &Range<usize>,
    offset_a: usize,
    offset_b: usize,
) {
    let len = range.end - range.start;
    let simd_len = len & !15;
    let mut i = range.start;
    while i < range.start + simd_len {
        let cur = u8x16::load(token, chunk16_ref(image_data, i));
        let a = u8x16::load(token, chunk16_ref(image_data, i - offset_a));
        let b = u8x16::load(token, chunk16_ref(image_data, i - offset_b));
        // Floor average: (a & b) + ((a ^ b) >> 1)
        let avg = (a & b) + u8x16::shr_logical_const::<1>(a ^ b);
        (cur + avg).store(chunk16(image_data, i));
        i += 16;
    }
    while i < range.end {
        let a_val = image_data[i - offset_a];
        let b_val = image_data[i - offset_b];
        let avg = (a_val & b_val) + ((a_val ^ b_val) >> 1);
        image_data[i] = image_data[i].wrapping_add(avg);
        i += 1;
    }
}

/// Helper: get a mutable reference to a 32-byte array from a slice.
#[inline(always)]
fn chunk32(data: &mut [u8], offset: usize) -> &mut [u8; 32] {
    data[offset..].first_chunk_mut::<32>().unwrap()
}

/// Helper: get an immutable reference to a 32-byte array from a slice.
#[inline(always)]
fn chunk32_ref(data: &[u8], offset: usize) -> &[u8; 32] {
    data[offset..].first_chunk::<32>().unwrap()
}

/// Wide (32-byte) add-from-offset predictor body using u8x32.
/// On AVX2 this uses native 256-bit ops; on NEON/WASM it decomposes to 2× 128-bit.
fn predictor_add_body_wide<
    T: magetypes::simd::backends::U8x32Backend + magetypes::simd::backends::U8x16Backend,
>(
    token: T,
    image_data: &mut [u8],
    range: &Range<usize>,
    offset: usize,
) {
    let len = range.end - range.start;
    let simd32_len = len & !31;
    let mut i = range.start;
    while i < range.start + simd32_len {
        let cur = u8x32::load(token, chunk32_ref(image_data, i));
        let src = u8x32::load(token, chunk32_ref(image_data, i - offset));
        (cur + src).store(chunk32(image_data, i));
        i += 32;
    }
    // 16-byte tail
    while i + 16 <= range.end {
        let cur = u8x16::load(token, chunk16_ref(image_data, i));
        let src = u8x16::load(token, chunk16_ref(image_data, i - offset));
        (cur + src).store(chunk16(image_data, i));
        i += 16;
    }
    while i < range.end {
        image_data[i] = image_data[i].wrapping_add(image_data[i - offset]);
        i += 1;
    }
}

/// Wide (32-byte) floor-average predictor body using u8x32.
fn predictor_avg_body_wide<
    T: magetypes::simd::backends::U8x32Backend + magetypes::simd::backends::U8x16Backend,
>(
    token: T,
    image_data: &mut [u8],
    range: &Range<usize>,
    offset_a: usize,
    offset_b: usize,
) {
    let len = range.end - range.start;
    let simd32_len = len & !31;
    let mut i = range.start;
    while i < range.start + simd32_len {
        let cur = u8x32::load(token, chunk32_ref(image_data, i));
        let a = u8x32::load(token, chunk32_ref(image_data, i - offset_a));
        let b = u8x32::load(token, chunk32_ref(image_data, i - offset_b));
        let avg = (a & b) + u8x32::shr_logical_const::<1>(a ^ b);
        (cur + avg).store(chunk32(image_data, i));
        i += 32;
    }
    // 16-byte tail
    while i + 16 <= range.end {
        let cur = u8x16::load(token, chunk16_ref(image_data, i));
        let a = u8x16::load(token, chunk16_ref(image_data, i - offset_a));
        let b = u8x16::load(token, chunk16_ref(image_data, i - offset_b));
        let avg = (a & b) + u8x16::shr_logical_const::<1>(a ^ b);
        (cur + avg).store(chunk16(image_data, i));
        i += 16;
    }
    while i < range.end {
        let a_val = image_data[i - offset_a];
        let b_val = image_data[i - offset_b];
        let avg = (a_val & b_val) + ((a_val ^ b_val) >> 1);
        image_data[i] = image_data[i].wrapping_add(avg);
        i += 1;
    }
}

// --- x86_64 V3 (AVX2) predictor wrappers using u8x32 ---

#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_2_v3(
    token: X64V3Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body_wide(token, image_data, &range, width * 4);
}
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_3_v3(
    token: X64V3Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body_wide(token, image_data, &range, width * 4 - 4);
}
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_4_v3(
    token: X64V3Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body_wide(token, image_data, &range, width * 4 + 4);
}
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_8_v3(
    token: X64V3Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body_wide(token, image_data, &range, width * 4 + 4, width * 4);
}
#[cfg(target_arch = "x86_64")]
#[rite]
fn apply_predictor_9_v3(
    token: X64V3Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body_wide(token, image_data, &range, width * 4, width * 4 - 4);
}

// --- x86_64 V3 dispatch ---

#[cfg(target_arch = "x86_64")]
#[rite]
fn dispatch_predictor_v3(
    _token: X64V3Token,
    predictor: u8,
    image_data: &mut [u8],
    start: usize,
    end: usize,
    width: usize,
) {
    let range = start..end;
    if range.is_empty() {
        return;
    }
    match predictor {
        0 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_0(image_data, range, width);
        }
        // Predictor 1 uses existing SSE2 prefix-sum (serial dependency limits AVX2 benefit)
        1 => apply_predictor_1_sse2(_token.v1(), image_data, range, width),
        2 => apply_predictor_2_v3(_token, image_data, range, width),
        3 => apply_predictor_3_v3(_token, image_data, range, width),
        4 => apply_predictor_4_v3(_token, image_data, range, width),
        5 => super::lossless_transform::apply_predictor_transform_5(image_data, range, width),
        6 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_6(image_data, range, width);
        }
        7 => super::lossless_transform::apply_predictor_transform_7(image_data, range, width),
        8 => apply_predictor_8_v3(_token, image_data, range, width),
        9 => apply_predictor_9_v3(_token, image_data, range, width),
        10 => super::lossless_transform::apply_predictor_transform_10(image_data, range, width),
        11 => super::lossless_transform::apply_predictor_transform_11(image_data, range, width),
        12 => super::lossless_transform::apply_predictor_transform_12(image_data, range, width),
        13 => super::lossless_transform::apply_predictor_transform_13(image_data, range, width),
        _ => {}
    }
}

/// AVX2 predictor transform entry point — 32-byte processing for predictors 2-4, 8-9.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn apply_predictor_transform_v3_entry(
    _token: X64V3Token,
    image_data: &mut [u8],
    width: u16,
    height: u16,
    size_bits: u8,
    predictor_data: &[u8],
) {
    let block_xsize = super::lossless_transform::block_xsize(width, size_bits);
    let width = usize::from(width);
    let height = usize::from(height);

    let _ = super::lossless_transform::predictor_transform_borders(image_data, width, height);

    for y in 1..height {
        let row_block_base = (y >> size_bits) * block_xsize;
        let mut run_start = 0usize;
        let mut run_end = 0usize;
        let mut run_pred = 255u8;

        for block_x in 0..block_xsize {
            let predictor = predictor_data[(row_block_base + block_x) * 4 + 1];
            let start_index = (y * width + (block_x << size_bits).max(1)) * 4;
            let end_index = (y * width + ((block_x + 1) << size_bits).min(width)) * 4;

            if predictor == run_pred && start_index == run_end {
                run_end = end_index;
            } else {
                if run_start < run_end {
                    dispatch_predictor_v3(_token, run_pred, image_data, run_start, run_end, width);
                }
                run_pred = predictor;
                run_start = start_index;
                run_end = end_index;
            }
        }
        if run_start < run_end {
            dispatch_predictor_v3(_token, run_pred, image_data, run_start, run_end, width);
        }
    }
}

// --- aarch64 NEON predictor wrappers ---

#[cfg(target_arch = "aarch64")]
#[rite]
fn apply_predictor_2_neon(
    token: NeonToken,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4);
}
#[cfg(target_arch = "aarch64")]
#[rite]
fn apply_predictor_3_neon(
    token: NeonToken,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4 - 4);
}
#[cfg(target_arch = "aarch64")]
#[rite]
fn apply_predictor_4_neon(
    token: NeonToken,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4 + 4);
}
#[cfg(target_arch = "aarch64")]
#[rite]
fn apply_predictor_8_neon(
    token: NeonToken,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body(token, image_data, &range, width * 4 + 4, width * 4);
}
#[cfg(target_arch = "aarch64")]
#[rite]
fn apply_predictor_9_neon(
    token: NeonToken,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body(token, image_data, &range, width * 4, width * 4 - 4);
}

// --- wasm32 SIMD128 predictor wrappers ---

#[cfg(target_arch = "wasm32")]
#[rite]
fn apply_predictor_2_wasm128(
    token: Wasm128Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4);
}
#[cfg(target_arch = "wasm32")]
#[rite]
fn apply_predictor_3_wasm128(
    token: Wasm128Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4 - 4);
}
#[cfg(target_arch = "wasm32")]
#[rite]
fn apply_predictor_4_wasm128(
    token: Wasm128Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_add_body(token, image_data, &range, width * 4 + 4);
}
#[cfg(target_arch = "wasm32")]
#[rite]
fn apply_predictor_8_wasm128(
    token: Wasm128Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body(token, image_data, &range, width * 4 + 4, width * 4);
}
#[cfg(target_arch = "wasm32")]
#[rite]
fn apply_predictor_9_wasm128(
    token: Wasm128Token,
    image_data: &mut [u8],
    range: Range<usize>,
    width: usize,
) {
    predictor_avg_body(token, image_data, &range, width * 4, width * 4 - 4);
}

// =============================================================================
// NEON dispatch and entry point
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[rite]
fn dispatch_predictor_neon(
    _token: NeonToken,
    predictor: u8,
    image_data: &mut [u8],
    start: usize,
    end: usize,
    width: usize,
) {
    let range = start..end;
    if range.is_empty() {
        return;
    }
    match predictor {
        0 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_0(image_data, range, width);
        }
        // Predictor 1 (prefix-sum) falls back to scalar — serial dependency
        1 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_1(image_data, range, width);
        }
        2 => apply_predictor_2_neon(_token, image_data, range, width),
        3 => apply_predictor_3_neon(_token, image_data, range, width),
        4 => apply_predictor_4_neon(_token, image_data, range, width),
        5 => super::lossless_transform::apply_predictor_transform_5(image_data, range, width),
        6 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_6(image_data, range, width);
        }
        7 => super::lossless_transform::apply_predictor_transform_7(image_data, range, width),
        8 => apply_predictor_8_neon(_token, image_data, range, width),
        9 => apply_predictor_9_neon(_token, image_data, range, width),
        10 => super::lossless_transform::apply_predictor_transform_10(image_data, range, width),
        11 => super::lossless_transform::apply_predictor_transform_11(image_data, range, width),
        12 => super::lossless_transform::apply_predictor_transform_12(image_data, range, width),
        13 => super::lossless_transform::apply_predictor_transform_13(image_data, range, width),
        _ => {}
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn apply_predictor_transform_neon_entry(
    _token: NeonToken,
    image_data: &mut [u8],
    width: u16,
    height: u16,
    size_bits: u8,
    predictor_data: &[u8],
) {
    let block_xsize = super::lossless_transform::block_xsize(width, size_bits);
    let width = usize::from(width);
    let height = usize::from(height);

    let _ = super::lossless_transform::predictor_transform_borders(image_data, width, height);

    for y in 1..height {
        let row_block_base = (y >> size_bits) * block_xsize;
        let mut run_start = 0usize;
        let mut run_end = 0usize;
        let mut run_pred = 255u8;

        for block_x in 0..block_xsize {
            let predictor = predictor_data[(row_block_base + block_x) * 4 + 1];
            let start_index = (y * width + (block_x << size_bits).max(1)) * 4;
            let end_index = (y * width + ((block_x + 1) << size_bits).min(width)) * 4;

            if predictor == run_pred && start_index == run_end {
                run_end = end_index;
            } else {
                if run_start < run_end {
                    dispatch_predictor_neon(
                        _token, run_pred, image_data, run_start, run_end, width,
                    );
                }
                run_pred = predictor;
                run_start = start_index;
                run_end = end_index;
            }
        }
        if run_start < run_end {
            dispatch_predictor_neon(_token, run_pred, image_data, run_start, run_end, width);
        }
    }
}

// Stub NEON entry points for color transforms (fall through to scalar for now)
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn add_green_to_blue_and_red_neon_entry(_token: NeonToken, image_data: &mut [u8]) {
    add_green_to_blue_and_red_neon(_token, image_data);
}

/// NEON add-green inverse using portable magetypes with u16x8 shift trick.
#[cfg(target_arch = "aarch64")]
#[rite]
fn add_green_to_blue_and_red_neon(_token: NeonToken, image_data: &mut [u8]) {
    add_green_portable(_token, image_data);
}

#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn transform_color_inverse_neon_entry(
    _token: NeonToken,
    image_data: &mut [u8],
    width: usize,
    size_bits: u8,
    transform_data: &[u8],
) {
    // TODO: NEON color_inverse SIMD implementation
    color_inverse_scalar_fallback(image_data, width, size_bits, transform_data);
}

// =============================================================================
// WASM128 dispatch and entry point
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[rite]
fn dispatch_predictor_wasm128(
    _token: Wasm128Token,
    predictor: u8,
    image_data: &mut [u8],
    start: usize,
    end: usize,
    width: usize,
) {
    let range = start..end;
    if range.is_empty() {
        return;
    }
    match predictor {
        0 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_0(image_data, range, width);
        }
        1 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_1(image_data, range, width);
        }
        2 => apply_predictor_2_wasm128(_token, image_data, range, width),
        3 => apply_predictor_3_wasm128(_token, image_data, range, width),
        4 => apply_predictor_4_wasm128(_token, image_data, range, width),
        5 => super::lossless_transform::apply_predictor_transform_5(image_data, range, width),
        6 => {
            let _ =
                super::lossless_transform::apply_predictor_transform_6(image_data, range, width);
        }
        7 => super::lossless_transform::apply_predictor_transform_7(image_data, range, width),
        8 => apply_predictor_8_wasm128(_token, image_data, range, width),
        9 => apply_predictor_9_wasm128(_token, image_data, range, width),
        10 => super::lossless_transform::apply_predictor_transform_10(image_data, range, width),
        11 => super::lossless_transform::apply_predictor_transform_11(image_data, range, width),
        12 => super::lossless_transform::apply_predictor_transform_12(image_data, range, width),
        13 => super::lossless_transform::apply_predictor_transform_13(image_data, range, width),
        _ => {}
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn apply_predictor_transform_wasm128_entry(
    _token: Wasm128Token,
    image_data: &mut [u8],
    width: u16,
    height: u16,
    size_bits: u8,
    predictor_data: &[u8],
) {
    let block_xsize = super::lossless_transform::block_xsize(width, size_bits);
    let width = usize::from(width);
    let height = usize::from(height);

    let _ = super::lossless_transform::predictor_transform_borders(image_data, width, height);

    for y in 1..height {
        let row_block_base = (y >> size_bits) * block_xsize;
        let mut run_start = 0usize;
        let mut run_end = 0usize;
        let mut run_pred = 255u8;

        for block_x in 0..block_xsize {
            let predictor = predictor_data[(row_block_base + block_x) * 4 + 1];
            let start_index = (y * width + (block_x << size_bits).max(1)) * 4;
            let end_index = (y * width + ((block_x + 1) << size_bits).min(width)) * 4;

            if predictor == run_pred && start_index == run_end {
                run_end = end_index;
            } else {
                if run_start < run_end {
                    dispatch_predictor_wasm128(
                        _token, run_pred, image_data, run_start, run_end, width,
                    );
                }
                run_pred = predictor;
                run_start = start_index;
                run_end = end_index;
            }
        }
        if run_start < run_end {
            dispatch_predictor_wasm128(_token, run_pred, image_data, run_start, run_end, width);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn add_green_to_blue_and_red_wasm128_entry(_token: Wasm128Token, image_data: &mut [u8]) {
    add_green_to_blue_and_red_wasm128(_token, image_data);
}

#[cfg(target_arch = "wasm32")]
#[rite]
fn add_green_to_blue_and_red_wasm128(_token: Wasm128Token, image_data: &mut [u8]) {
    add_green_portable(_token, image_data);
}

#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn transform_color_inverse_wasm128_entry(
    _token: Wasm128Token,
    image_data: &mut [u8],
    width: usize,
    size_bits: u8,
    transform_data: &[u8],
) {
    // TODO: WASM128 color_inverse SIMD implementation
    color_inverse_scalar_fallback(image_data, width, size_bits, transform_data);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use alloc::vec;
    #[cfg(target_arch = "x86_64")]
    use archmage::prelude::*;

    /// Test that SSE2 subtract-green matches scalar.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_green_sse2_matches_scalar() {
        let Some(token) = X64V1Token::summon() else {
            std::eprintln!("SSE2 not available, skipping test");
            return;
        };

        let mut scalar_data = vec![0u8; 256];
        let mut simd_data = vec![0u8; 256];

        for i in 0..256 {
            scalar_data[i] = i as u8;
            simd_data[i] = i as u8;
        }

        super::super::lossless_transform::apply_subtract_green_transform(&mut scalar_data);
        super::add_green_to_blue_and_red_sse2_entry(token, &mut simd_data);

        assert_eq!(
            scalar_data, simd_data,
            "SSE2 subtract-green doesn't match scalar"
        );
    }

    /// Test subtract-green with non-aligned length (not multiple of 16).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_green_sse2_odd_length() {
        let Some(token) = X64V1Token::summon() else {
            return;
        };

        // 7 pixels = 28 bytes
        let mut scalar_data = vec![0u8; 28];
        let mut simd_data = vec![0u8; 28];
        for i in 0..28 {
            scalar_data[i] = (i * 37 + 13) as u8;
            simd_data[i] = (i * 37 + 13) as u8;
        }

        super::super::lossless_transform::apply_subtract_green_transform(&mut scalar_data);
        super::add_green_to_blue_and_red_sse2_entry(token, &mut simd_data);

        assert_eq!(scalar_data, simd_data);
    }

    /// Test color transform inverse SSE2 matches scalar.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_color_transform_sse2_matches_scalar() {
        let Some(token) = X64V1Token::summon() else {
            return;
        };

        let width: usize = 32;
        let height: usize = 4;
        let size_bits: u8 = 2;
        let block_xsize = super::super::lossless_transform::block_xsize(width as u16, size_bits);

        let mut scalar_data = vec![0u8; width * height * 4];
        let mut simd_data = vec![0u8; width * height * 4];

        for i in 0..scalar_data.len() {
            scalar_data[i] = ((i * 97 + 31) % 256) as u8;
            simd_data[i] = ((i * 97 + 31) % 256) as u8;
        }

        let mut tf_data = vec![0u8; block_xsize * ((height >> size_bits) + 1) * 4];
        for chunk in tf_data.chunks_exact_mut(4) {
            chunk[0] = 23; // red_to_blue
            chunk[1] = 170; // green_to_blue (negative as i8)
            chunk[2] = 50; // green_to_red
        }

        super::super::lossless_transform::apply_color_transform(
            &mut scalar_data,
            width as u16,
            size_bits,
            &tf_data,
        );

        super::transform_color_inverse_sse2_entry(
            token,
            &mut simd_data,
            width,
            size_bits,
            &tf_data,
        );

        assert_eq!(
            scalar_data, simd_data,
            "SSE2 color transform doesn't match scalar"
        );
    }

    /// Test color transform with varied multiplier signs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_color_transform_sse2_all_multiplier_signs() {
        let Some(token) = X64V1Token::summon() else {
            return;
        };

        let multiplier_sets: &[(u8, u8, u8)] = &[
            (0, 0, 0),
            (1, 1, 1),
            (127, 127, 127),
            (128, 128, 128),
            (255, 255, 255),
            (50, 170, 200),
        ];

        for &(r2b, g2b, g2r) in multiplier_sets {
            let width: usize = 8;
            let height: usize = 1;
            let size_bits: u8 = 3;

            let block_xsize =
                super::super::lossless_transform::block_xsize(width as u16, size_bits);
            let mut scalar_data = vec![0u8; width * height * 4];
            let mut simd_data = vec![0u8; width * height * 4];

            for i in 0..scalar_data.len() {
                scalar_data[i] = (i * 73 + 17) as u8;
                simd_data[i] = (i * 73 + 17) as u8;
            }

            let mut tf_data = vec![0u8; block_xsize * 4 + 16];
            for chunk in tf_data.chunks_exact_mut(4) {
                chunk[0] = r2b;
                chunk[1] = g2b;
                chunk[2] = g2r;
            }

            super::super::lossless_transform::apply_color_transform(
                &mut scalar_data,
                width as u16,
                size_bits,
                &tf_data,
            );
            super::transform_color_inverse_sse2_entry(
                token,
                &mut simd_data,
                width,
                size_bits,
                &tf_data,
            );

            assert_eq!(
                scalar_data, simd_data,
                "Mismatch for multipliers r2b={r2b} g2b={g2b} g2r={g2r}"
            );
        }
    }

    /// Test predictor 1 (left) SSE2 matches scalar via full roundtrip.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_predictor_1_sse2_matches_scalar() {
        let Some(token) = X64V1Token::summon() else {
            return;
        };

        // Use a large block that covers entire rows (size_bits=6 → 64 pixels per block)
        let width: usize = 64;
        let height: usize = 4;
        let size_bits: u8 = 6;
        let block_xsize = super::super::lossless_transform::block_xsize(width as u16, size_bits);

        let mut scalar_data = vec![0u8; width * height * 4];
        let mut simd_data = vec![0u8; width * height * 4];

        for i in 0..scalar_data.len() {
            scalar_data[i] = (i * 53 + 7) as u8;
            simd_data[i] = (i * 53 + 7) as u8;
        }

        // Predictor data: all blocks use predictor 1 (left)
        let pred_data_len = block_xsize * ((height >> size_bits) + 2) * 4;
        let mut pred_data = vec![0u8; pred_data_len];
        for chunk in pred_data.chunks_exact_mut(4) {
            chunk[1] = 1; // predictor index in byte 1
        }

        super::super::lossless_transform::apply_predictor_transform_scalar(
            &mut scalar_data,
            width as u16,
            height as u16,
            size_bits,
            &pred_data,
        )
        .unwrap();

        super::apply_predictor_transform_sse2_entry(
            token,
            &mut simd_data,
            width as u16,
            height as u16,
            size_bits,
            &pred_data,
        );

        // Find first mismatch for debugging
        for i in 0..scalar_data.len() {
            if scalar_data[i] != simd_data[i] {
                panic!(
                    "Mismatch at byte {i} (pixel {}, channel {}, row {}, col {}): scalar={} simd={}",
                    i / 4,
                    i % 4,
                    (i / 4) / width,
                    (i / 4) % width,
                    scalar_data[i],
                    simd_data[i],
                );
            }
        }
    }

    /// Test predictor 2 (top) SSE2 matches scalar.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_predictor_2_sse2_matches_scalar() {
        let Some(token) = X64V1Token::summon() else {
            return;
        };

        let width: usize = 32;
        let height: usize = 8;
        let size_bits: u8 = 3;
        let block_xsize = super::super::lossless_transform::block_xsize(width as u16, size_bits);

        let mut scalar_data = vec![0u8; width * height * 4];
        let mut simd_data = vec![0u8; width * height * 4];

        for i in 0..scalar_data.len() {
            scalar_data[i] = (i * 41 + 19) as u8;
            simd_data[i] = (i * 41 + 19) as u8;
        }

        let mut pred_data = vec![0u8; block_xsize * (height + 1) * 4];
        for chunk in pred_data.chunks_exact_mut(4) {
            chunk[1] = 2; // predictor 2 (top)
        }

        super::super::lossless_transform::apply_predictor_transform_scalar(
            &mut scalar_data,
            width as u16,
            height as u16,
            size_bits,
            &pred_data,
        )
        .unwrap();

        super::apply_predictor_transform_sse2_entry(
            token,
            &mut simd_data,
            width as u16,
            height as u16,
            size_bits,
            &pred_data,
        );

        assert_eq!(
            scalar_data, simd_data,
            "Predictor 2 SSE2 doesn't match scalar"
        );
    }
}
