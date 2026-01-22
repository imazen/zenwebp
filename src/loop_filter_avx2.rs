//! AVX2-optimized VP8 loop filter.
//!
//! Processes 16 pixels at once by loading entire rows, matching libwebp's approach.
//! For horizontal filtering (vertical edges), uses the transpose technique:
//! load 16 rows × 8 columns, transpose to 8 × 16, filter as vertical, transpose back.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Compute the "needs filter" mask for simple filter.
/// Returns a mask where each byte is 0xFF if the pixel should be filtered, 0x00 otherwise.
/// Condition: |p0 - q0| * 2 + |p1 - q1| / 2 <= thresh
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn needs_filter_16(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i, thresh: i32) -> __m128i {
    let t = _mm_set1_epi8(thresh as i8);

    // |p0 - q0|
    let abs_p0_q0 = _mm_or_si128(
        _mm_subs_epu8(p0, q0),
        _mm_subs_epu8(q0, p0),
    );

    // |p1 - q1|
    let abs_p1_q1 = _mm_or_si128(
        _mm_subs_epu8(p1, q1),
        _mm_subs_epu8(q1, p1),
    );

    // |p0 - q0| * 2
    let doubled = _mm_adds_epu8(abs_p0_q0, abs_p0_q0);

    // |p1 - q1| / 2
    let halved = _mm_and_si128(_mm_srli_epi16(abs_p1_q1, 1), _mm_set1_epi8(0x7F));

    // |p0 - q0| * 2 + |p1 - q1| / 2
    let sum = _mm_adds_epu8(doubled, halved);

    // sum <= thresh  =>  !(sum > thresh)  =>  (thresh - sum) >= 0 using saturating sub
    let exceeds = _mm_subs_epu8(sum, t);
    _mm_cmpeq_epi8(exceeds, _mm_setzero_si128())
}

/// Get the base delta for the simple filter: clamp(p1 - q1 + 3*(q0 - p0))
/// Uses signed arithmetic with sign bit flipping.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_base_delta_16(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i) -> __m128i {
    // Convert to signed by XOR with 0x80
    let sign = _mm_set1_epi8(-128i8);
    let p1s = _mm_xor_si128(p1, sign);
    let p0s = _mm_xor_si128(p0, sign);
    let q0s = _mm_xor_si128(q0, sign);
    let q1s = _mm_xor_si128(q1, sign);

    // p1 - q1 (saturating)
    let p1_q1 = _mm_subs_epi8(p1s, q1s);

    // q0 - p0 (saturating)
    let q0_p0 = _mm_subs_epi8(q0s, p0s);

    // p1 - q1 + 3*(q0 - p0) = p1 - q1 + (q0 - p0) + (q0 - p0) + (q0 - p0)
    let s1 = _mm_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm_adds_epi8(s1, q0_p0);
    _mm_adds_epi8(s2, q0_p0)
}

/// Signed right shift by 3 for packed bytes (in signed domain).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn signed_shift_right_3(v: __m128i) -> __m128i {
    // For signed bytes, we need to handle sign extension properly.
    // Unpack to 16-bit, shift, pack back.
    let lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 11); // sign-extend and shift
    let hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 11);
    _mm_packs_epi16(lo, hi)
}

/// Apply the simple filter to p0 and q0 given the filter value.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn do_simple_filter_16(p0: &mut __m128i, q0: &mut __m128i, fl: __m128i) {
    let sign = _mm_set1_epi8(-128i8);
    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);

    // v3 = (fl + 3) >> 3
    // v4 = (fl + 4) >> 3
    let v3 = _mm_adds_epi8(fl, k3);
    let v4 = _mm_adds_epi8(fl, k4);

    let v3 = signed_shift_right_3(v3);
    let v4 = signed_shift_right_3(v4);

    // Convert p0, q0 to signed
    let mut p0s = _mm_xor_si128(*p0, sign);
    let mut q0s = _mm_xor_si128(*q0, sign);

    // q0 -= v4, p0 += v3
    q0s = _mm_subs_epi8(q0s, v4);
    p0s = _mm_adds_epi8(p0s, v3);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(p0s, sign);
    *q0 = _mm_xor_si128(q0s, sign);
}

/// Apply simple vertical filter to 16 pixels across a horizontal edge.
///
/// This filters the edge between row (point - stride) and row (point).
/// Processes 16 consecutive pixels in a single call.
///
/// # Safety
/// Requires SSE4.1. Buffer must have at least point + stride + 16 bytes.
/// point must be >= 2 * stride (for p1 access).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_v_filter16(pixels: &mut [u8], point: usize, stride: usize, thresh: i32) {
    // Load 16 pixels from each of the 4 rows
    let p1 = _mm_loadu_si128(pixels.as_ptr().add(point - 2 * stride) as *const __m128i);
    let mut p0 = _mm_loadu_si128(pixels.as_ptr().add(point - stride) as *const __m128i);
    let mut q0 = _mm_loadu_si128(pixels.as_ptr().add(point) as *const __m128i);
    let q1 = _mm_loadu_si128(pixels.as_ptr().add(point + stride) as *const __m128i);

    // Check which pixels need filtering
    let mask = needs_filter_16(p1, p0, q0, q1, thresh);

    // Get filter delta
    let fl = get_base_delta_16(p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);

    // Apply filter
    do_simple_filter_16(&mut p0, &mut q0, fl_masked);

    // Store results
    _mm_storeu_si128(pixels.as_mut_ptr().add(point - stride) as *mut __m128i, p0);
    _mm_storeu_si128(pixels.as_mut_ptr().add(point) as *mut __m128i, q0);
}

/// Transpose an 8x16 matrix of bytes to 16x8.
/// Input: 16 __m128i values, each containing 8 bytes (low 64 bits used).
/// Output: 8 __m128i values, each containing 16 bytes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose_8x16_to_16x8(
    rows: &[__m128i; 16],
) -> [__m128i; 8] {
    // Stage 1: interleave pairs
    let t0 = _mm_unpacklo_epi8(rows[0], rows[1]);
    let t1 = _mm_unpacklo_epi8(rows[2], rows[3]);
    let t2 = _mm_unpacklo_epi8(rows[4], rows[5]);
    let t3 = _mm_unpacklo_epi8(rows[6], rows[7]);
    let t4 = _mm_unpacklo_epi8(rows[8], rows[9]);
    let t5 = _mm_unpacklo_epi8(rows[10], rows[11]);
    let t6 = _mm_unpacklo_epi8(rows[12], rows[13]);
    let t7 = _mm_unpacklo_epi8(rows[14], rows[15]);

    // Stage 2: interleave 16-bit pairs
    let u0 = _mm_unpacklo_epi16(t0, t1);
    let u1 = _mm_unpackhi_epi16(t0, t1);
    let u2 = _mm_unpacklo_epi16(t2, t3);
    let u3 = _mm_unpackhi_epi16(t2, t3);
    let u4 = _mm_unpacklo_epi16(t4, t5);
    let u5 = _mm_unpackhi_epi16(t4, t5);
    let u6 = _mm_unpacklo_epi16(t6, t7);
    let u7 = _mm_unpackhi_epi16(t6, t7);

    // Stage 3: interleave 32-bit pairs
    let v0 = _mm_unpacklo_epi32(u0, u2);
    let v1 = _mm_unpackhi_epi32(u0, u2);
    let v2 = _mm_unpacklo_epi32(u4, u6);
    let v3 = _mm_unpackhi_epi32(u4, u6);
    let v4 = _mm_unpacklo_epi32(u1, u3);
    let v5 = _mm_unpackhi_epi32(u1, u3);
    let v6 = _mm_unpacklo_epi32(u5, u7);
    let v7 = _mm_unpackhi_epi32(u5, u7);

    // Stage 4: interleave 64-bit to get final columns
    [
        _mm_unpacklo_epi64(v0, v2), // column 0 (p3)
        _mm_unpackhi_epi64(v0, v2), // column 1 (p2)
        _mm_unpacklo_epi64(v1, v3), // column 2 (p1)
        _mm_unpackhi_epi64(v1, v3), // column 3 (p0)
        _mm_unpacklo_epi64(v4, v6), // column 4 (q0)
        _mm_unpackhi_epi64(v4, v6), // column 5 (q1)
        _mm_unpacklo_epi64(v5, v7), // column 6 (q2)
        _mm_unpackhi_epi64(v5, v7), // column 7 (q3)
    ]
}

/// Transpose 4 columns (p1, p0, q0, q1) of 16 bytes each back to 16 rows of 4 bytes.
/// Returns values suitable for storing as 32-bit integers per row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose_4x16_to_16x4(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
) -> [i32; 16] {
    // Interleave p1,p0 and q0,q1
    let p1p0_lo = _mm_unpacklo_epi8(p1, p0); // rows 0-7: p1[i], p0[i]
    let p1p0_hi = _mm_unpackhi_epi8(p1, p0); // rows 8-15
    let q0q1_lo = _mm_unpacklo_epi8(q0, q1);
    let q0q1_hi = _mm_unpackhi_epi8(q0, q1);

    // Combine to 32-bit: p1, p0, q0, q1
    let r0 = _mm_unpacklo_epi16(p1p0_lo, q0q1_lo); // rows 0-3
    let r1 = _mm_unpackhi_epi16(p1p0_lo, q0q1_lo); // rows 4-7
    let r2 = _mm_unpacklo_epi16(p1p0_hi, q0q1_hi); // rows 8-11
    let r3 = _mm_unpackhi_epi16(p1p0_hi, q0q1_hi); // rows 12-15

    // Extract 32-bit values
    let mut result = [0i32; 16];
    result[0] = _mm_extract_epi32(r0, 0);
    result[1] = _mm_extract_epi32(r0, 1);
    result[2] = _mm_extract_epi32(r0, 2);
    result[3] = _mm_extract_epi32(r0, 3);
    result[4] = _mm_extract_epi32(r1, 0);
    result[5] = _mm_extract_epi32(r1, 1);
    result[6] = _mm_extract_epi32(r1, 2);
    result[7] = _mm_extract_epi32(r1, 3);
    result[8] = _mm_extract_epi32(r2, 0);
    result[9] = _mm_extract_epi32(r2, 1);
    result[10] = _mm_extract_epi32(r2, 2);
    result[11] = _mm_extract_epi32(r2, 3);
    result[12] = _mm_extract_epi32(r3, 0);
    result[13] = _mm_extract_epi32(r3, 1);
    result[14] = _mm_extract_epi32(r3, 2);
    result[15] = _mm_extract_epi32(r3, 3);
    result
}

/// Apply simple horizontal filter to 16 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses the transpose technique: load 16 rows × 8 columns, transpose,
/// apply vertical filter logic, transpose back, store.
///
/// # Safety
/// Requires SSE4.1. Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_h_filter16(pixels: &mut [u8], x: usize, y_start: usize, stride: usize, thresh: i32) {
    // Load 16 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    // The edge is between p0 (x-1) and q0 (x), so we load from x-4 to x+3
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = _mm_loadl_epi64(pixels.as_ptr().add(row_start) as *const __m128i);
    }

    // Transpose 8x16 to 16x8: now we have 8 columns of 16 pixels each
    let cols = transpose_8x16_to_16x8(&rows);
    // cols[2] = p1 (16 pixels from 16 different rows)
    // cols[3] = p0
    // cols[4] = q0
    // cols[5] = q1

    let p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let q1 = cols[5];

    // Apply simple filter (same logic as vertical, but on transposed data)
    let mask = needs_filter_16(p1, p0, q0, q1, thresh);
    let fl = get_base_delta_16(p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);
    do_simple_filter_16(&mut p0, &mut q0, fl_masked);

    // Transpose back: convert 4 columns of 16 to 16 rows of 4
    let packed = transpose_4x16_to_16x4(p1, p0, q0, q1);

    // Store 4 bytes per row (p1, p0, q0, q1) using unaligned write
    for (i, &val) in packed.iter().enumerate() {
        let row_start = (y_start + i) * stride + x - 2;
        let ptr = pixels.as_mut_ptr().add(row_start) as *mut i32;
        std::ptr::write_unaligned(ptr, val);
    }
}

/// Apply simple vertical filter to entire macroblock edge (16 pixels).
/// This is the main entry point for filtering horizontal edges between macroblocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_mb_edge_v(
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let point = mb_y * 16 * stride + mb_x * 16;
    simple_v_filter16(pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to entire macroblock edge (16 rows).
/// This is the main entry point for filtering vertical edges between macroblocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_mb_edge_h(
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16;
    let y_start = mb_y * 16;
    simple_h_filter16(pixels, x, y_start, stride, thresh);
}

/// Apply simple vertical filter to a subblock edge within a macroblock.
/// y_offset is the row offset within the macroblock (4, 8, or 12).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_subblock_edge_v(
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    y_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let point = (mb_y * 16 + y_offset) * stride + mb_x * 16;
    simple_v_filter16(pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to a subblock edge within a macroblock.
/// x_offset is the column offset within the macroblock (4, 8, or 12).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_subblock_edge_h(
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    x_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16 + x_offset;
    let y_start = mb_y * 16;
    simple_h_filter16(pixels, x, y_start, stride, thresh);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scalar simple filter for comparison
    fn scalar_simple_filter(p1: u8, p0: u8, q0: u8, q1: u8, thresh: i32) -> (u8, u8) {
        // Check threshold
        let diff_p0_q0 = (p0 as i32 - q0 as i32).abs();
        let diff_p1_q1 = (p1 as i32 - q1 as i32).abs();
        if diff_p0_q0 * 2 + diff_p1_q1 / 2 > thresh {
            return (p0, q0); // No filtering
        }

        // Convert to signed
        let p1s = p1 as i32 - 128;
        let p0s = p0 as i32 - 128;
        let q0s = q0 as i32 - 128;
        let q1s = q1 as i32 - 128;

        // Compute filter value
        let a = (p1s - q1s + 3 * (q0s - p0s)).clamp(-128, 127);
        let a_plus_4 = (a + 4) >> 3;
        let a_plus_3 = (a + 3) >> 3;

        // Apply
        let new_q0 = (q0s - a_plus_4).clamp(-128, 127) + 128;
        let new_p0 = (p0s + a_plus_3).clamp(-128, 127) + 128;

        (new_p0 as u8, new_q0 as u8)
    }

    #[test]
    fn test_simple_v_filter16_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        let stride = 32;
        let mut pixels = vec![128u8; stride * 8];
        let mut pixels_scalar = pixels.clone();

        // Set up a gradient that should be filtered
        // Row 0 = p1, Row 1 = p0, Row 2 = q0, Row 3 = q1
        for x in 0..16 {
            pixels[x] = 100;                    // row 0: p1
            pixels[stride + x] = 110;           // row 1: p0
            pixels[2 * stride + x] = 140;       // row 2: q0
            pixels[3 * stride + x] = 150;       // row 3: q1

            pixels_scalar[x] = 100;
            pixels_scalar[stride + x] = 110;
            pixels_scalar[2 * stride + x] = 140;
            pixels_scalar[3 * stride + x] = 150;
        }

        let thresh = 40;

        // Apply SIMD filter (edge between row 1 and row 2, so point = 2 * stride)
        unsafe {
            simple_v_filter16(&mut pixels, 2 * stride, stride, thresh);
        }

        // Apply scalar filter
        for x in 0..16 {
            let p1 = pixels_scalar[x];
            let p0 = pixels_scalar[stride + x];
            let q0 = pixels_scalar[2 * stride + x];
            let q1 = pixels_scalar[3 * stride + x];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[stride + x] = new_p0;
            pixels_scalar[2 * stride + x] = new_q0;
        }

        // Compare
        for x in 0..16 {
            assert_eq!(
                pixels[stride + x], pixels_scalar[stride + x],
                "p0 mismatch at x={}", x
            );
            assert_eq!(
                pixels[2 * stride + x], pixels_scalar[2 * stride + x],
                "q0 mismatch at x={}", x
            );
        }
    }

    #[test]
    fn test_simple_h_filter16_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        let stride = 32;
        let mut pixels = vec![128u8; stride * 20];
        let mut pixels_scalar = pixels.clone();

        // Set up a vertical edge with gradient
        for y in 0..16 {
            pixels[y * stride + 2] = 100; // p1
            pixels[y * stride + 3] = 110; // p0
            pixels[y * stride + 4] = 140; // q0
            pixels[y * stride + 5] = 150; // q1

            pixels_scalar[y * stride + 2] = 100;
            pixels_scalar[y * stride + 3] = 110;
            pixels_scalar[y * stride + 4] = 140;
            pixels_scalar[y * stride + 5] = 150;
        }

        let thresh = 40;

        // Apply SIMD filter (edge at x=4)
        unsafe {
            simple_h_filter16(&mut pixels, 4, 0, stride, thresh);
        }

        // Apply scalar filter
        for y in 0..16 {
            let p1 = pixels_scalar[y * stride + 2];
            let p0 = pixels_scalar[y * stride + 3];
            let q0 = pixels_scalar[y * stride + 4];
            let q1 = pixels_scalar[y * stride + 5];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[y * stride + 3] = new_p0;
            pixels_scalar[y * stride + 4] = new_q0;
        }

        // Compare
        for y in 0..16 {
            assert_eq!(
                pixels[y * stride + 3], pixels_scalar[y * stride + 3],
                "p0 mismatch at y={}", y
            );
            assert_eq!(
                pixels[y * stride + 4], pixels_scalar[y * stride + 4],
                "q0 mismatch at y={}", y
            );
        }
    }
}
