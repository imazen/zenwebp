//! VP8 loop filter: scalar, SSE2/AVX2, NEON, and WASM SIMD implementations.
//!
//! All platform variants live in this single file. The `#[rite]` and `#[arcane]`
//! macros from archmage handle `#[cfg(target_arch)]` gating automatically.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(dead_code)]

use archmage::prelude::*;
use core::convert::TryFrom;

// Safe memory ops aliases (platform-specific, needed for load/store helpers)
#[cfg(target_arch = "aarch64")]
use archmage::intrinsics::aarch64 as simd_mem_neon;
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem_x86;

// ============================================================================
// Scalar filter implementations
// ============================================================================

#[inline]
fn c(val: i32) -> i32 {
    val.clamp(-128, 127)
}

//unsigned to signed
#[inline]
fn u2s(val: u8) -> i32 {
    i32::from(val) - 128
}

//signed to unsigned
#[inline]
fn s2u(val: i32) -> u8 {
    (c(val) + 128) as u8
}

#[inline]
const fn diff(val1: u8, val2: u8) -> u8 {
    u8::abs_diff(val1, val2)
}

/// Used in both the simple and normal filters described in 15.2 and 15.3
///
/// Adjusts the 2 middle pixels in a vertical loop filter
fn common_adjust_vertical(
    use_outer_taps: bool,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) -> i32 {
    let p1 = u2s(pixels[point - 2 * stride]);
    let p0 = u2s(pixels[point - stride]);
    let q0 = u2s(pixels[point]);
    let q1 = u2s(pixels[point + stride]);

    //value for the outer 2 pixels
    let outer = if use_outer_taps { c(p1 - q1) } else { 0 };

    let a = c(outer + 3 * (q0 - p0));

    let b = (c(a + 3)) >> 3;

    let a = (c(a + 4)) >> 3;

    pixels[point] = s2u(q0 - a);
    pixels[point - stride] = s2u(p0 + b);

    a
}

/// Used in both the simple and normal filters described in 15.2 and 15.3
///
/// Adjusts the 2 middle pixels in a horizontal loop filter
fn common_adjust_horizontal(use_outer_taps: bool, pixels: &mut [u8]) -> i32 {
    let p1 = u2s(pixels[2]);
    let p0 = u2s(pixels[3]);
    let q0 = u2s(pixels[4]);
    let q1 = u2s(pixels[5]);

    //value for the outer 2 pixels
    let outer = if use_outer_taps { c(p1 - q1) } else { 0 };

    let a = c(outer + 3 * (q0 - p0));

    let b = (c(a + 3)) >> 3;

    let a = (c(a + 4)) >> 3;

    pixels[4] = s2u(q0 - a);
    pixels[3] = s2u(p0 + b);

    a
}

#[inline]
fn simple_threshold_vertical(
    filter_limit: i32,
    pixels: &[u8],
    point: usize,
    stride: usize,
) -> bool {
    i32::from(diff(pixels[point - stride], pixels[point])) * 2
        + i32::from(diff(pixels[point - 2 * stride], pixels[point + stride])) / 2
        <= filter_limit
}

#[inline]
fn simple_threshold_horizontal(filter_limit: i32, pixels: &[u8]) -> bool {
    assert!(pixels.len() >= 6); // one bounds check up front eliminates all subsequent checks in this function
    i32::from(diff(pixels[3], pixels[4])) * 2 + i32::from(diff(pixels[2], pixels[5])) / 2
        <= filter_limit
}

fn should_filter_vertical(
    interior_limit: u8,
    edge_limit: u8,
    pixels: &[u8],
    point: usize,
    stride: usize,
) -> bool {
    simple_threshold_vertical(i32::from(edge_limit), pixels, point, stride)
        // this looks like an erroneous way to compute differences between 8 points, but isn't:
        // there are actually only 6 diff comparisons required as per the spec:
        // https://www.rfc-editor.org/rfc/rfc6386#section-20.6
        && diff(pixels[point - 4 * stride], pixels[point - 3 * stride]) <= interior_limit
        && diff(pixels[point - 3 * stride], pixels[point - 2 * stride]) <= interior_limit
        && diff(pixels[point - 2 * stride], pixels[point - stride]) <= interior_limit
        && diff(pixels[point + 3 * stride], pixels[point + 2 * stride]) <= interior_limit
        && diff(pixels[point + 2 * stride], pixels[point + stride]) <= interior_limit
        && diff(pixels[point + stride], pixels[point]) <= interior_limit
}

fn should_filter_horizontal(interior_limit: u8, edge_limit: u8, pixels: &[u8]) -> bool {
    assert!(pixels.len() >= 8); // one bounds check up front eliminates all subsequent checks in this function
    simple_threshold_horizontal(i32::from(edge_limit), pixels)
        // this looks like an erroneous way to compute differences between 8 points, but isn't:
        // there are actually only 6 diff comparisons required as per the spec:
        // https://www.rfc-editor.org/rfc/rfc6386#section-20.6
        && diff(pixels[0], pixels[1]) <= interior_limit
        && diff(pixels[1], pixels[2]) <= interior_limit
        && diff(pixels[2], pixels[3]) <= interior_limit
        && diff(pixels[7], pixels[6]) <= interior_limit
        && diff(pixels[6], pixels[5]) <= interior_limit
        && diff(pixels[5], pixels[4]) <= interior_limit
}

#[inline]
fn high_edge_variance_vertical(threshold: u8, pixels: &[u8], point: usize, stride: usize) -> bool {
    diff(pixels[point - 2 * stride], pixels[point - stride]) > threshold
        || diff(pixels[point + stride], pixels[point]) > threshold
}

#[inline]
fn high_edge_variance_horizontal(threshold: u8, pixels: &[u8]) -> bool {
    diff(pixels[2], pixels[3]) > threshold || diff(pixels[5], pixels[4]) > threshold
}

/// Part of the simple filter described in 15.2 in the specification
///
/// Affects 4 pixels on an edge(2 each side)
pub(crate) fn simple_segment_vertical(
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if simple_threshold_vertical(i32::from(edge_limit), pixels, point, stride) {
        common_adjust_vertical(true, pixels, point, stride);
    }
}

/// Part of the simple filter described in 15.2 in the specification
///
/// Affects 4 pixels on an edge(2 each side)
pub(crate) fn simple_segment_horizontal(edge_limit: u8, pixels: &mut [u8]) {
    if simple_threshold_horizontal(i32::from(edge_limit), pixels) {
        common_adjust_horizontal(true, pixels);
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the edges between subblocks inside a macroblock
pub(crate) fn subblock_filter_vertical(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if should_filter_vertical(interior_limit, edge_limit, pixels, point, stride) {
        let hv = high_edge_variance_vertical(hev_threshold, pixels, point, stride);

        let a = (common_adjust_vertical(hv, pixels, point, stride) + 1) >> 1;

        if !hv {
            pixels[point + stride] = s2u(u2s(pixels[point + stride]) - a);
            pixels[point - 2 * stride] = s2u(u2s(pixels[point - 2 * stride]) + a);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the edges between subblocks inside a macroblock
pub(crate) fn subblock_filter_horizontal(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
) {
    if should_filter_horizontal(interior_limit, edge_limit, pixels) {
        let hv = high_edge_variance_horizontal(hev_threshold, pixels);

        let a = (common_adjust_horizontal(hv, pixels) + 1) >> 1;

        if !hv {
            pixels[5] = s2u(u2s(pixels[5]) - a);
            pixels[2] = s2u(u2s(pixels[2]) + a);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the vertical edges between macroblocks\
/// The point passed in must be the first vertical pixel on the bottom macroblock
pub(crate) fn macroblock_filter_vertical(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if should_filter_vertical(interior_limit, edge_limit, pixels, point, stride) {
        if !high_edge_variance_vertical(hev_threshold, pixels, point, stride) {
            // p0-3 are the pixels on the left macroblock from right to left
            let p2 = u2s(pixels[point - 3 * stride]);
            let p1 = u2s(pixels[point - 2 * stride]);
            let p0 = u2s(pixels[point - stride]);
            // q0-3 are the pixels on the right macroblock from left to right
            let q0 = u2s(pixels[point]);
            let q1 = u2s(pixels[point + stride]);
            let q2 = u2s(pixels[point + 2 * stride]);

            let w = c(c(p1 - q1) + 3 * (q0 - p0));

            let a = c((27 * w + 63) >> 7);

            pixels[point] = s2u(q0 - a);
            pixels[point - stride] = s2u(p0 + a);

            let a = c((18 * w + 63) >> 7);

            pixels[point + stride] = s2u(q1 - a);
            pixels[point - 2 * stride] = s2u(p1 + a);

            let a = c((9 * w + 63) >> 7);

            pixels[point + 2 * stride] = s2u(q2 - a);
            pixels[point - 3 * stride] = s2u(p2 + a);
        } else {
            common_adjust_vertical(true, pixels, point, stride);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the horizontal edges between macroblocks\
/// The pixels passed in must be a slice containing the 4 pixels on each macroblock
pub(crate) fn macroblock_filter_horizontal(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
) {
    assert!(pixels.len() >= 8);
    if should_filter_horizontal(interior_limit, edge_limit, pixels) {
        if !high_edge_variance_horizontal(hev_threshold, pixels) {
            // p0-3 are the pixels on the left macroblock from right to left
            let p2 = u2s(pixels[1]);
            let p1 = u2s(pixels[2]);
            let p0 = u2s(pixels[3]);
            // q0-3 are the pixels on the right macroblock from left to right
            let q0 = u2s(pixels[4]);
            let q1 = u2s(pixels[5]);
            let q2 = u2s(pixels[6]);

            let w = c(c(p1 - q1) + 3 * (q0 - p0));

            let a = c((27 * w + 63) >> 7);

            pixels[4] = s2u(q0 - a);
            pixels[3] = s2u(p0 + a);

            let a = c((18 * w + 63) >> 7);

            pixels[5] = s2u(q1 - a);
            pixels[2] = s2u(p1 + a);

            let a = c((9 * w + 63) >> 7);

            pixels[6] = s2u(q2 - a);
            pixels[1] = s2u(p2 + a);
        } else {
            common_adjust_horizontal(true, pixels);
        }
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benches {
    use super::*;
    use test::{Bencher, black_box};

    #[rustfmt::skip]
    const TEST_DATA: [u8; 8 * 8] = [
        177, 192, 179, 181, 185, 174, 186, 193,
        185, 180, 175, 179, 175, 190, 189, 190,
        185, 181, 177, 190, 190, 174, 176, 188,
        192, 179, 186, 175, 190, 184, 190, 175,
        175, 183, 183, 190, 187, 186, 176, 181,
        183, 177, 182, 185, 183, 179, 178, 181,
        191, 183, 188, 181, 180, 193, 185, 180,
        177, 182, 177, 178, 179, 178, 191, 178,
    ];

    #[bench]
    fn measure_horizontal_macroblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for y in 0..8 {
                macroblock_filter_horizontal(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data[y * stride..][..8],
                );
                black_box(());
            }
        });
    }

    #[bench]
    fn measure_vertical_macroblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for x in 0..8 {
                macroblock_filter_vertical(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data,
                    4 * stride + x,
                    stride,
                );
                black_box(());
            }
        });
    }

    #[bench]
    fn measure_horizontal_subblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for y in 0usize..8 {
                subblock_filter_horizontal(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data[y * stride..][..8],
                );
                black_box(());
            }
        });
    }

    #[bench]
    fn measure_vertical_subblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for x in 0..8 {
                subblock_filter_vertical(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data,
                    4 * stride + x,
                    stride,
                );
                black_box(());
            }
        });
    }

    #[bench]
    fn measure_simple_segment_horizontal_filter(b: &mut Bencher) {
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for y in 0usize..8 {
                simple_segment_horizontal(edge_limit, &mut data[y * stride..][..8]);
                black_box(());
            }
        });
    }

    #[bench]
    fn measure_simple_segment_vertical_filter(b: &mut Bencher) {
        let edge_limit = 15;

        let mut data = TEST_DATA;
        let stride = 8;

        b.iter(|| {
            for x in 0usize..16 {
                simple_segment_vertical(edge_limit, &mut data, 4 * stride + x, stride);
                black_box(());
            }
        });
    }
}

// ============================================================================
// SSE2/AVX2 filter implementations (x86_64)
// ============================================================================

/// Maximum stride for bounds-check-free filtering.
/// WebP max dimension is 16383, rounded up to MB boundary = 16384.
const MAX_STRIDE: usize = 16384;

/// Fixed region size for simple vertical filter: 4 rows (p1, p0, q0, q1) plus 16 bytes.
const V_FILTER_REGION: usize = 3 * MAX_STRIDE + 16;

/// Fixed region size for normal vertical filter: 8 rows (p3-p0, q0-q3) plus 16 bytes.
const V_FILTER_NORMAL_REGION: usize = 7 * MAX_STRIDE + 16;

/// Fixed region size for simple horizontal filter: 16 rows of 4 bytes each.
const H_FILTER_SIMPLE_REGION: usize = 15 * MAX_STRIDE + 4;

/// Fixed region size for normal horizontal filter: 16 rows of 8 bytes each.
const H_FILTER_NORMAL_REGION: usize = 15 * MAX_STRIDE + 8;

/// Fixed region size for normal horizontal filter fused 3-edge: 16 rows of 16 bytes each.
const H_FILTER_FUSED_REGION: usize = 15 * MAX_STRIDE + 16;

/// Fixed region size for UV horizontal filter: 8 rows of 8 bytes each.
const H_FILTER_UV_REGION: usize = 7 * MAX_STRIDE + 8;

/// Fixed region size for UV vertical filter: 8 rows of 16 bytes (load 8, store needs 16).
const V_FILTER_UV_REGION: usize = 7 * MAX_STRIDE + 16;

/// Compute the "needs filter" mask for simple filter.
/// Returns a mask where each byte is 0xFF if the pixel should be filtered, 0x00 otherwise.
/// Condition: |p0 - q0| * 2 + |p1 - q1| / 2 <= thresh
#[cfg(target_arch = "x86_64")]
#[rite]
fn needs_filter_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    thresh: i32,
) -> __m128i {
    let t = _mm_set1_epi8(thresh as i8);

    // |p0 - q0|
    let abs_p0_q0 = _mm_or_si128(_mm_subs_epu8(p0, q0), _mm_subs_epu8(q0, p0));

    // |p1 - q1|
    let abs_p1_q1 = _mm_or_si128(_mm_subs_epu8(p1, q1), _mm_subs_epu8(q1, p1));

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
#[rite]
fn get_base_delta_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
) -> __m128i {
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
#[rite]
fn signed_shift_right_3(_token: X64V3Token, v: __m128i) -> __m128i {
    // For signed bytes, we need to handle sign extension properly.
    // Unpack to 16-bit, shift, pack back.
    let lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 11); // sign-extend and shift
    let hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 11);
    _mm_packs_epi16(lo, hi)
}

/// Apply the simple filter to p0 and q0 given the filter value.
#[cfg(target_arch = "x86_64")]
#[rite]
fn do_simple_filter_16(_token: X64V3Token, p0: &mut __m128i, q0: &mut __m128i, fl: __m128i) {
    let sign = _mm_set1_epi8(-128i8);
    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);

    // v3 = (fl + 3) >> 3
    // v4 = (fl + 4) >> 3
    let v3 = _mm_adds_epi8(fl, k3);
    let v4 = _mm_adds_epi8(fl, k4);

    let v3 = signed_shift_right_3(_token, v3);
    let v4 = signed_shift_right_3(_token, v4);

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
/// Buffer must have at least point + stride + 16 bytes.
/// point must be >= 2 * stride (for p1 access).
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_v_filter16(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    // Requires pixel buffer to have FILTER_PADDING bytes at the end.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 2 * stride;
    let region: &mut [u8; V_FILTER_REGION] =
        <&mut [u8; V_FILTER_REGION]>::try_from(&mut pixels[start..start + V_FILTER_REGION])
            .expect("simple_v_filter16: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region - compiler proves these are in-bounds
    let off_p1 = 0;
    let off_p0 = stride;
    let off_q0 = 2 * stride;
    let off_q1 = 3 * stride;

    // Load 16 pixels from each row - NO per-access bounds checks
    let p1 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let q1 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());

    // Check which pixels need filtering
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);

    // Get filter delta
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);

    // Apply filter
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Store results - NO per-access bounds checks
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
}

// ============================================================================
// AVX2 32-pixel filters (2x throughput vs SSE2)
// ============================================================================

/// Fixed region size for 32-pixel simple vertical filter.
const V_FILTER_REGION_32: usize = 3 * MAX_STRIDE + 32;

/// Compute "needs filter" mask for 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn needs_filter_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    thresh: i32,
) -> __m256i {
    let t = _mm256_set1_epi8(thresh as i8);

    // |p0 - q0|
    let abs_p0_q0 = _mm256_or_si256(_mm256_subs_epu8(p0, q0), _mm256_subs_epu8(q0, p0));

    // |p1 - q1|
    let abs_p1_q1 = _mm256_or_si256(_mm256_subs_epu8(p1, q1), _mm256_subs_epu8(q1, p1));

    // |p0 - q0| * 2
    let doubled = _mm256_adds_epu8(abs_p0_q0, abs_p0_q0);

    // |p1 - q1| / 2
    let halved = _mm256_and_si256(_mm256_srli_epi16(abs_p1_q1, 1), _mm256_set1_epi8(0x7F));

    // |p0 - q0| * 2 + |p1 - q1| / 2
    let sum = _mm256_adds_epu8(doubled, halved);

    // sum <= thresh
    let exceeds = _mm256_subs_epu8(sum, t);
    _mm256_cmpeq_epi8(exceeds, _mm256_setzero_si256())
}

/// Get base delta for 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn get_base_delta_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
) -> __m256i {
    // Convert to signed by XOR with 0x80
    let sign = _mm256_set1_epi8(-128i8);
    let p1s = _mm256_xor_si256(p1, sign);
    let p0s = _mm256_xor_si256(p0, sign);
    let q0s = _mm256_xor_si256(q0, sign);
    let q1s = _mm256_xor_si256(q1, sign);

    // p1 - q1 (saturating)
    let p1_q1 = _mm256_subs_epi8(p1s, q1s);

    // q0 - p0 (saturating)
    let q0_p0 = _mm256_subs_epi8(q0s, p0s);

    // p1 - q1 + 3*(q0 - p0)
    let s1 = _mm256_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm256_adds_epi8(s1, q0_p0);
    _mm256_adds_epi8(s2, q0_p0)
}

/// Signed right shift by 3 for packed bytes using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn signed_shift_right_3_avx2(_token: X64V3Token, v: __m256i) -> __m256i {
    // Unpack to 16-bit, shift, pack back
    let lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(v, v), 11);
    let hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(v, v), 11);
    _mm256_packs_epi16(lo, hi)
}

/// Apply simple filter to 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn do_simple_filter_32(_token: X64V3Token, p0: &mut __m256i, q0: &mut __m256i, fl: __m256i) {
    let sign = _mm256_set1_epi8(-128i8);
    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);

    // v3 = (fl + 3) >> 3, v4 = (fl + 4) >> 3
    let v3 = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(fl, k3));
    let v4 = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(fl, k4));

    // Convert p0, q0 to signed
    let mut p0s = _mm256_xor_si256(*p0, sign);
    let mut q0s = _mm256_xor_si256(*q0, sign);

    // q0 -= v4, p0 += v3
    q0s = _mm256_subs_epi8(q0s, v4);
    p0s = _mm256_adds_epi8(p0s, v3);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(p0s, sign);
    *q0 = _mm256_xor_si256(q0s, sign);
}

/// Apply simple vertical filter to 32 pixels using AVX2.
/// Processes 32 consecutive pixels in a single call - 2x throughput vs SSE2.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_v_filter32(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    // Fixed-region approach for bounds check elimination
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 2 * stride;
    let region: &mut [u8; V_FILTER_REGION_32] =
        <&mut [u8; V_FILTER_REGION_32]>::try_from(&mut pixels[start..start + V_FILTER_REGION_32])
            .expect("simple_v_filter32: buffer too small");

    let off_p1 = 0;
    let off_p0 = stride;
    let off_q0 = 2 * stride;
    let off_q1 = 3 * stride;

    // Load 32 pixels from each row using AVX2
    let p1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let q1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());

    // Check which pixels need filtering
    let mask = needs_filter_32(_token, p1, p0, q0, q1, thresh);

    // Get filter delta and mask it
    let fl = get_base_delta_32(_token, p1, p0, q0, q1);
    let fl_masked = _mm256_and_si256(fl, mask);

    // Apply filter
    do_simple_filter_32(_token, &mut p0, &mut q0, fl_masked);

    // Store results
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
}

// =============================================================================
// AVX2 32-pixel Normal Filter Helpers
// =============================================================================

/// Fixed region size for normal vertical filter with 32 pixels.
const V_FILTER_NORMAL_REGION_32: usize = 7 * MAX_STRIDE + 32;

/// Check if normal filtering is needed for 32 pixels using AVX2.
/// Returns a mask where each byte is 0xFF if the pixel should be filtered.
#[cfg(target_arch = "x86_64")]
#[rite]
fn needs_filter_normal_32(
    _token: X64V3Token,
    p3: __m256i,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
    q3: __m256i,
    edge_limit: i32,
    interior_limit: i32,
) -> __m256i {
    // First check simple threshold
    let simple_mask = needs_filter_32(_token, p1, p0, q0, q1, edge_limit);

    let i_limit = _mm256_set1_epi8(interior_limit as i8);

    // Helper macro for abs diff
    macro_rules! abs_diff {
        ($a:expr, $b:expr) => {
            _mm256_or_si256(_mm256_subs_epu8($a, $b), _mm256_subs_epu8($b, $a))
        };
    }

    let d_p3_p2 = abs_diff!(p3, p2);
    let d_p2_p1 = abs_diff!(p2, p1);
    let d_p1_p0 = abs_diff!(p1, p0);
    let d_q0_q1 = abs_diff!(q0, q1);
    let d_q1_q2 = abs_diff!(q1, q2);
    let d_q2_q3 = abs_diff!(q2, q3);

    // Take max of all differences
    let max1 = _mm256_max_epu8(d_p3_p2, d_p2_p1);
    let max2 = _mm256_max_epu8(d_p1_p0, d_q0_q1);
    let max3 = _mm256_max_epu8(d_q1_q2, d_q2_q3);
    let max4 = _mm256_max_epu8(max1, max2);
    let max_diff = _mm256_max_epu8(max3, max4);

    // Check if max_diff <= interior_limit
    let exceeds = _mm256_subs_epu8(max_diff, i_limit);
    let interior_ok = _mm256_cmpeq_epi8(exceeds, _mm256_setzero_si256());

    // Both conditions must be true
    _mm256_and_si256(simple_mask, interior_ok)
}

/// Check high edge variance for 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn high_edge_variance_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    hev_thresh: i32,
) -> __m256i {
    let t = _mm256_set1_epi8(hev_thresh as i8);

    // |p1 - p0|
    let d_p1_p0 = _mm256_or_si256(_mm256_subs_epu8(p1, p0), _mm256_subs_epu8(p0, p1));

    // |q1 - q0|
    let d_q1_q0 = _mm256_or_si256(_mm256_subs_epu8(q1, q0), _mm256_subs_epu8(q0, q1));

    // Check if either > thresh
    let p_exceeds = _mm256_subs_epu8(d_p1_p0, t);
    let q_exceeds = _mm256_subs_epu8(d_q1_q0, t);

    // hev = true if exceeds > 0
    let p_hev = _mm256_xor_si256(
        _mm256_cmpeq_epi8(p_exceeds, _mm256_setzero_si256()),
        _mm256_set1_epi8(-1),
    );
    let q_hev = _mm256_xor_si256(
        _mm256_cmpeq_epi8(q_exceeds, _mm256_setzero_si256()),
        _mm256_set1_epi8(-1),
    );

    _mm256_or_si256(p_hev, q_hev)
}

/// Signed right shift by 1 for packed bytes using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn signed_shift_right_1_avx2(_token: X64V3Token, v: __m256i) -> __m256i {
    let lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(v, v), 9);
    let hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(v, v), 9);
    _mm256_packs_epi16(lo, hi)
}

/// Apply the subblock/inner filter (DoFilter4) for 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn do_filter4_32(
    _token: X64V3Token,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    mask: __m256i,
    hev: __m256i,
) {
    let sign = _mm256_set1_epi8(-128i8);

    // Convert to signed
    let p1s = _mm256_xor_si256(*p1, sign);
    let p0s = _mm256_xor_si256(*p0, sign);
    let q0s = _mm256_xor_si256(*q0, sign);
    let q1s = _mm256_xor_si256(*q1, sign);

    // Compute base filter value
    let outer = _mm256_subs_epi8(p1s, q1s);
    let outer_masked = _mm256_and_si256(outer, hev);

    let q0_p0 = _mm256_subs_epi8(q0s, p0s);

    // a = outer + 3*(q0 - p0)
    let a = _mm256_adds_epi8(outer_masked, q0_p0);
    let a = _mm256_adds_epi8(a, q0_p0);
    let a = _mm256_adds_epi8(a, q0_p0);

    // Apply mask
    let a = _mm256_and_si256(a, mask);

    // Compute filter1 = (a + 4) >> 3 and filter2 = (a + 3) >> 3
    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);

    let f1 = _mm256_adds_epi8(a, k4);
    let f2 = _mm256_adds_epi8(a, k3);

    let f1 = signed_shift_right_3_avx2(_token, f1);
    let f2 = signed_shift_right_3_avx2(_token, f2);

    // Update p0, q0
    let new_p0s = _mm256_adds_epi8(p0s, f2);
    let new_q0s = _mm256_subs_epi8(q0s, f1);

    // For !hev case, also update p1, q1
    let a2 = _mm256_adds_epi8(f1, _mm256_set1_epi8(1));
    let a2 = signed_shift_right_1_avx2(_token, a2);
    let a2 = _mm256_andnot_si256(hev, a2);
    let a2 = _mm256_and_si256(a2, mask);

    let new_p1s = _mm256_adds_epi8(p1s, a2);
    let new_q1s = _mm256_subs_epi8(q1s, a2);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(new_p0s, sign);
    *q0 = _mm256_xor_si256(new_q0s, sign);
    *p1 = _mm256_xor_si256(new_p1s, sign);
    *q1 = _mm256_xor_si256(new_q1s, sign);
}

/// Helper for filter6 wide path using AVX2 - processes 16 pixels in 16-bit precision.
#[cfg(target_arch = "x86_64")]
#[rite]
fn filter6_wide_half_avx2(
    _token: X64V3Token,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i, __m256i, __m256i) {
    // Sign extend to 16-bit
    let p2_16 = _mm256_srai_epi16(p2, 8);
    let p1_16 = _mm256_srai_epi16(p1, 8);
    let p0_16 = _mm256_srai_epi16(p0, 8);
    let q0_16 = _mm256_srai_epi16(q0, 8);
    let q1_16 = _mm256_srai_epi16(q1, 8);
    let q2_16 = _mm256_srai_epi16(q2, 8);

    // w = clamp(p1 - q1 + 3*(q0 - p0))
    let p1_q1 = _mm256_sub_epi16(p1_16, q1_16);
    let q0_p0 = _mm256_sub_epi16(q0_16, p0_16);
    let three_q0_p0 = _mm256_add_epi16(_mm256_add_epi16(q0_p0, q0_p0), q0_p0);
    let w = _mm256_add_epi16(p1_q1, three_q0_p0);
    let w = _mm256_max_epi16(
        _mm256_min_epi16(w, _mm256_set1_epi16(127)),
        _mm256_set1_epi16(-128),
    );

    // a0 = (27*w + 63) >> 7
    let k27 = _mm256_set1_epi16(27);
    let k18 = _mm256_set1_epi16(18);
    let k9 = _mm256_set1_epi16(9);
    let k63 = _mm256_set1_epi16(63);

    let a0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k27), k63), 7);
    let a1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k18), k63), 7);
    let a2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k9), k63), 7);

    // Apply adjustments
    let new_p0 = _mm256_add_epi16(p0_16, a0);
    let new_q0 = _mm256_sub_epi16(q0_16, a0);
    let new_p1 = _mm256_add_epi16(p1_16, a1);
    let new_q1 = _mm256_sub_epi16(q1_16, a1);
    let new_p2 = _mm256_add_epi16(p2_16, a2);
    let new_q2 = _mm256_sub_epi16(q2_16, a2);

    // Clamp to [-128, 127]
    let clamp = |v: __m256i| {
        _mm256_max_epi16(
            _mm256_min_epi16(v, _mm256_set1_epi16(127)),
            _mm256_set1_epi16(-128),
        )
    };

    (
        clamp(new_p2),
        clamp(new_p1),
        clamp(new_p0),
        clamp(new_q0),
        clamp(new_q1),
        clamp(new_q2),
    )
}

/// Apply the macroblock/outer filter (DoFilter6) for 32 pixels using AVX2.
#[cfg(target_arch = "x86_64")]
#[rite]
#[allow(clippy::too_many_arguments)]
fn do_filter6_32(
    _token: X64V3Token,
    p2: &mut __m256i,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    q2: &mut __m256i,
    mask: __m256i,
    hev: __m256i,
) {
    let sign = _mm256_set1_epi8(-128i8);
    let not_hev = _mm256_andnot_si256(hev, _mm256_set1_epi8(-1));

    // Convert to signed
    let p2s = _mm256_xor_si256(*p2, sign);
    let p1s = _mm256_xor_si256(*p1, sign);
    let p0s = _mm256_xor_si256(*p0, sign);
    let q0s = _mm256_xor_si256(*q0, sign);
    let q1s = _mm256_xor_si256(*q1, sign);
    let q2s = _mm256_xor_si256(*q2, sign);

    // For hev path: same as simple filter
    let outer = _mm256_subs_epi8(p1s, q1s);
    let outer_hev = _mm256_and_si256(outer, hev);

    let q0_p0 = _mm256_subs_epi8(q0s, p0s);
    let a_hev = _mm256_adds_epi8(outer_hev, q0_p0);
    let a_hev = _mm256_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm256_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm256_and_si256(a_hev, _mm256_and_si256(mask, hev));

    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);
    let f1_hev = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(a_hev, k4));
    let f2_hev = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(a_hev, k3));

    // For !hev path: wide filter using 16-bit precision
    // Process low and high halves separately
    let (new_p2_lo, new_p1_lo, new_p0_lo, new_q0_lo, new_q1_lo, new_q2_lo) = filter6_wide_half_avx2(
        _token,
        _mm256_unpacklo_epi8(p2s, p2s),
        _mm256_unpacklo_epi8(p1s, p1s),
        _mm256_unpacklo_epi8(p0s, p0s),
        _mm256_unpacklo_epi8(q0s, q0s),
        _mm256_unpacklo_epi8(q1s, q1s),
        _mm256_unpacklo_epi8(q2s, q2s),
    );

    let (new_p2_hi, new_p1_hi, new_p0_hi, new_q0_hi, new_q1_hi, new_q2_hi) = filter6_wide_half_avx2(
        _token,
        _mm256_unpackhi_epi8(p2s, p2s),
        _mm256_unpackhi_epi8(p1s, p1s),
        _mm256_unpackhi_epi8(p0s, p0s),
        _mm256_unpackhi_epi8(q0s, q0s),
        _mm256_unpackhi_epi8(q1s, q1s),
        _mm256_unpackhi_epi8(q2s, q2s),
    );

    // Pack back to bytes
    let new_p2_wide = _mm256_packs_epi16(new_p2_lo, new_p2_hi);
    let new_p1_wide = _mm256_packs_epi16(new_p1_lo, new_p1_hi);
    let new_p0_wide = _mm256_packs_epi16(new_p0_lo, new_p0_hi);
    let new_q0_wide = _mm256_packs_epi16(new_q0_lo, new_q0_hi);
    let new_q1_wide = _mm256_packs_epi16(new_q1_lo, new_q1_hi);
    let new_q2_wide = _mm256_packs_epi16(new_q2_lo, new_q2_hi);

    // Blend hev and !hev results
    let mask_not_hev = _mm256_and_si256(mask, not_hev);

    // For p0, q0: use hev result where hev, wide result where !hev
    let new_p0s = _mm256_adds_epi8(p0s, f2_hev);
    let new_q0s = _mm256_subs_epi8(q0s, f1_hev);

    // Blend using blendv
    let final_p0s = _mm256_blendv_epi8(new_p0s, new_p0_wide, mask_not_hev);
    let final_q0s = _mm256_blendv_epi8(new_q0s, new_q0_wide, mask_not_hev);

    // For p1, q1, p2, q2: only update when !hev
    let final_p1s = _mm256_blendv_epi8(p1s, new_p1_wide, mask_not_hev);
    let final_q1s = _mm256_blendv_epi8(q1s, new_q1_wide, mask_not_hev);
    let final_p2s = _mm256_blendv_epi8(p2s, new_p2_wide, mask_not_hev);
    let final_q2s = _mm256_blendv_epi8(q2s, new_q2_wide, mask_not_hev);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(final_p0s, sign);
    *q0 = _mm256_xor_si256(final_q0s, sign);
    *p1 = _mm256_xor_si256(final_p1s, sign);
    *q1 = _mm256_xor_si256(final_q1s, sign);
    *p2 = _mm256_xor_si256(final_p2s, sign);
    *q2 = _mm256_xor_si256(final_q2s, sign);
}

/// Apply normal vertical filter (DoFilter4) to 32 pixels across a horizontal edge.
/// This is for subblock edges within a macroblock - 2x throughput vs SSE2.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter32_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION_32] =
        <&mut [u8; V_FILTER_NORMAL_REGION_32]>::try_from(
            &mut pixels[start..start + V_FILTER_NORMAL_REGION_32],
        )
        .expect("normal_v_filter32_inner: buffer too small");

    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 32 pixels each
    let p3 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p3..][..32]).unwrap());
    let p2 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p2..][..32]).unwrap());
    let mut p1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let mut q1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());
    let q2 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q2..][..32]).unwrap());
    let q3 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q3..][..32]).unwrap());

    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    do_filter4_32(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p1..][..32]).unwrap(),
        p1,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q1..][..32]).unwrap(),
        q1,
    );
}

/// Apply normal vertical filter (DoFilter6) to 32 pixels across a horizontal macroblock edge.
/// 2x throughput vs SSE2 version.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter32_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION_32] =
        <&mut [u8; V_FILTER_NORMAL_REGION_32]>::try_from(
            &mut pixels[start..start + V_FILTER_NORMAL_REGION_32],
        )
        .expect("normal_v_filter32_edge: buffer too small");

    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 32 pixels each
    let p3 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p3..][..32]).unwrap());
    let mut p2 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p2..][..32]).unwrap());
    let mut p1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let mut q1 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());
    let mut q2 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q2..][..32]).unwrap());
    let q3 =
        simd_mem_x86::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q3..][..32]).unwrap());

    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    do_filter6_32(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p2..][..32]).unwrap(),
        p2,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p1..][..32]).unwrap(),
        p1,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q1..][..32]).unwrap(),
        q1,
    );
    simd_mem_x86::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q2..][..32]).unwrap(),
        q2,
    );
}

// ============================================================================
// Load16x4 / Store16x4 — libwebp's approach for horizontal filters.
//
// Instead of loading 8 bytes per row and doing a full 8×16→16×8 transpose,
// load only 4 bytes per row and use a lighter 4-column transpose.
// This halves memory reads for the simple filter and reduces transpose ops
// from 32 to 18 for the 4-column case.
//
// All helpers are fully inlined (load_8x4 logic is fused into each caller)
// to avoid function-call overhead that prevents the compiler from inlining
// #[rite] functions into the #[arcane] filter entry points.
// ============================================================================

/// Load 8 rows of 4 bytes each from `pixels` starting at `base` with `stride`.
/// Transposes into 2 packed registers: (cols01, cols23).
///
/// Mirrors libwebp's `Load8x4_SSE2`.
macro_rules! load_8x4_impl {
    ($pixels:expr, $base:expr, $stride:expr) => {{
        let base_ = $base;
        let stride_ = $stride;
        let pixels_: &[u8] = &*$pixels;
        let r0 = i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_..][..4]).unwrap());
        let r1 = i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + stride_..][..4]).unwrap());
        let r2 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 2 * stride_..][..4]).unwrap());
        let r3 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 3 * stride_..][..4]).unwrap());
        let r4 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 4 * stride_..][..4]).unwrap());
        let r5 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 5 * stride_..][..4]).unwrap());
        let r6 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 6 * stride_..][..4]).unwrap());
        let r7 =
            i32::from_ne_bytes(<[u8; 4]>::try_from(&pixels_[base_ + 7 * stride_..][..4]).unwrap());

        let a0 = _mm_set_epi32(r6, r2, r4, r0);
        let a1 = _mm_set_epi32(r7, r3, r5, r1);

        let b0 = _mm_unpacklo_epi8(a0, a1);
        let b1 = _mm_unpackhi_epi8(a0, a1);
        let c0 = _mm_unpacklo_epi16(b0, b1);
        let c1 = _mm_unpackhi_epi16(b0, b1);

        (_mm_unpacklo_epi32(c0, c1), _mm_unpackhi_epi32(c0, c1))
    }};
}

/// Store 4 rows of 4 bytes each from a single __m128i register.
macro_rules! store_4x4_impl {
    ($pixels:expr, $base:expr, $stride:expr, $vals:expr) => {{
        let base_ = $base;
        let stride_ = $stride;
        let vals_ = $vals;
        let p_ = &mut *$pixels;
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut p_[base_..][..4]).unwrap(),
            vals_,
        );
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut p_[base_ + stride_..][..4]).unwrap(),
            _mm_srli_si128(vals_, 4),
        );
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut p_[base_ + 2 * stride_..][..4]).unwrap(),
            _mm_srli_si128(vals_, 8),
        );
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut p_[base_ + 3 * stride_..][..4]).unwrap(),
            _mm_srli_si128(vals_, 12),
        );
    }};
}

/// Load 4 bytes from each of 16 rows, producing 4 column __m128i values.
/// Mirrors libwebp's `Load16x4_SSE2`.
macro_rules! load_16x4_impl {
    ($pixels:expr, $base:expr, $stride:expr) => {{
        let (pq_lo_a, pq_hi_a) = load_8x4_impl!($pixels, $base, $stride);
        let (pq_lo_b, pq_hi_b) = load_8x4_impl!($pixels, $base + 8 * $stride, $stride);
        (
            _mm_unpacklo_epi64(pq_lo_a, pq_lo_b),
            _mm_unpackhi_epi64(pq_lo_a, pq_lo_b),
            _mm_unpacklo_epi64(pq_hi_a, pq_hi_b),
            _mm_unpackhi_epi64(pq_hi_a, pq_hi_b),
        )
    }};
}

/// Transpose 4 columns and store as 16 rows of 4 bytes.
/// Mirrors libwebp's `Store16x4_SSE2`.
macro_rules! store_16x4_impl {
    ($pixels:expr, $base:expr, $stride:expr, $col0:expr, $col1:expr, $col2:expr, $col3:expr) => {{
        let t0_ = _mm_unpacklo_epi8($col0, $col1);
        let t1_ = _mm_unpackhi_epi8($col0, $col1);
        let t2_ = _mm_unpacklo_epi8($col2, $col3);
        let t3_ = _mm_unpackhi_epi8($col2, $col3);
        let r0_ = _mm_unpacklo_epi16(t0_, t2_);
        let r1_ = _mm_unpackhi_epi16(t0_, t2_);
        let r2_ = _mm_unpacklo_epi16(t1_, t3_);
        let r3_ = _mm_unpackhi_epi16(t1_, t3_);
        store_4x4_impl!($pixels, $base, $stride, r0_);
        store_4x4_impl!($pixels, $base + 4 * $stride, $stride, r1_);
        store_4x4_impl!($pixels, $base + 8 * $stride, $stride, r2_);
        store_4x4_impl!($pixels, $base + 12 * $stride, $stride, r3_);
    }};
}

/// Store q1,q2 as 2-byte rows for 16 rows.
macro_rules! store_q1q2_16_impl {
    ($pixels:expr, $base:expr, $stride:expr, $q1:expr, $q2:expr) => {{
        let t0_ = _mm_unpacklo_epi8($q1, $q2);
        let t1_ = _mm_unpackhi_epi8($q1, $q2);
        macro_rules! _sq {
            ($r:expr, $l:expr, $row:expr) => {
                let val_ = _mm_extract_epi16($r, $l) as u16;
                let off_ = $base + $row * $stride;
                $pixels[off_] = val_ as u8;
                $pixels[off_ + 1] = (val_ >> 8) as u8;
            };
        }
        _sq!(t0_, 0, 0);
        _sq!(t0_, 1, 1);
        _sq!(t0_, 2, 2);
        _sq!(t0_, 3, 3);
        _sq!(t0_, 4, 4);
        _sq!(t0_, 5, 5);
        _sq!(t0_, 6, 6);
        _sq!(t0_, 7, 7);
        _sq!(t1_, 0, 8);
        _sq!(t1_, 1, 9);
        _sq!(t1_, 2, 10);
        _sq!(t1_, 3, 11);
        _sq!(t1_, 4, 12);
        _sq!(t1_, 5, 13);
        _sq!(t1_, 6, 14);
        _sq!(t1_, 7, 15);
    }};
}

/// Store 4 columns as 8 U rows and 8 V rows of 4 bytes each.
macro_rules! store_uv_16x4_impl {
    ($u:expr, $v:expr, $base:expr, $stride:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr) => {{
        let t0_ = _mm_unpacklo_epi8($c0, $c1);
        let t1_ = _mm_unpackhi_epi8($c0, $c1);
        let t2_ = _mm_unpacklo_epi8($c2, $c3);
        let t3_ = _mm_unpackhi_epi8($c2, $c3);
        let r0_ = _mm_unpacklo_epi16(t0_, t2_);
        let r1_ = _mm_unpackhi_epi16(t0_, t2_);
        let r2_ = _mm_unpacklo_epi16(t1_, t3_);
        let r3_ = _mm_unpackhi_epi16(t1_, t3_);
        store_4x4_impl!($u, $base, $stride, r0_);
        store_4x4_impl!($u, $base + 4 * $stride, $stride, r1_);
        store_4x4_impl!($v, $base, $stride, r2_);
        store_4x4_impl!($v, $base + 4 * $stride, $stride, r3_);
    }};
}

/// Store q1,q2 as 2-byte rows for 8 U + 8 V rows.
macro_rules! store_q1q2_uv_16_impl {
    ($u:expr, $v:expr, $base:expr, $stride:expr, $q1:expr, $q2:expr) => {{
        let t0_ = _mm_unpacklo_epi8($q1, $q2); // U rows 0-7
        let t1_ = _mm_unpackhi_epi8($q1, $q2); // V rows 0-7
        macro_rules! _sq {
            ($r:expr, $l:expr, $buf:expr, $row:expr) => {
                let val_ = _mm_extract_epi16($r, $l) as u16;
                let off_ = $base + $row * $stride;
                $buf[off_] = val_ as u8;
                $buf[off_ + 1] = (val_ >> 8) as u8;
            };
        }
        _sq!(t0_, 0, $u, 0);
        _sq!(t0_, 1, $u, 1);
        _sq!(t0_, 2, $u, 2);
        _sq!(t0_, 3, $u, 3);
        _sq!(t0_, 4, $u, 4);
        _sq!(t0_, 5, $u, 5);
        _sq!(t0_, 6, $u, 6);
        _sq!(t0_, 7, $u, 7);
        _sq!(t1_, 0, $v, 0);
        _sq!(t1_, 1, $v, 1);
        _sq!(t1_, 2, $v, 2);
        _sq!(t1_, 3, $v, 3);
        _sq!(t1_, 4, $v, 4);
        _sq!(t1_, 5, $v, 5);
        _sq!(t1_, 6, $v, 6);
        _sq!(t1_, 7, $v, 7);
    }};
}

/// Transpose an 8x16 matrix of bytes to 16x8.
/// Input: 16 __m128i values, each containing 8 bytes (low 64 bits used).
/// Output: 8 __m128i values, each containing 16 bytes.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transpose_8x16_to_16x8(_token: X64V3Token, rows: &[__m128i; 16]) -> [__m128i; 8] {
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
#[rite]
fn transpose_4x16_to_16x4(
    _token: X64V3Token,
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
/// Uses Load16x4/Store16x4 (libwebp's approach): load only 4 bytes per row,
/// transpose 4 columns with a lightweight 3-stage unpack, filter, store back.
///
/// Each row must have at least 2 bytes before and 2 bytes after the edge point.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_h_filter16(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x - 2;
    let region: &mut [u8; H_FILTER_SIMPLE_REGION] = <&mut [u8; H_FILTER_SIMPLE_REGION]>::try_from(
        &mut pixels[base..base + H_FILTER_SIMPLE_REGION],
    )
    .expect("simple_h_filter16: buffer too small (missing FILTER_PADDING?)");

    // Load 4 bytes per row (p1,p0,q0,q1) from 16 rows, transpose into 4 columns
    let (p1, p0, q0, q1) = load_16x4_impl!(region, 0, stride);

    let mut p0 = p0;
    let mut q0 = q0;

    // Apply simple filter (same logic as vertical, but on transposed data)
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Transpose back and store 4 bytes per row
    store_16x4_impl!(region, 0, stride, p1, p0, q0, q1);
}

/// Apply simple vertical filter to entire macroblock edge (16 pixels).
/// This is the main entry point for filtering horizontal edges between macroblocks.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_filter_mb_edge_v(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let point = mb_y * 16 * stride + mb_x * 16;
    simple_v_filter16(_token, pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to entire macroblock edge (16 rows).
/// This is the main entry point for filtering vertical edges between macroblocks.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_filter_mb_edge_h(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16;
    let y_start = mb_y * 16;
    simple_h_filter16(_token, pixels, x, y_start, stride, thresh);
}

/// Apply simple vertical filter to a subblock edge within a macroblock.
/// y_offset is the row offset within the macroblock (4, 8, or 12).
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_filter_subblock_edge_v(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    y_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let point = (mb_y * 16 + y_offset) * stride + mb_x * 16;
    simple_v_filter16(_token, pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to a subblock edge within a macroblock.
/// x_offset is the column offset within the macroblock (4, 8, or 12).
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn simple_filter_subblock_edge_h(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    x_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16 + x_offset;
    let y_start = mb_y * 16;
    simple_h_filter16(_token, pixels, x, y_start, stride, thresh);
}

// =============================================================================
// Normal filter implementation (DoFilter4 equivalent from libwebp)
// =============================================================================

/// Check if pixels need filtering using the full normal filter threshold.
/// Condition: simple_threshold AND all interior differences <= interior_limit
#[cfg(target_arch = "x86_64")]
#[rite]
#[allow(clippy::too_many_arguments)]
fn needs_filter_normal_16(
    _token: X64V3Token,
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    edge_limit: i32,
    interior_limit: i32,
) -> __m128i {
    // First check simple threshold
    let simple_mask = needs_filter_16(_token, p1, p0, q0, q1, edge_limit);

    let i_limit = _mm_set1_epi8(interior_limit as i8);

    // Check interior differences: all must be <= interior_limit
    // |p3-p2|, |p2-p1|, |p1-p0|, |q0-q1|, |q1-q2|, |q2-q3|

    // Helper macro for abs diff
    macro_rules! abs_diff {
        ($a:expr, $b:expr) => {
            _mm_or_si128(_mm_subs_epu8($a, $b), _mm_subs_epu8($b, $a))
        };
    }

    let d_p3_p2 = abs_diff!(p3, p2);
    let d_p2_p1 = abs_diff!(p2, p1);
    let d_p1_p0 = abs_diff!(p1, p0);
    let d_q0_q1 = abs_diff!(q0, q1);
    let d_q1_q2 = abs_diff!(q1, q2);
    let d_q2_q3 = abs_diff!(q2, q3);

    // Take max of all differences
    let max1 = _mm_max_epu8(d_p3_p2, d_p2_p1);
    let max2 = _mm_max_epu8(d_p1_p0, d_q0_q1);
    let max3 = _mm_max_epu8(d_q1_q2, d_q2_q3);
    let max4 = _mm_max_epu8(max1, max2);
    let max_diff = _mm_max_epu8(max3, max4);

    // Check if max_diff <= interior_limit
    let exceeds = _mm_subs_epu8(max_diff, i_limit);
    let interior_ok = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());

    // Both conditions must be true
    _mm_and_si128(simple_mask, interior_ok)
}

/// Check high edge variance: |p1 - p0| > thresh OR |q1 - q0| > thresh
#[cfg(target_arch = "x86_64")]
#[rite]
fn high_edge_variance_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    hev_thresh: i32,
) -> __m128i {
    let t = _mm_set1_epi8(hev_thresh as i8);

    // |p1 - p0|
    let d_p1_p0 = _mm_or_si128(_mm_subs_epu8(p1, p0), _mm_subs_epu8(p0, p1));

    // |q1 - q0|
    let d_q1_q0 = _mm_or_si128(_mm_subs_epu8(q1, q0), _mm_subs_epu8(q0, q1));

    // Check if either > thresh
    // subs_epu8 saturates at 0, so d > t means d - t > 0
    let p_exceeds = _mm_subs_epu8(d_p1_p0, t);
    let q_exceeds = _mm_subs_epu8(d_q1_q0, t);

    // hev = true if exceeds > 0 (i.e., not equal to zero)
    let p_hev = _mm_xor_si128(
        _mm_cmpeq_epi8(p_exceeds, _mm_setzero_si128()),
        _mm_set1_epi8(-1),
    );
    let q_hev = _mm_xor_si128(
        _mm_cmpeq_epi8(q_exceeds, _mm_setzero_si128()),
        _mm_set1_epi8(-1),
    );

    _mm_or_si128(p_hev, q_hev)
}

/// Apply the subblock/inner filter (DoFilter4 from libwebp).
/// When hev=true: only modify p0, q0 (use outer taps)
/// When hev=false: modify p1, p0, q0, q1
#[cfg(target_arch = "x86_64")]
#[rite]
fn do_filter4_16(
    _token: X64V3Token,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    mask: __m128i,
    hev: __m128i,
) {
    let sign = _mm_set1_epi8(-128i8);

    // Convert to signed
    let p1s = _mm_xor_si128(*p1, sign);
    let p0s = _mm_xor_si128(*p0, sign);
    let q0s = _mm_xor_si128(*q0, sign);
    let q1s = _mm_xor_si128(*q1, sign);

    // Compute base filter value
    // When hev: use outer taps (p1 - q1)
    // When !hev: no outer taps
    let outer = _mm_subs_epi8(p1s, q1s);
    let outer_masked = _mm_and_si128(outer, hev); // Only use outer when hev

    // q0 - p0 (will multiply by 3)
    let q0_p0 = _mm_subs_epi8(q0s, p0s);

    // a = outer + 3*(q0 - p0)
    let a = _mm_adds_epi8(outer_masked, q0_p0);
    let a = _mm_adds_epi8(a, q0_p0);
    let a = _mm_adds_epi8(a, q0_p0);

    // Apply mask
    let a = _mm_and_si128(a, mask);

    // Compute filter1 = (a + 4) >> 3 and filter2 = (a + 3) >> 3
    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);

    let f1 = _mm_adds_epi8(a, k4);
    let f2 = _mm_adds_epi8(a, k3);

    let f1 = signed_shift_right_3(_token, f1);
    let f2 = signed_shift_right_3(_token, f2);

    // Update p0, q0
    let new_p0s = _mm_adds_epi8(p0s, f2);
    let new_q0s = _mm_subs_epi8(q0s, f1);

    // For !hev case, also update p1, q1
    // a2 = (f1 + 1) >> 1 -- spread the filter to outer pixels
    let a2 = _mm_adds_epi8(f1, _mm_set1_epi8(1));
    let a2 = signed_shift_right_1(_token, a2);
    let a2 = _mm_andnot_si128(hev, a2); // Only when !hev
    let a2 = _mm_and_si128(a2, mask); // Only when filtered

    let new_p1s = _mm_adds_epi8(p1s, a2);
    let new_q1s = _mm_subs_epi8(q1s, a2);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(new_p0s, sign);
    *q0 = _mm_xor_si128(new_q0s, sign);
    *p1 = _mm_xor_si128(new_p1s, sign);
    *q1 = _mm_xor_si128(new_q1s, sign);
}

/// Signed right shift by 1 for packed bytes
#[cfg(target_arch = "x86_64")]
#[rite]
fn signed_shift_right_1(_token: X64V3Token, v: __m128i) -> __m128i {
    let lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 9);
    let hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 9);
    _mm_packs_epi16(lo, hi)
}

/// Apply the macroblock/outer filter (DoFilter6 from libwebp).
/// When hev=true: only modify p0, q0 (use outer taps)
/// When hev=false: modify p2, p1, p0, q0, q1, q2 with weighted filter
#[cfg(target_arch = "x86_64")]
#[rite]
#[allow(clippy::too_many_arguments)]
fn do_filter6_16(
    _token: X64V3Token,
    p2: &mut __m128i,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    q2: &mut __m128i,
    mask: __m128i,
    hev: __m128i,
) {
    let sign = _mm_set1_epi8(-128i8);
    let not_hev = _mm_andnot_si128(hev, _mm_set1_epi8(-1));

    // Convert to signed
    let p2s = _mm_xor_si128(*p2, sign);
    let p1s = _mm_xor_si128(*p1, sign);
    let p0s = _mm_xor_si128(*p0, sign);
    let q0s = _mm_xor_si128(*q0, sign);
    let q1s = _mm_xor_si128(*q1, sign);
    let q2s = _mm_xor_si128(*q2, sign);

    // For hev path: same as simple filter
    let outer = _mm_subs_epi8(p1s, q1s);
    let outer_hev = _mm_and_si128(outer, hev);

    let q0_p0 = _mm_subs_epi8(q0s, p0s);
    let a_hev = _mm_adds_epi8(outer_hev, q0_p0);
    let a_hev = _mm_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm_and_si128(a_hev, _mm_and_si128(mask, hev));

    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);
    let f1_hev = signed_shift_right_3(_token, _mm_adds_epi8(a_hev, k4));
    let f2_hev = signed_shift_right_3(_token, _mm_adds_epi8(a_hev, k3));

    // For !hev path: wide filter using 16-bit precision
    // w = clamp(p1 - q1 + 3*(q0 - p0))
    // a0 = (27*w + 63) >> 7  (applied to p0, q0)
    // a1 = (18*w + 63) >> 7  (applied to p1, q1)
    // a2 = (9*w + 63) >> 7   (applied to p2, q2)

    // We need 16-bit precision for the multiply
    // Process low and high halves separately
    let (new_p2_lo, new_p1_lo, new_p0_lo, new_q0_lo, new_q1_lo, new_q2_lo) = filter6_wide_half(
        _token,
        _mm_unpacklo_epi8(p2s, p2s),
        _mm_unpacklo_epi8(p1s, p1s),
        _mm_unpacklo_epi8(p0s, p0s),
        _mm_unpacklo_epi8(q0s, q0s),
        _mm_unpacklo_epi8(q1s, q1s),
        _mm_unpacklo_epi8(q2s, q2s),
    );

    let (new_p2_hi, new_p1_hi, new_p0_hi, new_q0_hi, new_q1_hi, new_q2_hi) = filter6_wide_half(
        _token,
        _mm_unpackhi_epi8(p2s, p2s),
        _mm_unpackhi_epi8(p1s, p1s),
        _mm_unpackhi_epi8(p0s, p0s),
        _mm_unpackhi_epi8(q0s, q0s),
        _mm_unpackhi_epi8(q1s, q1s),
        _mm_unpackhi_epi8(q2s, q2s),
    );

    // Pack back to bytes
    let new_p2_wide = _mm_packs_epi16(new_p2_lo, new_p2_hi);
    let new_p1_wide = _mm_packs_epi16(new_p1_lo, new_p1_hi);
    let new_p0_wide = _mm_packs_epi16(new_p0_lo, new_p0_hi);
    let new_q0_wide = _mm_packs_epi16(new_q0_lo, new_q0_hi);
    let new_q1_wide = _mm_packs_epi16(new_q1_lo, new_q1_hi);
    let new_q2_wide = _mm_packs_epi16(new_q2_lo, new_q2_hi);

    // Blend hev and !hev results
    let mask_not_hev = _mm_and_si128(mask, not_hev);

    // For p0, q0: use hev result where hev, wide result where !hev
    let new_p0s = _mm_adds_epi8(p0s, f2_hev); // hev path
    let new_q0s = _mm_subs_epi8(q0s, f1_hev);

    // Blend: select wide where !hev, hev result where hev (and filtered)
    let final_p0s = _mm_blendv_epi8(new_p0s, new_p0_wide, mask_not_hev);
    let final_q0s = _mm_blendv_epi8(new_q0s, new_q0_wide, mask_not_hev);

    // For p1, q1, p2, q2: only update when !hev
    let final_p1s = _mm_blendv_epi8(p1s, new_p1_wide, mask_not_hev);
    let final_q1s = _mm_blendv_epi8(q1s, new_q1_wide, mask_not_hev);
    let final_p2s = _mm_blendv_epi8(p2s, new_p2_wide, mask_not_hev);
    let final_q2s = _mm_blendv_epi8(q2s, new_q2_wide, mask_not_hev);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(final_p0s, sign);
    *q0 = _mm_xor_si128(final_q0s, sign);
    *p1 = _mm_xor_si128(final_p1s, sign);
    *q1 = _mm_xor_si128(final_q1s, sign);
    *p2 = _mm_xor_si128(final_p2s, sign);
    *q2 = _mm_xor_si128(final_q2s, sign);
}

/// Helper for filter6 wide path - processes 8 pixels in 16-bit precision
#[cfg(target_arch = "x86_64")]
#[rite]
fn filter6_wide_half(
    _token: X64V3Token,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // Sign extend to 16-bit (values are duplicated, take every other)
    let p2_16 = _mm_srai_epi16(p2, 8);
    let p1_16 = _mm_srai_epi16(p1, 8);
    let p0_16 = _mm_srai_epi16(p0, 8);
    let q0_16 = _mm_srai_epi16(q0, 8);
    let q1_16 = _mm_srai_epi16(q1, 8);
    let q2_16 = _mm_srai_epi16(q2, 8);

    // w = clamp(p1 - q1 + 3*(q0 - p0))
    let p1_q1 = _mm_sub_epi16(p1_16, q1_16);
    let q0_p0 = _mm_sub_epi16(q0_16, p0_16);
    let three_q0_p0 = _mm_add_epi16(_mm_add_epi16(q0_p0, q0_p0), q0_p0);
    let w = _mm_add_epi16(p1_q1, three_q0_p0);
    let w = _mm_max_epi16(_mm_min_epi16(w, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    // a0 = (27*w + 63) >> 7
    let k27 = _mm_set1_epi16(27);
    let k18 = _mm_set1_epi16(18);
    let k9 = _mm_set1_epi16(9);
    let k63 = _mm_set1_epi16(63);

    let a0 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k27), k63), 7);
    let a1 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k18), k63), 7);
    let a2 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k9), k63), 7);

    // Apply adjustments
    let new_p0 = _mm_add_epi16(p0_16, a0);
    let new_q0 = _mm_sub_epi16(q0_16, a0);
    let new_p1 = _mm_add_epi16(p1_16, a1);
    let new_q1 = _mm_sub_epi16(q1_16, a1);
    let new_p2 = _mm_add_epi16(p2_16, a2);
    let new_q2 = _mm_sub_epi16(q2_16, a2);

    // Clamp to [-128, 127]
    let clamp =
        |v: __m128i| _mm_max_epi16(_mm_min_epi16(v, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    (
        clamp(new_p2),
        clamp(new_p1),
        clamp(new_p0),
        clamp(new_q0),
        clamp(new_q1),
        clamp(new_q2),
    )
}

/// Apply normal vertical filter (DoFilter4) to 16 pixels across a horizontal edge.
/// This is for subblock edges within a macroblock.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter16_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION] = <&mut [u8; V_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[start..start + V_FILTER_NORMAL_REGION],
    )
    .expect("normal_v_filter16_inner: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region - compiler proves these are in-bounds
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 16 pixels each - NO per-access bounds checks
    let p3 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p3..][..16]).unwrap());
    let p2 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p2..][..16]).unwrap());
    let mut p1 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let mut q1 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());
    let q2 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q2..][..16]).unwrap());
    let q3 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q3..][..16]).unwrap());

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store results - NO per-access bounds checks
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p1..][..16]).unwrap(),
        p1,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q1..][..16]).unwrap(),
        q1,
    );
}

/// Apply normal vertical filter (DoFilter6) to 16 pixels across a horizontal macroblock edge.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter16_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION] = <&mut [u8; V_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[start..start + V_FILTER_NORMAL_REGION],
    )
    .expect("normal_v_filter16_edge: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 16 pixels each - NO per-access bounds checks
    let p3 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p3..][..16]).unwrap());
    let mut p2 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p2..][..16]).unwrap());
    let mut p1 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let mut q1 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());
    let mut q2 =
        simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q2..][..16]).unwrap());
    let q3 = simd_mem_x86::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q3..][..16]).unwrap());

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store results - NO per-access bounds checks
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p2..][..16]).unwrap(),
        p2,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p1..][..16]).unwrap(),
        p1,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q1..][..16]).unwrap(),
        q1,
    );
    simd_mem_x86::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q2..][..16]).unwrap(),
        q2,
    );
}

/// Transpose 6 columns (p2, p1, p0, q0, q1, q2) of 16 bytes each back to 16 rows of 6 bytes.
/// Returns values suitable for storing back to memory.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transpose_6x16_to_16x6(
    _token: X64V3Token,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
) -> ([i32; 16], [i16; 16]) {
    // We need to return 6 bytes per row: p2, p1, p0, q0, q1, q2
    // We'll return two arrays: one with 4-byte values (p2,p1,p0,q0) and one with 2-byte values (q1,q2)

    // Interleave pairs
    let p2p1_lo = _mm_unpacklo_epi8(p2, p1); // rows 0-7: p2[i], p1[i]
    let p2p1_hi = _mm_unpackhi_epi8(p2, p1); // rows 8-15
    let p0q0_lo = _mm_unpacklo_epi8(p0, q0);
    let p0q0_hi = _mm_unpackhi_epi8(p0, q0);
    let q1q2_lo = _mm_unpacklo_epi8(q1, q2);
    let q1q2_hi = _mm_unpackhi_epi8(q1, q2);

    // Combine p2p1 and p0q0 to 32-bit
    let r0_lo = _mm_unpacklo_epi16(p2p1_lo, p0q0_lo); // rows 0-3
    let r1_lo = _mm_unpackhi_epi16(p2p1_lo, p0q0_lo); // rows 4-7
    let r0_hi = _mm_unpacklo_epi16(p2p1_hi, p0q0_hi); // rows 8-11
    let r1_hi = _mm_unpackhi_epi16(p2p1_hi, p0q0_hi); // rows 12-15

    // Extract 32-bit values for p2,p1,p0,q0
    let mut result4 = [0i32; 16];
    result4[0] = _mm_extract_epi32(r0_lo, 0);
    result4[1] = _mm_extract_epi32(r0_lo, 1);
    result4[2] = _mm_extract_epi32(r0_lo, 2);
    result4[3] = _mm_extract_epi32(r0_lo, 3);
    result4[4] = _mm_extract_epi32(r1_lo, 0);
    result4[5] = _mm_extract_epi32(r1_lo, 1);
    result4[6] = _mm_extract_epi32(r1_lo, 2);
    result4[7] = _mm_extract_epi32(r1_lo, 3);
    result4[8] = _mm_extract_epi32(r0_hi, 0);
    result4[9] = _mm_extract_epi32(r0_hi, 1);
    result4[10] = _mm_extract_epi32(r0_hi, 2);
    result4[11] = _mm_extract_epi32(r0_hi, 3);
    result4[12] = _mm_extract_epi32(r1_hi, 0);
    result4[13] = _mm_extract_epi32(r1_hi, 1);
    result4[14] = _mm_extract_epi32(r1_hi, 2);
    result4[15] = _mm_extract_epi32(r1_hi, 3);

    // Extract 16-bit values for q1,q2
    let mut result2 = [0i16; 16];
    result2[0] = _mm_extract_epi16(q1q2_lo, 0) as i16;
    result2[1] = _mm_extract_epi16(q1q2_lo, 1) as i16;
    result2[2] = _mm_extract_epi16(q1q2_lo, 2) as i16;
    result2[3] = _mm_extract_epi16(q1q2_lo, 3) as i16;
    result2[4] = _mm_extract_epi16(q1q2_lo, 4) as i16;
    result2[5] = _mm_extract_epi16(q1q2_lo, 5) as i16;
    result2[6] = _mm_extract_epi16(q1q2_lo, 6) as i16;
    result2[7] = _mm_extract_epi16(q1q2_lo, 7) as i16;
    result2[8] = _mm_extract_epi16(q1q2_hi, 0) as i16;
    result2[9] = _mm_extract_epi16(q1q2_hi, 1) as i16;
    result2[10] = _mm_extract_epi16(q1q2_hi, 2) as i16;
    result2[11] = _mm_extract_epi16(q1q2_hi, 3) as i16;
    result2[12] = _mm_extract_epi16(q1q2_hi, 4) as i16;
    result2[13] = _mm_extract_epi16(q1q2_hi, 5) as i16;
    result2[14] = _mm_extract_epi16(q1q2_hi, 6) as i16;
    result2[15] = _mm_extract_epi16(q1q2_hi, 7) as i16;

    (result4, result2)
}

/// Apply normal horizontal filter (DoFilter4) to 16 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses 8-wide load + full transpose for all 8 columns, then Store16x4 for output.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[rite]
pub(crate) fn normal_h_filter16_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x - 4;
    let region: &mut [u8; H_FILTER_NORMAL_REGION] = <&mut [u8; H_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[base..base + H_FILTER_NORMAL_REGION],
    )
    .expect("normal_h_filter16_inner: buffer too small (missing FILTER_PADDING?)");

    // Load 16 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = i * stride;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&region[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store only the 4 modified columns via Store16x4
    // Region base is (y_start * stride + x - 4), store base is (y_start * stride + x - 2) = +2
    store_16x4_impl!(region, 2, stride, p1, p0, q0, q1);
}

/// Apply normal horizontal filter (DoFilter6) to 16 rows at a vertical macroblock edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses 8-wide load + full transpose for all 8 columns, then Store16x4+q1q2 for output.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[rite]
pub(crate) fn normal_h_filter16_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x - 4;
    let region: &mut [u8; H_FILTER_NORMAL_REGION] = <&mut [u8; H_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[base..base + H_FILTER_NORMAL_REGION],
    )
    .expect("normal_h_filter16_edge: buffer too small (missing FILTER_PADDING?)");

    // Load 16 rows of 8 pixels each
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = i * stride;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&region[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store 6 modified columns via Store16x4 + q1q2 store
    // Region base is (y_start * stride + x - 4), store base is (y_start * stride + x - 3) = +1
    store_16x4_impl!(region, 1, stride, p2, p1, p0, q0);
    store_q1q2_16_impl!(region, 5, stride, q1, q2);
}

/// Fused normal horizontal subblock filter for all 3 inner edges of a macroblock.
///
/// Matches libwebp's HFilter16i_SSE2: loads 4 columns at a time via Load16x4,
/// reuses columns between adjacent edges. Each edge only needs one new Load16x4
/// instead of a full 8-column load + 8x16 transpose.
///
/// Processes edges at x_start+4, x_start+8, x_start+12.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[rite]
pub(crate) fn normal_h_filter16i(
    _token: X64V3Token,
    pixels: &mut [u8],
    x_start: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x_start;
    let region: &mut [u8; H_FILTER_FUSED_REGION] = <&mut [u8; H_FILTER_FUSED_REGION]>::try_from(
        &mut pixels[base..base + H_FILTER_FUSED_REGION],
    )
    .expect("normal_h_filter16i: buffer too small (missing FILTER_PADDING?)");

    // Prologue: load first 4 columns (x_start..x_start+3 = p3, p2, p1, p0 for first edge)
    let (p3, p2, p1, p0) = load_16x4_impl!(region, 0, stride);
    let mut p3 = p3;
    let mut p2 = p2;
    let mut p1 = p1;
    let mut p0 = p0;

    for k in 0..3 {
        let edge_x = x_start + 4 + k * 4;

        // MAX_DIFF1: compute left-side interior diffs from current p3,p2,p1,p0
        let d_p1_p0 = _mm_or_si128(_mm_subs_epu8(p1, p0), _mm_subs_epu8(p0, p1));
        let d_p3_p2 = _mm_or_si128(_mm_subs_epu8(p3, p2), _mm_subs_epu8(p2, p3));
        let d_p2_p1 = _mm_or_si128(_mm_subs_epu8(p2, p1), _mm_subs_epu8(p1, p2));
        let mut max_diff = _mm_max_epu8(d_p1_p0, _mm_max_epu8(d_p3_p2, d_p2_p1));

        // Load next 4 columns (q0, q1, q2, q3 relative to this edge)
        // In libwebp: Load16x4 overwrites p3, p2, and puts extras in tmp1, tmp2
        let next_off = edge_x - x_start;
        let (new_p3, new_p2, tmp1, tmp2) = load_16x4_impl!(region, next_off, stride);
        // new_p3 = col at edge_x (= q0), new_p2 = col at edge_x+1 (= q1)
        // tmp1 = col at edge_x+2 (= q2), tmp2 = col at edge_x+3 (= q3)

        // MAX_DIFF2: add right-side interior diffs
        let d1 = _mm_or_si128(_mm_subs_epu8(new_p3, new_p2), _mm_subs_epu8(new_p2, new_p3));
        let d2 = _mm_or_si128(_mm_subs_epu8(tmp1, tmp2), _mm_subs_epu8(tmp2, tmp1));
        let d3 = _mm_or_si128(_mm_subs_epu8(new_p2, tmp1), _mm_subs_epu8(tmp1, new_p2));
        max_diff = _mm_max_epu8(max_diff, _mm_max_epu8(d1, _mm_max_epu8(d2, d3)));

        // ComplexMask: interior check + simple threshold
        // p1, p0 are the left pair; new_p3 (=q0), new_p2 (=q1) are the right pair
        let i_limit = _mm_set1_epi8(interior_limit as i8);
        let exceeds = _mm_subs_epu8(max_diff, i_limit);
        let interior_ok = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());
        let simple_mask = needs_filter_16(_token, p1, p0, new_p3, new_p2, edge_limit);
        let mask = _mm_and_si128(simple_mask, interior_ok);

        // HEV check
        let hev = high_edge_variance_16(_token, p1, p0, new_p3, new_p2, hev_thresh);

        // DoFilter4: modifies p1, p0, q0, q1 in place
        let mut fp1 = p1;
        let mut fp0 = p0;
        let mut fq0 = new_p3;
        let mut fq1 = new_p2;
        do_filter4_16(_token, &mut fp1, &mut fp0, &mut fq0, &mut fq1, mask, hev);

        // Store 4 modified columns: p1, p0, q0, q1 centered on the edge
        let store_off = edge_x - x_start - 2;
        store_16x4_impl!(region, store_off, stride, fp1, fp0, fq0, fq1);

        // Rotate for next edge (matching libwebp):
        // After filter: fq0 = filtered q0, fq1 = filtered q1
        // These become p3, p2 for the next edge.
        // tmp1, tmp2 (= q2, q3) become p1, p0.
        p3 = fq0;
        p2 = fq1;
        p1 = tmp1;
        p0 = tmp2;
    }
}

// ============================================================================
// 8-row chroma filter functions
// These process both U and V planes together as 16 rows for efficiency.
// ============================================================================

/// Apply normal horizontal filter (DoFilter6) to U and V planes together.
/// Processes 8 U rows and 8 V rows as a single 16-row operation.
/// Uses Load16x4/Store16x6 (libwebp's approach) with split U/V loads.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_h_filter_uv_edge(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion per plane,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x - 4;
    let u_region: &mut [u8; H_FILTER_UV_REGION] =
        <&mut [u8; H_FILTER_UV_REGION]>::try_from(&mut u_pixels[base..base + H_FILTER_UV_REGION])
            .expect("normal_h_filter_uv_edge: u_pixels buffer too small");
    let v_region: &mut [u8; H_FILTER_UV_REGION] =
        <&mut [u8; H_FILTER_UV_REGION]>::try_from(&mut v_pixels[base..base + H_FILTER_UV_REGION])
            .expect("normal_h_filter_uv_edge: v_pixels buffer too small");

    // Load 8 U rows and 8 V rows into a 16-row array
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..8 {
        let row_start = i * stride;
        rows[i] = simd_mem_x86::_mm_loadu_si64(
            <&[u8; 8]>::try_from(&u_region[row_start..][..8]).unwrap(),
        );
        rows[i + 8] = simd_mem_x86::_mm_loadu_si64(
            <&[u8; 8]>::try_from(&v_region[row_start..][..8]).unwrap(),
        );
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter6
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store 6 modified columns back to U and V planes via Store16x4 + q1q2
    // Region base is (y_start * stride + x - 4), store base is (y_start * stride + x - 3) = +1
    store_uv_16x4_impl!(u_region, v_region, 1, stride, p2, p1, p0, q0);
    store_q1q2_uv_16_impl!(u_region, v_region, 5, stride, q1, q2);
}

/// Apply normal horizontal filter (DoFilter4) to U and V planes together at subblock edges.
/// Processes 8 U rows and 8 V rows as a single 16-row operation.
/// Uses 8-wide load + full transpose, Store16x4 for output.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_h_filter_uv_inner(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion per plane,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let base = y_start * stride + x - 4;
    let u_region: &mut [u8; H_FILTER_UV_REGION] =
        <&mut [u8; H_FILTER_UV_REGION]>::try_from(&mut u_pixels[base..base + H_FILTER_UV_REGION])
            .expect("normal_h_filter_uv_inner: u_pixels buffer too small");
    let v_region: &mut [u8; H_FILTER_UV_REGION] =
        <&mut [u8; H_FILTER_UV_REGION]>::try_from(&mut v_pixels[base..base + H_FILTER_UV_REGION])
            .expect("normal_h_filter_uv_inner: v_pixels buffer too small");

    // Load 8 U rows and 8 V rows
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..8 {
        let row_start = i * stride;
        rows[i] = simd_mem_x86::_mm_loadu_si64(
            <&[u8; 8]>::try_from(&u_region[row_start..][..8]).unwrap(),
        );
        rows[i + 8] = simd_mem_x86::_mm_loadu_si64(
            <&[u8; 8]>::try_from(&v_region[row_start..][..8]).unwrap(),
        );
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter4
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store 4 modified columns back to U and V planes via Store16x4
    // Region base is (y_start * stride + x - 4), store base is (y_start * stride + x - 2) = +2
    store_uv_16x4_impl!(u_region, v_region, 2, stride, p1, p0, q0, q1);
}

// ============================================================================
// AVX2 32-row horizontal filter functions (filtering vertical edges)
// These process 32 rows at once by transposing two 16-row groups, filtering
// with AVX2, then transposing back. Useful for filtering across two vertically
// adjacent macroblocks.
// ============================================================================

/// Transpose 32 rows of 8 bytes each into 8 columns of 32 bytes each.
/// Uses the existing 16-row transpose twice and combines into __m256i.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transpose_8x32_to_32x8(
    _token: X64V3Token,
    rows_lo: &[__m128i; 16],
    rows_hi: &[__m128i; 16],
) -> [__m256i; 8] {
    // Transpose each half using existing SSE transpose
    let cols_lo = transpose_8x16_to_16x8(_token, rows_lo);
    let cols_hi = transpose_8x16_to_16x8(_token, rows_hi);

    // Combine low and high halves into AVX2 registers
    [
        _mm256_set_m128i(cols_hi[0], cols_lo[0]),
        _mm256_set_m128i(cols_hi[1], cols_lo[1]),
        _mm256_set_m128i(cols_hi[2], cols_lo[2]),
        _mm256_set_m128i(cols_hi[3], cols_lo[3]),
        _mm256_set_m128i(cols_hi[4], cols_lo[4]),
        _mm256_set_m128i(cols_hi[5], cols_lo[5]),
        _mm256_set_m128i(cols_hi[6], cols_lo[6]),
        _mm256_set_m128i(cols_hi[7], cols_lo[7]),
    ]
}

/// Transpose 4 columns of 32 bytes each back to 32 rows of 4 bytes.
/// Returns values suitable for storing as 32-bit integers per row.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transpose_4x32_to_32x4(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
) -> [i32; 32] {
    // Split into low and high halves
    let p1_lo = _mm256_castsi256_si128(p1);
    let p1_hi = _mm256_extracti128_si256(p1, 1);
    let p0_lo = _mm256_castsi256_si128(p0);
    let p0_hi = _mm256_extracti128_si256(p0, 1);
    let q0_lo = _mm256_castsi256_si128(q0);
    let q0_hi = _mm256_extracti128_si256(q0, 1);
    let q1_lo = _mm256_castsi256_si128(q1);
    let q1_hi = _mm256_extracti128_si256(q1, 1);

    // Use existing 16-row transpose on each half
    let lo = transpose_4x16_to_16x4(_token, p1_lo, p0_lo, q0_lo, q1_lo);
    let hi = transpose_4x16_to_16x4(_token, p1_hi, p0_hi, q0_hi, q1_hi);

    // Combine results
    let mut result = [0i32; 32];
    result[..16].copy_from_slice(&lo);
    result[16..].copy_from_slice(&hi);
    result
}

/// Transpose 6 columns of 32 bytes each back to 32 rows of 6 bytes.
/// Returns (4-byte values, 2-byte values) suitable for storing per row.
#[cfg(target_arch = "x86_64")]
#[rite]
fn transpose_6x32_to_32x6(
    _token: X64V3Token,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
) -> ([i32; 32], [i16; 32]) {
    // Split into low and high halves
    let p2_lo = _mm256_castsi256_si128(p2);
    let p2_hi = _mm256_extracti128_si256(p2, 1);
    let p1_lo = _mm256_castsi256_si128(p1);
    let p1_hi = _mm256_extracti128_si256(p1, 1);
    let p0_lo = _mm256_castsi256_si128(p0);
    let p0_hi = _mm256_extracti128_si256(p0, 1);
    let q0_lo = _mm256_castsi256_si128(q0);
    let q0_hi = _mm256_extracti128_si256(q0, 1);
    let q1_lo = _mm256_castsi256_si128(q1);
    let q1_hi = _mm256_extracti128_si256(q1, 1);
    let q2_lo = _mm256_castsi256_si128(q2);
    let q2_hi = _mm256_extracti128_si256(q2, 1);

    // Use existing 16-row transpose on each half
    let (lo4, lo2) = transpose_6x16_to_16x6(_token, p2_lo, p1_lo, p0_lo, q0_lo, q1_lo, q2_lo);
    let (hi4, hi2) = transpose_6x16_to_16x6(_token, p2_hi, p1_hi, p0_hi, q0_hi, q1_hi, q2_hi);

    // Combine results
    let mut result4 = [0i32; 32];
    let mut result2 = [0i16; 32];
    result4[..16].copy_from_slice(&lo4);
    result4[16..].copy_from_slice(&hi4);
    result2[..16].copy_from_slice(&lo2);
    result2[16..].copy_from_slice(&hi2);
    (result4, result2)
}

/// Apply normal horizontal filter (DoFilter4) to 32 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x), processing
/// 32 consecutive rows (e.g., two vertically adjacent macroblocks).
/// Uses the transpose technique: load 32 rows × 8 columns, transpose,
/// apply DoFilter4 logic with AVX2, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[rite]
pub(crate) fn normal_h_filter32_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront and truncate slice so LLVM can prove all interior
    // accesses are in-bounds (eliminates per-row bounds checks).
    let validated_end = (y_start + 31) * stride + x + 4;
    assert!(
        x >= 4 && validated_end <= pixels.len(),
        "normal_h_filter32_inner: bounds check failed"
    );
    let pixels = &mut pixels[..validated_end];

    // Load 32 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    // Split into two groups of 16 for the transpose
    let mut rows_lo = [_mm_setzero_si128(); 16];
    let mut rows_hi = [_mm_setzero_si128(); 16];

    for (i, row) in rows_lo.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }
    for (i, row) in rows_hi.iter_mut().enumerate() {
        let row_start = (y_start + 16 + i) * stride + x - 4;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x32 to 32x8
    let cols = transpose_8x32_to_32x8(_token, &rows_lo, &rows_hi);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_32(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Transpose back: convert 4 columns of 32 to 32 rows of 4
    let packed = transpose_4x32_to_32x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row (p1, p0, q0, q1)
    for (i, &val) in packed.iter().enumerate() {
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
}

/// Apply normal horizontal filter (DoFilter6) to 32 rows at a vertical macroblock edge.
///
/// This filters the edge between column (x-1) and column (x), processing
/// 32 consecutive rows (e.g., two vertically adjacent macroblocks).
/// Uses the transpose technique: load 32 rows × 8 columns, transpose,
/// apply DoFilter6 logic with AVX2, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[rite]
pub(crate) fn normal_h_filter32_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront and truncate slice so LLVM can prove all interior
    // accesses are in-bounds (eliminates per-row bounds checks).
    let validated_end = (y_start + 31) * stride + x + 4;
    assert!(
        x >= 4 && validated_end <= pixels.len(),
        "normal_h_filter32_edge: bounds check failed"
    );
    let pixels = &mut pixels[..validated_end];

    // Load 32 rows of 8 pixels each
    let mut rows_lo = [_mm_setzero_si128(); 16];
    let mut rows_hi = [_mm_setzero_si128(); 16];

    for (i, row) in rows_lo.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }
    for (i, row) in rows_hi.iter_mut().enumerate() {
        let row_start = (y_start + 16 + i) * stride + x - 4;
        *row =
            simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x32 to 32x8
    let cols = transpose_8x32_to_32x8(_token, &rows_lo, &rows_hi);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_32(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Transpose back: convert 6 columns of 32 to 32 rows of 6
    let (packed4, packed2) = transpose_6x32_to_32x6(_token, p2, p1, p0, q0, q1, q2);

    // Store 6 bytes per row (p2, p1, p0, q0, q1, q2)
    for (i, (&val4, &val2)) in packed4.iter().zip(packed2.iter()).enumerate() {
        let row_start = (y_start + i) * stride + x - 3;
        // Store first 4 bytes (p2, p1, p0, q0)
        simd_mem_x86::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val4),
        );
        // Store next 2 bytes (q1, q2)
        simd_mem_x86::_mm_storeu_si16(
            <&mut [u8; 2]>::try_from(&mut pixels[row_start + 4..][..2]).unwrap(),
            _mm_cvtsi32_si128(val2 as i32),
        );
    }
}

// ============================================================================
// 8-pixel chroma vertical filter functions (filtering horizontal edges)
// These process both U and V planes together by packing 8 U + 8 V pixels
// into each __m128i register.
// ============================================================================

/// Apply normal vertical filter (DoFilter6) to U and V planes together at macroblock edges.
/// Processes 8 U pixels and 8 V pixels per row, packed into __m128i registers.
///
/// Must have at least 4 rows before and after the point.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter_uv_edge(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion per plane,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let region_start = point - 4 * stride;
    let u_region: &mut [u8; V_FILTER_UV_REGION] = <&mut [u8; V_FILTER_UV_REGION]>::try_from(
        &mut u_pixels[region_start..region_start + V_FILTER_UV_REGION],
    )
    .expect("normal_v_filter_uv_edge: u_pixels buffer too small");
    let v_region: &mut [u8; V_FILTER_UV_REGION] = <&mut [u8; V_FILTER_UV_REGION]>::try_from(
        &mut v_pixels[region_start..region_start + V_FILTER_UV_REGION],
    )
    .expect("normal_v_filter_uv_edge: v_pixels buffer too small");

    // Helper to load a row with 8 U + 8 V pixels packed together
    // Offsets are relative to region start (point - 4*stride)
    let load_uv_row = |u_reg: &[u8; V_FILTER_UV_REGION],
                       v_reg: &[u8; V_FILTER_UV_REGION],
                       offset: usize|
     -> __m128i {
        let u = simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_reg[offset..][..8]).unwrap());
        let v = simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_reg[offset..][..8]).unwrap());
        _mm_unpacklo_epi64(u, v)
    };

    // Offsets relative to region start: point - 4*stride maps to offset 0
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    let p3 = load_uv_row(u_region, v_region, off_p3);
    let mut p2 = load_uv_row(u_region, v_region, off_p2);
    let mut p1 = load_uv_row(u_region, v_region, off_p1);
    let mut p0 = load_uv_row(u_region, v_region, off_p0);
    let mut q0 = load_uv_row(u_region, v_region, off_q0);
    let mut q1 = load_uv_row(u_region, v_region, off_q1);
    let mut q2 = load_uv_row(u_region, v_region, off_q2);
    let q3 = load_uv_row(u_region, v_region, off_q3);

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter6
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store results back - low 64 bits to U, high 64 bits to V
    // Offsets relative to region start (point - 4*stride)
    let store_uv_row = |u_reg: &mut [u8; V_FILTER_UV_REGION],
                        v_reg: &mut [u8; V_FILTER_UV_REGION],
                        offset: usize,
                        reg: __m128i| {
        simd_mem_x86::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut u_reg[offset..][..16]).unwrap(),
            reg,
        );
        simd_mem_x86::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut v_reg[offset..][..16]).unwrap(),
            _mm_srli_si128(reg, 8),
        );
    };

    store_uv_row(u_region, v_region, off_p2, p2);
    store_uv_row(u_region, v_region, off_p1, p1);
    store_uv_row(u_region, v_region, off_p0, p0);
    store_uv_row(u_region, v_region, off_q0, q0);
    store_uv_row(u_region, v_region, off_q1, q1);
    store_uv_row(u_region, v_region, off_q2, q2);
}

/// Apply normal vertical filter (DoFilter4) to U and V planes together at subblock edges.
/// Processes 8 U pixels and 8 V pixels per row, packed into __m128i registers.
///
/// Must have at least 4 rows before and after the point.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn normal_v_filter_uv_inner(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: assert stride bound + single array conversion per plane,
    // then all interior accesses are check-free.
    assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let region_start = point - 4 * stride;
    let u_region: &mut [u8; V_FILTER_UV_REGION] = <&mut [u8; V_FILTER_UV_REGION]>::try_from(
        &mut u_pixels[region_start..region_start + V_FILTER_UV_REGION],
    )
    .expect("normal_v_filter_uv_inner: u_pixels buffer too small");
    let v_region: &mut [u8; V_FILTER_UV_REGION] = <&mut [u8; V_FILTER_UV_REGION]>::try_from(
        &mut v_pixels[region_start..region_start + V_FILTER_UV_REGION],
    )
    .expect("normal_v_filter_uv_inner: v_pixels buffer too small");

    // Helper to load a row with 8 U + 8 V pixels packed together
    // Offsets are relative to region start (point - 4*stride)
    let load_uv_row = |u_reg: &[u8; V_FILTER_UV_REGION],
                       v_reg: &[u8; V_FILTER_UV_REGION],
                       offset: usize|
     -> __m128i {
        let u = simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_reg[offset..][..8]).unwrap());
        let v = simd_mem_x86::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_reg[offset..][..8]).unwrap());
        _mm_unpacklo_epi64(u, v)
    };

    // Offsets relative to region start: point - 4*stride maps to offset 0
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    let p3 = load_uv_row(u_region, v_region, off_p3);
    let p2 = load_uv_row(u_region, v_region, off_p2);
    let mut p1 = load_uv_row(u_region, v_region, off_p1);
    let mut p0 = load_uv_row(u_region, v_region, off_p0);
    let mut q0 = load_uv_row(u_region, v_region, off_q0);
    let mut q1 = load_uv_row(u_region, v_region, off_q1);
    let q2 = load_uv_row(u_region, v_region, off_q2);
    let q3 = load_uv_row(u_region, v_region, off_q3);

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter4
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store results back - low 64 bits to U, high 64 bits to V
    // Offsets relative to region start (point - 4*stride)
    let store_uv_row = |u_reg: &mut [u8; V_FILTER_UV_REGION],
                        v_reg: &mut [u8; V_FILTER_UV_REGION],
                        offset: usize,
                        reg: __m128i| {
        simd_mem_x86::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut u_reg[offset..][..16]).unwrap(),
            reg,
        );
        simd_mem_x86::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut v_reg[offset..][..16]).unwrap(),
            _mm_srli_si128(reg, 8),
        );
    };

    store_uv_row(u_region, v_region, off_p1, p1);
    store_uv_row(u_region, v_region, off_p0, p0);
    store_uv_row(u_region, v_region, off_q0, q0);
    store_uv_row(u_region, v_region, off_q1, q1);
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec;

    // #[arcane] wrappers for calling #[rite] filter functions from tests.
    // These are needed because #[rite] functions have #[target_feature] and
    // cannot be called directly from non-target_feature test functions.
    // Performance doesn't matter here — only the hot decode path uses
    // the single #[arcane] entry: filter_row_simd.

    #[archmage::arcane]
    fn call_simple_v_filter16(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        thresh: i32,
    ) {
        simple_v_filter16(t, pixels, point, stride, thresh);
    }

    #[archmage::arcane]
    fn call_simple_v_filter32(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        thresh: i32,
    ) {
        simple_v_filter32(t, pixels, point, stride, thresh);
    }

    #[archmage::arcane]
    fn call_simple_h_filter16(
        t: X64V3Token,
        pixels: &mut [u8],
        x: usize,
        y_start: usize,
        stride: usize,
        thresh: i32,
    ) {
        simple_h_filter16(t, pixels, x, y_start, stride, thresh);
    }

    #[archmage::arcane]
    fn call_normal_v_filter16_inner(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_v_filter16_inner(
            t,
            pixels,
            point,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_v_filter16_edge(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_v_filter16_edge(
            t,
            pixels,
            point,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_v_filter32_inner(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_v_filter32_inner(
            t,
            pixels,
            point,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_v_filter32_edge(
        t: X64V3Token,
        pixels: &mut [u8],
        point: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_v_filter32_edge(
            t,
            pixels,
            point,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_h_filter16_inner(
        t: X64V3Token,
        pixels: &mut [u8],
        x: usize,
        y_start: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_h_filter16_inner(
            t,
            pixels,
            x,
            y_start,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_h_filter16_edge(
        t: X64V3Token,
        pixels: &mut [u8],
        x: usize,
        y_start: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_h_filter16_edge(
            t,
            pixels,
            x,
            y_start,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_h_filter32_inner(
        t: X64V3Token,
        pixels: &mut [u8],
        x: usize,
        y_start: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_h_filter32_inner(
            t,
            pixels,
            x,
            y_start,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

    #[archmage::arcane]
    fn call_normal_h_filter32_edge(
        t: X64V3Token,
        pixels: &mut [u8],
        x: usize,
        y_start: usize,
        stride: usize,
        hev_thresh: i32,
        interior_limit: i32,
        edge_limit: i32,
    ) {
        normal_h_filter32_edge(
            t,
            pixels,
            x,
            y_start,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
    }

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
    #[cfg(target_arch = "x86_64")]
    fn test_simple_v_filter16_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 8 + V_FILTER_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up a gradient that should be filtered
        // Row 0 = p1, Row 1 = p0, Row 2 = q0, Row 3 = q1
        for x in 0..16 {
            pixels[x] = 100; // row 0: p1
            pixels[stride + x] = 110; // row 1: p0
            pixels[2 * stride + x] = 140; // row 2: q0
            pixels[3 * stride + x] = 150; // row 3: q1

            pixels_scalar[x] = 100;
            pixels_scalar[stride + x] = 110;
            pixels_scalar[2 * stride + x] = 140;
            pixels_scalar[3 * stride + x] = 150;
        }

        let thresh = 40;

        // Apply SIMD filter (edge between row 1 and row 2, so point = 2 * stride)
        call_simple_v_filter16(token, &mut pixels, 2 * stride, stride, thresh);

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
                pixels[stride + x],
                pixels_scalar[stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "q0 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simple_h_filter16_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        let mut pixels = vec![128u8; stride * 20 + H_FILTER_SIMPLE_REGION];
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
        call_simple_h_filter16(token, &mut pixels, 4, 0, stride, thresh);

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
                pixels[y * stride + 3],
                pixels_scalar[y * stride + 3],
                "p0 mismatch at y={}",
                y
            );
            assert_eq!(
                pixels[y * stride + 4],
                pixels_scalar[y * stride + 4],
                "q0 mismatch at y={}",
                y
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_v_filter16_inner_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_NORMAL_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient data for all 8 rows around the edge
        for x in 0..16 {
            pixels[0 * stride + x] = 100; // p3
            pixels[1 * stride + x] = 105; // p2
            pixels[2 * stride + x] = 110; // p1
            pixels[3 * stride + x] = 115; // p0
            pixels[4 * stride + x] = 145; // q0 (edge here)
            pixels[5 * stride + x] = 150; // q1
            pixels[6 * stride + x] = 155; // q2
            pixels[7 * stride + x] = 160; // q3

            pixels_scalar[0 * stride + x] = 100;
            pixels_scalar[1 * stride + x] = 105;
            pixels_scalar[2 * stride + x] = 110;
            pixels_scalar[3 * stride + x] = 115;
            pixels_scalar[4 * stride + x] = 145;
            pixels_scalar[5 * stride + x] = 150;
            pixels_scalar[6 * stride + x] = 155;
            pixels_scalar[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply SIMD filter
        call_normal_v_filter16_inner(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply scalar filter
        for x in 0..16 {
            crate::decoder::loop_filter::subblock_filter_vertical(
                hev_thresh as u8,
                interior_limit as u8,
                edge_limit as u8,
                &mut pixels_scalar,
                4 * stride + x,
                stride,
            );
        }

        // Compare - we compare p1, p0, q0, q1 which can be modified
        for x in 0..16 {
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_scalar[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_scalar[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_scalar[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_v_filter16_edge_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_NORMAL_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient data for all 8 rows around the edge
        for x in 0..16 {
            pixels[0 * stride + x] = 100; // p3
            pixels[1 * stride + x] = 105; // p2
            pixels[2 * stride + x] = 110; // p1
            pixels[3 * stride + x] = 115; // p0
            pixels[4 * stride + x] = 145; // q0 (edge here)
            pixels[5 * stride + x] = 150; // q1
            pixels[6 * stride + x] = 155; // q2
            pixels[7 * stride + x] = 160; // q3

            pixels_scalar[0 * stride + x] = 100;
            pixels_scalar[1 * stride + x] = 105;
            pixels_scalar[2 * stride + x] = 110;
            pixels_scalar[3 * stride + x] = 115;
            pixels_scalar[4 * stride + x] = 145;
            pixels_scalar[5 * stride + x] = 150;
            pixels_scalar[6 * stride + x] = 155;
            pixels_scalar[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 40;

        // Apply SIMD filter
        call_normal_v_filter16_edge(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply scalar filter
        for x in 0..16 {
            crate::decoder::loop_filter::macroblock_filter_vertical(
                hev_thresh as u8,
                interior_limit as u8,
                edge_limit as u8,
                &mut pixels_scalar,
                4 * stride + x,
                stride,
            );
        }

        // Compare - p2, p1, p0, q0, q1, q2 can be modified
        for x in 0..16 {
            assert_eq!(
                pixels[1 * stride + x],
                pixels_scalar[1 * stride + x],
                "p2 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_scalar[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_scalar[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_scalar[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[6 * stride + x],
                pixels_scalar[6 * stride + x],
                "q2 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simple_v_filter32_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_REGION_32 padding
        let mut pixels = vec![128u8; stride * 8 + V_FILTER_REGION_32];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient for 32 pixels
        for x in 0..32 {
            pixels[x] = 100;
            pixels[stride + x] = 110;
            pixels[2 * stride + x] = 140;
            pixels[3 * stride + x] = 150;

            pixels_scalar[x] = 100;
            pixels_scalar[stride + x] = 110;
            pixels_scalar[2 * stride + x] = 140;
            pixels_scalar[3 * stride + x] = 150;
        }

        let thresh = 40;

        // Apply AVX2 32-pixel filter
        call_simple_v_filter32(token, &mut pixels, 2 * stride, stride, thresh);

        // Apply scalar filter
        for x in 0..32 {
            let p1 = pixels_scalar[x];
            let p0 = pixels_scalar[stride + x];
            let q0 = pixels_scalar[2 * stride + x];
            let q1 = pixels_scalar[3 * stride + x];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[stride + x] = new_p0;
            pixels_scalar[2 * stride + x] = new_q0;
        }

        // Compare
        for x in 0..32 {
            assert_eq!(
                pixels[stride + x],
                pixels_scalar[stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "q0 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_v_filter32_inner_matches_two_16() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_NORMAL_REGION_32 padding
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION_32];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 8 rows around the edge (32 pixels wide)
        for x in 0..32 {
            pixels[0 * stride + x] = 100;
            pixels[1 * stride + x] = 105;
            pixels[2 * stride + x] = 110;
            pixels[3 * stride + x] = 115;
            pixels[4 * stride + x] = 145;
            pixels[5 * stride + x] = 150;
            pixels[6 * stride + x] = 155;
            pixels[7 * stride + x] = 160;

            pixels_16[0 * stride + x] = 100;
            pixels_16[1 * stride + x] = 105;
            pixels_16[2 * stride + x] = 110;
            pixels_16[3 * stride + x] = 115;
            pixels_16[4 * stride + x] = 145;
            pixels_16[5 * stride + x] = 150;
            pixels_16[6 * stride + x] = 155;
            pixels_16[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-pixel filter
        call_normal_v_filter32_inner(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-pixel filter
        call_normal_v_filter16_inner(
            token,
            &mut pixels_16,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        call_normal_v_filter16_inner(
            token,
            &mut pixels_16,
            4 * stride + 16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 pixels for modified rows (p1, p0, q0, q1)
        for x in 0..32 {
            assert_eq!(
                pixels[2 * stride + x],
                pixels_16[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_16[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_16[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_16[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_v_filter32_edge_matches_two_16() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_NORMAL_REGION_32 padding
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION_32];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 8 rows around the edge (32 pixels wide)
        for x in 0..32 {
            pixels[0 * stride + x] = 100;
            pixels[1 * stride + x] = 105;
            pixels[2 * stride + x] = 110;
            pixels[3 * stride + x] = 115;
            pixels[4 * stride + x] = 145;
            pixels[5 * stride + x] = 150;
            pixels[6 * stride + x] = 155;
            pixels[7 * stride + x] = 160;

            pixels_16[0 * stride + x] = 100;
            pixels_16[1 * stride + x] = 105;
            pixels_16[2 * stride + x] = 110;
            pixels_16[3 * stride + x] = 115;
            pixels_16[4 * stride + x] = 145;
            pixels_16[5 * stride + x] = 150;
            pixels_16[6 * stride + x] = 155;
            pixels_16[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-pixel filter
        call_normal_v_filter32_edge(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-pixel filter
        call_normal_v_filter16_edge(
            token,
            &mut pixels_16,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        call_normal_v_filter16_edge(
            token,
            &mut pixels_16,
            4 * stride + 16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 pixels for modified rows (p2, p1, p0, q0, q1, q2)
        for x in 0..32 {
            assert_eq!(
                pixels[1 * stride + x],
                pixels_16[1 * stride + x],
                "p2 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_16[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_16[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_16[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_16[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[6 * stride + x],
                pixels_16[6 * stride + x],
                "q2 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_h_filter32_inner_matches_two_16() {
        let Some(token) = X64V3Token::summon() else {
            std::eprintln!("AVX2 not available, skipping test");
            return;
        };

        let width = 64;
        let height = 48; // Need 32 rows + padding
        let stride = width;
        let mut pixels = vec![128u8; stride * height + H_FILTER_NORMAL_REGION];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 32 rows around the vertical edge at x=16
        for y in 0..32 {
            for x in 0..8 {
                let base = y * stride + 12 + x; // columns 12-19 around edge at x=16
                pixels[base] = (100 + x * 10) as u8;
                pixels_16[base] = (100 + x * 10) as u8;
            }
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-row filter (processing 32 rows at once)
        call_normal_h_filter32_inner(
            token,
            &mut pixels,
            16, // x
            0,  // y_start
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-row filter (rows 0-15 and rows 16-31)
        call_normal_h_filter16_inner(
            token,
            &mut pixels_16,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        call_normal_h_filter16_inner(
            token,
            &mut pixels_16,
            16,
            16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 rows for modified columns (p1, p0, q0, q1 = x-2 to x+1)
        for y in 0..32 {
            for x in 14..18 {
                assert_eq!(
                    pixels[y * stride + x],
                    pixels_16[y * stride + x],
                    "mismatch at y={}, x={}",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_normal_h_filter32_edge_matches_two_16() {
        let Some(token) = X64V3Token::summon() else {
            std::eprintln!("AVX2 not available, skipping test");
            return;
        };

        let width = 64;
        let height = 48;
        let stride = width;
        let mut pixels = vec![128u8; stride * height + H_FILTER_NORMAL_REGION];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 32 rows around the vertical edge at x=16
        for y in 0..32 {
            for x in 0..8 {
                let base = y * stride + 12 + x;
                pixels[base] = (100 + x * 10) as u8;
                pixels_16[base] = (100 + x * 10) as u8;
            }
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-row filter
        call_normal_h_filter32_edge(
            token,
            &mut pixels,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-row filter
        call_normal_h_filter16_edge(
            token,
            &mut pixels_16,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        call_normal_h_filter16_edge(
            token,
            &mut pixels_16,
            16,
            16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 rows for modified columns (p2, p1, p0, q0, q1, q2 = x-3 to x+2)
        for y in 0..32 {
            for x in 13..19 {
                assert_eq!(
                    pixels[y * stride + x],
                    pixels_16[y * stride + x],
                    "mismatch at y={}, x={}",
                    y,
                    x
                );
            }
        }
    }
}

// ============================================================================
// NEON filter implementations (aarch64)
// ============================================================================

// =============================================================================
// Core filter helpers (ported from libwebp dec_neon.c)
// =============================================================================

/// NeedsFilter: returns mask where 2*|p0-q0| + |p1-q1|/2 <= thresh

#[rite]
fn needs_filter_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    thresh: i32,
) -> uint8x16_t {
    let thresh_v = vdupq_n_u8(thresh as u8);
    let a_p0_q0 = vabdq_u8(p0, q0); // abs(p0 - q0)
    let a_p1_q1 = vabdq_u8(p1, q1); // abs(p1 - q1)
    let a_p0_q0_2 = vqaddq_u8(a_p0_q0, a_p0_q0); // 2 * abs(p0-q0) saturating
    let a_p1_q1_2 = vshrq_n_u8::<1>(a_p1_q1); // abs(p1-q1) / 2
    let sum = vqaddq_u8(a_p0_q0_2, a_p1_q1_2);
    vcgeq_u8(thresh_v, sum) // mask where thresh >= sum
}

/// NeedsFilter2: extended filter check for normal (non-simple) filter
/// Checks NeedsFilter AND all adjacent differences <= ithresh

#[rite]
fn needs_filter2_neon(
    _token: NeonToken,
    p3: uint8x16_t,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    q3: uint8x16_t,
    ithresh: i32,
    thresh: i32,
) -> uint8x16_t {
    let ithresh_v = vdupq_n_u8(ithresh as u8);
    let a_p3_p2 = vabdq_u8(p3, p2);
    let a_p2_p1 = vabdq_u8(p2, p1);
    let a_p1_p0 = vabdq_u8(p1, p0);
    let a_q3_q2 = vabdq_u8(q3, q2);
    let a_q2_q1 = vabdq_u8(q2, q1);
    let a_q1_q0 = vabdq_u8(q1, q0);
    let max1 = vmaxq_u8(a_p3_p2, a_p2_p1);
    let max2 = vmaxq_u8(a_p1_p0, a_q3_q2);
    let max3 = vmaxq_u8(a_q2_q1, a_q1_q0);
    let max12 = vmaxq_u8(max1, max2);
    let max123 = vmaxq_u8(max12, max3);
    let mask2 = vcgeq_u8(ithresh_v, max123);
    let mask1 = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    vandq_u8(mask1, mask2)
}

/// NeedsHev: returns mask where max(|p1-p0|, |q1-q0|) > hev_thresh

#[rite]
fn needs_hev_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    hev_thresh: i32,
) -> uint8x16_t {
    let hev_thresh_v = vdupq_n_u8(hev_thresh as u8);
    let a_p1_p0 = vabdq_u8(p1, p0);
    let a_q1_q0 = vabdq_u8(q1, q0);
    let a_max = vmaxq_u8(a_p1_p0, a_q1_q0);
    vcgtq_u8(a_max, hev_thresh_v) // mask where max > hev_thresh
}

/// Convert unsigned to signed by XOR with 0x80

#[rite]
fn flip_sign_neon(_token: NeonToken, v: uint8x16_t) -> int8x16_t {
    let sign_bit = vdupq_n_u8(0x80);
    vreinterpretq_s8_u8(veorq_u8(v, sign_bit))
}

/// Convert signed back to unsigned by XOR with 0x80

#[rite]
fn flip_sign_back_neon(_token: NeonToken, v: int8x16_t) -> uint8x16_t {
    let sign_bit = vdupq_n_s8(-128); // 0x80 as i8
    vreinterpretq_u8_s8(veorq_s8(v, sign_bit))
}

/// GetBaseDelta: compute (p1-q1) + 3*(q0-p0) with saturation

#[rite]
fn get_base_delta_neon(
    _token: NeonToken,
    p1s: int8x16_t,
    p0s: int8x16_t,
    q0s: int8x16_t,
    q1s: int8x16_t,
) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0s, p0s); // (q0 - p0) saturating
    let p1_q1 = vqsubq_s8(p1s, q1s); // (p1 - q1) saturating
    let s1 = vqaddq_s8(p1_q1, q0_p0); // (p1-q1) + 1*(q0-p0)
    let s2 = vqaddq_s8(q0_p0, s1); // (p1-q1) + 2*(q0-p0)
    vqaddq_s8(q0_p0, s2) // (p1-q1) + 3*(q0-p0)
}

/// GetBaseDelta0: compute 3*(q0-p0) with saturation (no p1/q1)

#[rite]
fn get_base_delta0_neon(_token: NeonToken, p0s: int8x16_t, q0s: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0s, p0s);
    let s1 = vqaddq_s8(q0_p0, q0_p0); // 2*(q0-p0)
    vqaddq_s8(q0_p0, s1) // 3*(q0-p0)
}

// =============================================================================
// Filter application functions
// =============================================================================

/// ApplyFilter2NoFlip: 2-tap filter without sign flip back (stays in signed domain)

#[rite]
fn apply_filter2_no_flip_neon(
    _token: NeonToken,
    p0s: int8x16_t,
    q0s: int8x16_t,
    delta: int8x16_t,
) -> (int8x16_t, int8x16_t) {
    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta_p3 = vqaddq_s8(delta, k3);
    let delta_p4 = vqaddq_s8(delta, k4);
    let delta3 = vshrq_n_s8::<3>(delta_p3);
    let delta4 = vshrq_n_s8::<3>(delta_p4);
    let op0 = vqaddq_s8(p0s, delta3);
    let oq0 = vqsubq_s8(q0s, delta4);
    (op0, oq0)
}

/// DoFilter2: simple 2-tap filter (used for simple loop filter)
/// Returns (filtered_p0, filtered_q0) as unsigned

#[rite]
fn do_filter2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let p1s = flip_sign_neon(_token, p1);
    let p0s = flip_sign_neon(_token, p0);
    let q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let delta0 = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);
    let delta1 = vandq_s8(delta0, vreinterpretq_s8_u8(mask));

    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta_p3 = vqaddq_s8(delta1, k3);
    let delta_p4 = vqaddq_s8(delta1, k4);
    let delta3 = vshrq_n_s8::<3>(delta_p3);
    let delta4 = vshrq_n_s8::<3>(delta_p4);
    let sp0 = vqaddq_s8(p0s, delta3);
    let sq0 = vqsubq_s8(q0s, delta4);
    (
        flip_sign_back_neon(_token, sp0),
        flip_sign_back_neon(_token, sq0),
    )
}

/// DoFilter4: 4-tap filter for subblock edges (inner edges)
/// Returns (op1, op0, oq0, oq1)

#[rite]
fn do_filter4_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    let p1s = flip_sign_neon(_token, p1);
    let mut p0s = flip_sign_neon(_token, p0);
    let mut q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let simple_lf_mask = vandq_u8(mask, hev_mask);

    // do_filter2 part: simple filter on pixels with hev
    {
        let delta = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);
        let simple_lf_delta = vandq_s8(delta, vreinterpretq_s8_u8(simple_lf_mask));
        let (new_p0s, new_q0s) = apply_filter2_no_flip_neon(_token, p0s, q0s, simple_lf_delta);
        p0s = new_p0s;
        q0s = new_q0s;
    }

    // do_filter4 part: complex filter on pixels without hev
    let delta0 = get_base_delta0_neon(_token, p0s, q0s);
    // (mask & hev_mask) ^ mask = mask & !hev_mask
    let complex_lf_mask = veorq_u8(simple_lf_mask, mask);
    let complex_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(complex_lf_mask));

    // ApplyFilter4
    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta1 = vqaddq_s8(complex_lf_delta, k4);
    let delta2 = vqaddq_s8(complex_lf_delta, k3);
    let a1 = vshrq_n_s8::<3>(delta1);
    let a2 = vshrq_n_s8::<3>(delta2);
    let a3 = vrshrq_n_s8::<1>(a1); // (a1 + 1) >> 1

    let op0 = flip_sign_back_neon(_token, vqaddq_s8(p0s, a2));
    let oq0 = flip_sign_back_neon(_token, vqsubq_s8(q0s, a1));
    let op1 = flip_sign_back_neon(_token, vqaddq_s8(p1s, a3));
    let oq1 = flip_sign_back_neon(_token, vqsubq_s8(q1s, a3));

    (op1, op0, oq0, oq1)
}

/// DoFilter6: 6-tap filter for macroblock edges
/// Returns (op2, op1, op0, oq0, oq1, oq2)

#[rite]
#[allow(clippy::type_complexity)]
fn do_filter6_neon(
    _token: NeonToken,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    let p2s = flip_sign_neon(_token, p2);
    let p1s = flip_sign_neon(_token, p1);
    let mut p0s = flip_sign_neon(_token, p0);
    let mut q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let q2s = flip_sign_neon(_token, q2);
    let simple_lf_mask = vandq_u8(mask, hev_mask);
    let delta0 = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);

    // do_filter2 part: simple filter on pixels with hev
    {
        let simple_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(simple_lf_mask));
        let (new_p0s, new_q0s) = apply_filter2_no_flip_neon(_token, p0s, q0s, simple_lf_delta);
        p0s = new_p0s;
        q0s = new_q0s;
    }

    // do_filter6 part: 6-tap filter on pixels without hev
    let complex_lf_mask = veorq_u8(simple_lf_mask, mask);
    let complex_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(complex_lf_mask));

    // ApplyFilter6: compute X=(9*a+63)>>7, Y=(18*a+63)>>7, Z=(27*a+63)>>7
    // Using S = 9*a - 1 trick with vqrshrn_n_s16
    let delta_lo = vget_low_s8(complex_lf_delta);
    let delta_hi = vget_high_s8(complex_lf_delta);
    let k9 = vdup_n_s8(9);
    let km1 = vdupq_n_s16(-1);
    let k18 = vdup_n_s8(18);

    let s_lo = vmlal_s8(km1, k9, delta_lo); // S = 9 * a - 1
    let s_hi = vmlal_s8(km1, k9, delta_hi);
    let z_lo = vmlal_s8(s_lo, k18, delta_lo); // S + 18 * a
    let z_hi = vmlal_s8(s_hi, k18, delta_hi);

    let a3_lo = vqrshrn_n_s16::<7>(s_lo); // (9*a + 63) >> 7
    let a3_hi = vqrshrn_n_s16::<7>(s_hi);
    let a2_lo = vqrshrn_n_s16::<6>(s_lo); // (9*a + 31) >> 6
    let a2_hi = vqrshrn_n_s16::<6>(s_hi);
    let a1_lo = vqrshrn_n_s16::<7>(z_lo); // (27*a + 63) >> 7
    let a1_hi = vqrshrn_n_s16::<7>(z_hi);

    let a1 = vcombine_s8(a1_lo, a1_hi);
    let a2 = vcombine_s8(a2_lo, a2_hi);
    let a3 = vcombine_s8(a3_lo, a3_hi);

    let op0 = flip_sign_back_neon(_token, vqaddq_s8(p0s, a1));
    let oq0 = flip_sign_back_neon(_token, vqsubq_s8(q0s, a1));
    let op1 = flip_sign_back_neon(_token, vqaddq_s8(p1s, a2));
    let oq1 = flip_sign_back_neon(_token, vqsubq_s8(q1s, a2));
    let op2 = flip_sign_back_neon(_token, vqaddq_s8(p2s, a3));
    let oq2 = flip_sign_back_neon(_token, vqsubq_s8(q2s, a3));

    (op2, op1, op0, oq0, oq1, oq2)
}

// =============================================================================
// Load/Store helpers
// =============================================================================

/// Load 16 pixels from 4 consecutive rows (for vertical filter)

#[rite]
fn load_16x4_neon(
    _token: NeonToken,
    buf: &[u8],
    point: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    let p1 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 2 * stride..][..16]).unwrap());
    let p0 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - stride..][..16]).unwrap());
    let q0 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point..][..16]).unwrap());
    let q1 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + stride..][..16]).unwrap());
    (p1, p0, q0, q1)
}

/// Load 16 pixels from 8 consecutive rows (for normal vertical filter)

#[rite]
#[allow(clippy::type_complexity)]
fn load_16x8_neon(
    _token: NeonToken,
    buf: &[u8],
    point: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    let p3 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 4 * stride..][..16]).unwrap());
    let p2 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 3 * stride..][..16]).unwrap());
    let p1 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 2 * stride..][..16]).unwrap());
    let p0 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - stride..][..16]).unwrap());
    let q0 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point..][..16]).unwrap());
    let q1 = simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + stride..][..16]).unwrap());
    let q2 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + 2 * stride..][..16]).unwrap());
    let q3 =
        simd_mem_neon::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + 3 * stride..][..16]).unwrap());
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 16 pixels to 2 consecutive rows

#[rite]
fn store_16x2_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    p0: uint8x16_t,
    q0: uint8x16_t,
) {
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        p0,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        q0,
    );
}

/// Load 8 U pixels + 8 V pixels into a single uint8x16_t per row (for UV vertical filter)
/// Loads 8 rows centered on the edge.

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x8x2_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    point: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Pack U in low half, V in high half
    let p3 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 4 * stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 4 * stride..][..8]).unwrap()),
    );
    let p2 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 3 * stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 3 * stride..][..8]).unwrap()),
    );
    let p1 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 2 * stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 2 * stride..][..8]).unwrap()),
    );
    let p0 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - stride..][..8]).unwrap()),
    );
    let q0 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point..][..8]).unwrap()),
    );
    let q1 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + stride..][..8]).unwrap()),
    );
    let q2 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + 2 * stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + 2 * stride..][..8]).unwrap()),
    );
    let q3 = vcombine_u8(
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + 3 * stride..][..8]).unwrap()),
        simd_mem_neon::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + 3 * stride..][..8]).unwrap()),
    );
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 8x2x2: store U+V packed results back to separate buffers (2 rows)

#[rite]
fn store_8x2x2_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    // low half = U, high half = V
    simd_mem_neon::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut u_buf[u_point - stride..][..8]).unwrap(),
        vget_low_u8(p0),
    );
    simd_mem_neon::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut u_buf[u_point..][..8]).unwrap(),
        vget_low_u8(q0),
    );
    simd_mem_neon::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut v_buf[v_point - stride..][..8]).unwrap(),
        vget_high_u8(p0),
    );
    simd_mem_neon::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut v_buf[v_point..][..8]).unwrap(),
        vget_high_u8(q0),
    );
}

/// Store 8x4x2: store U+V packed results (4 rows)

#[rite]
fn store_8x4x2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    store_8x2x2_neon(
        _token,
        p1,
        p0,
        u_buf,
        v_buf,
        u_point - stride,
        v_point - stride,
        stride,
    );
    store_8x2x2_neon(
        _token,
        q0,
        q1,
        u_buf,
        v_buf,
        u_point + stride,
        v_point + stride,
        stride,
    );
}

/// Store 6 rows (3 above, 3 below) for UV 6-tap filter

#[rite]
fn store_8x6x2_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    store_8x2x2_neon(
        _token,
        op2,
        op1,
        u_buf,
        v_buf,
        u_point - 2 * stride,
        v_point - 2 * stride,
        stride,
    );
    store_8x2x2_neon(_token, op0, oq0, u_buf, v_buf, u_point, v_point, stride);
    store_8x2x2_neon(
        _token,
        oq1,
        oq2,
        u_buf,
        v_buf,
        u_point + 2 * stride,
        v_point + 2 * stride,
        stride,
    );
}

// =============================================================================
// Horizontal filter load/store (transpose-based for vertical edges)
// =============================================================================

/// Load 4 columns from 16 rows as transposed uint8x16_t vectors.
/// Loads 4 bytes per row, packs into u32x4 registers, and transposes.
/// buf[y_start + row * stride + x0 - 2 .. x0 + 2] for row 0..16

#[rite]
fn load_4x16_neon(
    _token: NeonToken,
    buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    // Build column vectors directly: byte[i] = column value for row i.
    // This matches store_4x16_neon which writes byte[i] → row i.
    let base = y_start * stride + x0 - 2;
    let mut col0 = [0u8; 16];
    let mut col1 = [0u8; 16];
    let mut col2 = [0u8; 16];
    let mut col3 = [0u8; 16];

    for i in 0..16 {
        let offset = base + i * stride;
        col0[i] = buf[offset];
        col1[i] = buf[offset + 1];
        col2[i] = buf[offset + 2];
        col3[i] = buf[offset + 3];
    }

    let p1 = simd_mem_neon::vld1q_u8(&col0);
    let p0 = simd_mem_neon::vld1q_u8(&col1);
    let q0 = simd_mem_neon::vld1q_u8(&col2);
    let q1 = simd_mem_neon::vld1q_u8(&col3);

    (p1, p0, q0, q1)
}

/// Load 8 columns from 16 rows (for normal horizontal filter)

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x16_neon(
    _token: NeonToken,
    buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Load columns x0-4..x0+4 from 16 rows, transposed
    // Load the left 4 columns and right 4 columns separately
    let (p3, p2, p1, p0) = load_4x16_neon(_token, buf, x0 - 2, y_start, stride);
    let (q0, q1, q2, q3) = load_4x16_neon(_token, buf, x0 + 2, y_start, stride);
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 2 transposed columns back to 16 rows
/// Writes p0 and q0 to columns (x0-1) and (x0) for 16 rows

#[rite]
fn store_2x16_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    // Extract each lane pair and write 2 bytes per row
    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 1;
        buf[offset] = vgetq_lane_u8::<0>(vextq_u8::<0>(p0, p0)); // placeholder
        // This needs per-lane extraction. Let's use a different approach.
    }
    // Actually, extract all values at once
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut p0_bytes, p0);
    simd_mem_neon::vst1q_u8(&mut q0_bytes, q0);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 1;
        buf[offset] = p0_bytes[i];
        buf[offset + 1] = q0_bytes[i];
    }
}

/// Store 4 transposed columns back to 16 rows
/// Writes p1, p0, q0, q1 to columns (x0-2)..(x0+2) for 16 rows

#[rite]
fn store_4x16_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut p1_bytes = [0u8; 16];
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    let mut q1_bytes = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut p1_bytes, p1);
    simd_mem_neon::vst1q_u8(&mut p0_bytes, p0);
    simd_mem_neon::vst1q_u8(&mut q0_bytes, q0);
    simd_mem_neon::vst1q_u8(&mut q1_bytes, q1);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 2;
        buf[offset] = p1_bytes[i];
        buf[offset + 1] = p0_bytes[i];
        buf[offset + 2] = q0_bytes[i];
        buf[offset + 3] = q1_bytes[i];
    }
}

/// Store 6 transposed columns back to 16 rows (for normal h-filter edge)

#[rite]
fn store_6x16_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b2 = [0u8; 16];
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    let mut a2 = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut b2, op2);
    simd_mem_neon::vst1q_u8(&mut b1, op1);
    simd_mem_neon::vst1q_u8(&mut b0, op0);
    simd_mem_neon::vst1q_u8(&mut a0, oq0);
    simd_mem_neon::vst1q_u8(&mut a1, oq1);
    simd_mem_neon::vst1q_u8(&mut a2, oq2);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 3;
        buf[offset] = b2[i];
        buf[offset + 1] = b1[i];
        buf[offset + 2] = b0[i];
        buf[offset + 3] = a0[i];
        buf[offset + 4] = a1[i];
        buf[offset + 5] = a2[i];
    }
}

// =============================================================================
// UV horizontal filter load/store (transpose 8 rows from U + 8 from V)
// =============================================================================

/// Load 4 columns from 8 U rows + 8 V rows, packed in uint8x16_t (transposed)

#[rite]
fn load_4x8x2_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    // Build column vectors directly: bytes 0-7 = U rows 0-7, bytes 8-15 = V rows 0-7.
    // This matches store_4x8x2_neon which writes bytes[0..8] → U, bytes[8..16] → V.
    let base_u = y_start * stride + x0 - 2;
    let base_v = y_start * stride + x0 - 2;
    let mut col0 = [0u8; 16];
    let mut col1 = [0u8; 16];
    let mut col2 = [0u8; 16];
    let mut col3 = [0u8; 16];

    for i in 0..8 {
        let u_off = base_u + i * stride;
        let v_off = base_v + i * stride;
        col0[i] = u_buf[u_off];
        col1[i] = u_buf[u_off + 1];
        col2[i] = u_buf[u_off + 2];
        col3[i] = u_buf[u_off + 3];
        col0[8 + i] = v_buf[v_off];
        col1[8 + i] = v_buf[v_off + 1];
        col2[8 + i] = v_buf[v_off + 2];
        col3[8 + i] = v_buf[v_off + 3];
    }

    let p1 = simd_mem_neon::vld1q_u8(&col0);
    let p0 = simd_mem_neon::vld1q_u8(&col1);
    let q0 = simd_mem_neon::vld1q_u8(&col2);
    let q1 = simd_mem_neon::vld1q_u8(&col3);

    (p1, p0, q0, q1)
}

/// Load 8 columns from 8 U rows + 8 V rows for normal horizontal chroma filter

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x8x2_h_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Load left 4 and right 4 columns separately
    let (p3, p2, p1, p0) = load_4x8x2_neon(_token, u_buf, v_buf, x0 - 2, y_start, stride);
    // For the right side, shift x0 by 2 (the load_4x8x2 already does x0-2 internally)
    let (q0, q1, q2, q3) = load_4x8x2_neon(_token, u_buf, v_buf, x0 + 2, y_start, stride);
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 2 transposed columns back to 8 U rows + 8 V rows
#[allow(dead_code)]
#[rite]
fn store_2x8x2_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut p0_bytes, p0);
    simd_mem_neon::vst1q_u8(&mut q0_bytes, q0);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 1;
        u_buf[offset] = p0_bytes[i];
        u_buf[offset + 1] = q0_bytes[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 1;
        v_buf[offset] = p0_bytes[8 + i];
        v_buf[offset + 1] = q0_bytes[8 + i];
    }
}

/// Store 4 transposed columns back to 8 U rows + 8 V rows

#[rite]
fn store_4x8x2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut b1, p1);
    simd_mem_neon::vst1q_u8(&mut b0, p0);
    simd_mem_neon::vst1q_u8(&mut a0, q0);
    simd_mem_neon::vst1q_u8(&mut a1, q1);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 2;
        u_buf[offset] = b1[i];
        u_buf[offset + 1] = b0[i];
        u_buf[offset + 2] = a0[i];
        u_buf[offset + 3] = a1[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 2;
        v_buf[offset] = b1[8 + i];
        v_buf[offset + 1] = b0[8 + i];
        v_buf[offset + 2] = a0[8 + i];
        v_buf[offset + 3] = a1[8 + i];
    }
}

/// Store 6 transposed columns back to 8 U rows + 8 V rows (6-tap edge filter)

#[rite]
fn store_6x8x2_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b2 = [0u8; 16];
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    let mut a2 = [0u8; 16];
    simd_mem_neon::vst1q_u8(&mut b2, op2);
    simd_mem_neon::vst1q_u8(&mut b1, op1);
    simd_mem_neon::vst1q_u8(&mut b0, op0);
    simd_mem_neon::vst1q_u8(&mut a0, oq0);
    simd_mem_neon::vst1q_u8(&mut a1, oq1);
    simd_mem_neon::vst1q_u8(&mut a2, oq2);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 3;
        u_buf[offset] = b2[i];
        u_buf[offset + 1] = b1[i];
        u_buf[offset + 2] = b0[i];
        u_buf[offset + 3] = a0[i];
        u_buf[offset + 4] = a1[i];
        u_buf[offset + 5] = a2[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 3;
        v_buf[offset] = b2[8 + i];
        v_buf[offset + 1] = b1[8 + i];
        v_buf[offset + 2] = b0[8 + i];
        v_buf[offset + 3] = a0[8 + i];
        v_buf[offset + 4] = a1[8 + i];
        v_buf[offset + 5] = a2[8 + i];
    }
}

// =============================================================================
// Simple filter (2-tap) — luma
// =============================================================================

/// Simple vertical filter: 16 pixels across a horizontal edge

#[rite]
pub(crate) fn simple_v_filter16_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    let (p1, p0, q0, q1) = load_16x4_neon(_token, buf, point, stride);
    let mask = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    let (op0, oq0) = do_filter2_neon(_token, p1, p0, q0, q1, mask);
    store_16x2_neon(_token, buf, point, stride, op0, oq0);
}

/// Simple horizontal filter: 16 rows across a vertical edge

#[rite]
pub(crate) fn simple_h_filter16_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    let (p1, p0, q0, q1) = load_4x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    let (op0, oq0) = do_filter2_neon(_token, p1, p0, q0, q1, mask);
    store_2x16_neon(_token, op0, oq0, buf, x0, y_start, stride);
}

// =============================================================================
// Normal filter (4-tap inner, 6-tap edge) — luma
// =============================================================================

/// Normal vertical filter for subblock edge (inner edge, 4-tap)

#[rite]
pub(crate) fn normal_v_filter16_inner_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_16x8_neon(_token, buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);

    // Store 4 rows around the edge
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 2 * stride..][..16]).unwrap(),
        op1,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        op0,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        oq0,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + stride..][..16]).unwrap(),
        oq1,
    );
}

/// Normal vertical filter for macroblock edge (6-tap)

#[rite]
pub(crate) fn normal_v_filter16_edge_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_16x8_neon(_token, buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);

    // Store 6 rows
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 3 * stride..][..16]).unwrap(),
        op2,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 2 * stride..][..16]).unwrap(),
        op1,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        op0,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        oq0,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + stride..][..16]).unwrap(),
        oq1,
    );
    simd_mem_neon::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + 2 * stride..][..16]).unwrap(),
        oq2,
    );
}

/// Normal horizontal filter for subblock edge (inner edge, 4-tap)

#[rite]
pub(crate) fn normal_h_filter16_inner_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_4x16_neon(_token, op1, op0, oq0, oq1, buf, x0, y_start, stride);
}

/// Normal horizontal filter for macroblock edge (6-tap)

#[rite]
pub(crate) fn normal_h_filter16_edge_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_6x16_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, buf, x0, y_start, stride,
    );
}

// =============================================================================
// Normal filter — chroma (UV packed as 16 pixels)
// =============================================================================

/// Normal vertical filter for UV macroblock edge (6-tap)

#[rite]
pub(crate) fn normal_v_filter_uv_edge_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x8x2_neon(_token, u_buf, v_buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_8x6x2_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, u_buf, v_buf, point, point, stride,
    );
}

/// Normal vertical filter for UV subblock edge (4-tap)

#[rite]
pub(crate) fn normal_v_filter_uv_inner_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x8x2_neon(_token, u_buf, v_buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_8x4x2_neon(
        _token, op1, op0, oq0, oq1, u_buf, v_buf, point, point, stride,
    );
}

/// Normal horizontal filter for UV macroblock edge (6-tap)

#[rite]
pub(crate) fn normal_h_filter_uv_edge_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) =
        load_8x8x2_h_neon(_token, u_buf, v_buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_6x8x2_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, u_buf, v_buf, x0, y_start, stride,
    );
}

/// Normal horizontal filter for UV subblock edge (4-tap)

#[rite]
pub(crate) fn normal_h_filter_uv_inner_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) =
        load_8x8x2_h_neon(_token, u_buf, v_buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_4x8x2_neon(
        _token, op1, op0, oq0, oq1, u_buf, v_buf, x0, y_start, stride,
    );
}

// ============================================================================
// WASM SIMD128 filter implementations (wasm32)
// ============================================================================

// =============================================================================
// Load/store helpers
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> v128 {
    u8x16(
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
        a[14], a[15],
    )
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_u8x16(out: &mut [u8; 16], v: v128) {
    out[0] = u8x16_extract_lane::<0>(v);
    out[1] = u8x16_extract_lane::<1>(v);
    out[2] = u8x16_extract_lane::<2>(v);
    out[3] = u8x16_extract_lane::<3>(v);
    out[4] = u8x16_extract_lane::<4>(v);
    out[5] = u8x16_extract_lane::<5>(v);
    out[6] = u8x16_extract_lane::<6>(v);
    out[7] = u8x16_extract_lane::<7>(v);
    out[8] = u8x16_extract_lane::<8>(v);
    out[9] = u8x16_extract_lane::<9>(v);
    out[10] = u8x16_extract_lane::<10>(v);
    out[11] = u8x16_extract_lane::<11>(v);
    out[12] = u8x16_extract_lane::<12>(v);
    out[13] = u8x16_extract_lane::<13>(v);
    out[14] = u8x16_extract_lane::<14>(v);
    out[15] = u8x16_extract_lane::<15>(v);
}

/// Load 16 bytes from a buffer row
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_row(buf: &[u8], offset: usize) -> v128 {
    load_u8x16(<&[u8; 16]>::try_from(&buf[offset..offset + 16]).unwrap())
}

/// Store 16 bytes to a buffer row
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_row(buf: &mut [u8], offset: usize, v: v128) {
    store_u8x16(
        <&mut [u8; 16]>::try_from(&mut buf[offset..offset + 16]).unwrap(),
        v,
    );
}

/// Load a column of 16 bytes (one byte per row, stride apart)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_col(buf: &[u8], base: usize, stride: usize) -> v128 {
    u8x16(
        buf[base],
        buf[base + stride],
        buf[base + stride * 2],
        buf[base + stride * 3],
        buf[base + stride * 4],
        buf[base + stride * 5],
        buf[base + stride * 6],
        buf[base + stride * 7],
        buf[base + stride * 8],
        buf[base + stride * 9],
        buf[base + stride * 10],
        buf[base + stride * 11],
        buf[base + stride * 12],
        buf[base + stride * 13],
        buf[base + stride * 14],
        buf[base + stride * 15],
    )
}

/// Store a column of 16 bytes
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_col(buf: &mut [u8], base: usize, stride: usize, v: v128) {
    buf[base] = u8x16_extract_lane::<0>(v);
    buf[base + stride] = u8x16_extract_lane::<1>(v);
    buf[base + stride * 2] = u8x16_extract_lane::<2>(v);
    buf[base + stride * 3] = u8x16_extract_lane::<3>(v);
    buf[base + stride * 4] = u8x16_extract_lane::<4>(v);
    buf[base + stride * 5] = u8x16_extract_lane::<5>(v);
    buf[base + stride * 6] = u8x16_extract_lane::<6>(v);
    buf[base + stride * 7] = u8x16_extract_lane::<7>(v);
    buf[base + stride * 8] = u8x16_extract_lane::<8>(v);
    buf[base + stride * 9] = u8x16_extract_lane::<9>(v);
    buf[base + stride * 10] = u8x16_extract_lane::<10>(v);
    buf[base + stride * 11] = u8x16_extract_lane::<11>(v);
    buf[base + stride * 12] = u8x16_extract_lane::<12>(v);
    buf[base + stride * 13] = u8x16_extract_lane::<13>(v);
    buf[base + stride * 14] = u8x16_extract_lane::<14>(v);
    buf[base + stride * 15] = u8x16_extract_lane::<15>(v);
}

// =============================================================================
// Core filter helpers
// =============================================================================

/// Absolute difference of u8x16
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn abd_u8x16(a: v128, b: v128) -> v128 {
    u8x16_sub(u8x16_max(a, b), u8x16_min(a, b))
}

/// NeedsFilter: returns mask where 2*|p0-q0| + |p1-q1|/2 <= thresh
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn needs_filter(p1: v128, p0: v128, q0: v128, q1: v128, thresh: i32) -> v128 {
    let thresh_v = u8x16_splat(thresh as u8);
    let a_p0_q0 = abd_u8x16(p0, q0);
    let a_p1_q1 = abd_u8x16(p1, q1);
    let a_p0_q0_2 = u8x16_add_sat(a_p0_q0, a_p0_q0);
    let a_p1_q1_2 = u8x16_shr(a_p1_q1, 1);
    let sum = u8x16_add_sat(a_p0_q0_2, a_p1_q1_2);
    u8x16_le(sum, thresh_v)
}

/// NeedsFilter2: extended check for normal filter (NeedsFilter AND all |adj| <= ithresh)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn needs_filter2(
    p3: v128,
    p2: v128,
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    q2: v128,
    q3: v128,
    ithresh: i32,
    thresh: i32,
) -> v128 {
    let it = u8x16_splat(ithresh as u8);
    let mask = needs_filter(p1, p0, q0, q1, thresh);
    let m1 = u8x16_le(abd_u8x16(p3, p2), it);
    let m2 = u8x16_le(abd_u8x16(p2, p1), it);
    let m3 = u8x16_le(abd_u8x16(p1, p0), it);
    let m4 = u8x16_le(abd_u8x16(q0, q1), it);
    let m5 = u8x16_le(abd_u8x16(q1, q2), it);
    let m6 = u8x16_le(abd_u8x16(q2, q3), it);
    v128_and(
        mask,
        v128_and(
            v128_and(m1, m2),
            v128_and(v128_and(m3, m4), v128_and(m5, m6)),
        ),
    )
}

/// HEV: high edge variance (|p1-p0| > thresh || |q1-q0| > thresh)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn hev(p1: v128, p0: v128, q0: v128, q1: v128, thresh: i32) -> v128 {
    let t = u8x16_splat(thresh as u8);
    let h1 = u8x16_gt(abd_u8x16(p1, p0), t);
    let h2 = u8x16_gt(abd_u8x16(q1, q0), t);
    v128_or(h1, h2)
}

/// DoFilter2: simple VP8 filter (modifies p0, q0)
/// a = clamp((p0 - q0 + 3*(q1 - p1) + 4) >> 3)  for q0
/// Also computes a2 = (a + 1) >> 1 for p0
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn do_filter2(p1: v128, p0: &mut v128, q0: &mut v128, q1: v128, mask: v128) {
    let sign = u8x16_splat(0x80);
    // Convert to signed domain
    let sp1 = v128_xor(p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(q1, sign);

    // a = 3*(q0 - p0) + satu8(p1 - q1)
    let a0 = i8x16_sub_sat(sp1, sq1); // clamp(p1 - q1)
    let a1 = i8x16_sub_sat(sq0, sp0); // clamp(q0 - p0)
    let a2 = i8x16_add_sat(a1, a1); // 2*(q0 - p0)
    let a3 = i8x16_add_sat(a2, a1); // 3*(q0 - p0)
    let a = i8x16_add_sat(a0, a3); // 3*(q0-p0) + clamp(p1-q1)
    let a_masked = v128_and(a, mask);

    // Filter1 = clamp(a + 4) >> 3
    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    // Filter2 = clamp(a + 3) >> 3
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    *q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    *p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);
}

/// DoFilter4: normal inner VP8 filter (modifies p1, p0, q0, q1)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn do_filter4(
    p1: &mut v128,
    p0: &mut v128,
    q0: &mut v128,
    q1: &mut v128,
    mask: v128,
    hev_mask: v128,
) {
    let sign = u8x16_splat(0x80);
    let sp1 = v128_xor(*p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(*q1, sign);

    let a0 = i8x16_sub_sat(sp1, sq1);
    let a0_hev = v128_and(a0, hev_mask); // only apply p1-q1 term where HEV
    let a1 = i8x16_sub_sat(sq0, sp0);
    let a2 = i8x16_add_sat(a1, a1);
    let a3 = i8x16_add_sat(a2, a1);
    let a = i8x16_add_sat(a0_hev, a3);
    let a_masked = v128_and(a, mask);

    // Filter1, Filter2
    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    let new_q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    let new_p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);

    // For non-HEV pixels, also adjust p1 and q1
    let f3 = i8x16_add_sat(f1, i8x16_splat(1));
    let f3 = i8x16_shr(f3, 1); // (f1 + 1) >> 1
    let not_hev = v128_not(hev_mask);
    let f3_masked = v128_and(f3, not_hev);

    *q0 = new_q0;
    *p0 = new_p0;
    *q1 = v128_xor(i8x16_sub_sat(sq1, f3_masked), sign);
    *p1 = v128_xor(i8x16_add_sat(sp1, f3_masked), sign);
}

/// DoFilter6: normal edge VP8 filter (modifies p2, p1, p0, q0, q1, q2)
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn do_filter6(
    p2: &mut v128,
    p1: &mut v128,
    p0: &mut v128,
    q0: &mut v128,
    q1: &mut v128,
    q2: &mut v128,
    mask: v128,
    hev_mask: v128,
) {
    let sign = u8x16_splat(0x80);
    let sp2 = v128_xor(*p2, sign);
    let sp1 = v128_xor(*p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(*q1, sign);
    let sq2 = v128_xor(*q2, sign);

    // For HEV pixels: same as simple filter (do_filter2 logic)
    let a0 = i8x16_sub_sat(sp1, sq1);
    let a1 = i8x16_sub_sat(sq0, sp0);
    let a2 = i8x16_add_sat(a1, a1);
    let a3 = i8x16_add_sat(a2, a1);
    let a = i8x16_add_sat(a0, a3);
    let a_masked = v128_and(a, mask);

    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    let hev_q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    let hev_p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);

    // For non-HEV pixels: wider filter using p2,p1,p0,q0,q1,q2
    // a = clamp(p0 - q0), then compute 27*a+63 >> 7, 18*a+63 >> 7, 9*a+63 >> 7
    let not_hev = v128_and(mask, v128_not(hev_mask));
    let w = i8x16_sub_sat(sp0, sq0); // clamp(p0 - q0) in signed domain

    // Widen to i16 for wider filter computation
    let w_lo = i16x8_extend_low_i8x16(w);
    let w_hi = i16x8_extend_high_i8x16(w);
    let round = i16x8_splat(63);

    // 27 * w
    let w27_lo = i16x8_mul(w_lo, i16x8_splat(27));
    let w27_hi = i16x8_mul(w_hi, i16x8_splat(27));
    let a_27_lo = i16x8_shr(i16x8_add(w27_lo, round), 7);
    let a_27_hi = i16x8_shr(i16x8_add(w27_hi, round), 7);
    let a27 = i8x16_narrow_i16x8(a_27_lo, a_27_hi);

    // 18 * w
    let w18_lo = i16x8_mul(w_lo, i16x8_splat(18));
    let w18_hi = i16x8_mul(w_hi, i16x8_splat(18));
    let a_18_lo = i16x8_shr(i16x8_add(w18_lo, round), 7);
    let a_18_hi = i16x8_shr(i16x8_add(w18_hi, round), 7);
    let a18 = i8x16_narrow_i16x8(a_18_lo, a_18_hi);

    // 9 * w
    let w9_lo = i16x8_mul(w_lo, i16x8_splat(9));
    let w9_hi = i16x8_mul(w_hi, i16x8_splat(9));
    let a_9_lo = i16x8_shr(i16x8_add(w9_lo, round), 7);
    let a_9_hi = i16x8_shr(i16x8_add(w9_hi, round), 7);
    let a9 = i8x16_narrow_i16x8(a_9_lo, a_9_hi);

    let wide_q0 = v128_xor(i8x16_sub_sat(sq0, a27), sign);
    let wide_p0 = v128_xor(i8x16_add_sat(sp0, a27), sign);
    let wide_q1 = v128_xor(i8x16_sub_sat(sq1, a18), sign);
    let wide_p1 = v128_xor(i8x16_add_sat(sp1, a18), sign);
    let wide_q2 = v128_xor(i8x16_sub_sat(sq2, a9), sign);
    let wide_p2 = v128_xor(i8x16_add_sat(sp2, a9), sign);

    // Select: HEV pixels use simple filter, non-HEV use wide filter
    *q0 = v128_bitselect(hev_q0, wide_q0, hev_mask);
    *p0 = v128_bitselect(hev_p0, wide_p0, hev_mask);
    *q1 = v128_bitselect(*q1, wide_q1, not_hev); // only modify non-HEV
    *p1 = v128_bitselect(*p1, wide_p1, not_hev);
    *q2 = v128_bitselect(*q2, wide_q2, not_hev);
    *p2 = v128_bitselect(*p2, wide_p2, not_hev);
}

// =============================================================================
// Simple filter (vertical/horizontal)
// =============================================================================

/// Simple vertical filter: processes horizontal edge at `point`, 16 columns wide
#[rite]
pub(crate) fn simple_v_filter16_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    let p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let q1 = load_row(buf, point + stride);

    let mask = needs_filter(p1, p0, q0, q1, thresh);
    do_filter2(p1, &mut p0, &mut q0, q1, mask);

    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
}

/// Simple horizontal filter: processes vertical edge at column x0, 16 rows
#[rite]
pub(crate) fn simple_h_filter16_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    let base = y_start * stride + x0;
    let p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let q1 = load_col(buf, base + 1, stride);

    let mask = needs_filter(p1, p0, q0, q1, thresh);
    do_filter2(p1, &mut p0, &mut q0, q1, mask);

    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
}

// =============================================================================
// Normal filter (vertical/horizontal, inner/edge)
// =============================================================================

/// Normal vertical inner filter (subblock edge)
#[rite]
pub(crate) fn normal_v_filter16_inner_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_row(buf, point - 4 * stride);
    let p2 = load_row(buf, point - 3 * stride);
    let mut p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let mut q1 = load_row(buf, point + stride);
    let q2 = load_row(buf, point + 2 * stride);
    let q3 = load_row(buf, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_row(buf, point - 2 * stride, p1);
    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
    store_row(buf, point + stride, q1);
}

/// Normal vertical edge filter (macroblock edge)
#[rite]
pub(crate) fn normal_v_filter16_edge_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_row(buf, point - 4 * stride);
    let mut p2 = load_row(buf, point - 3 * stride);
    let mut p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let mut q1 = load_row(buf, point + stride);
    let mut q2 = load_row(buf, point + 2 * stride);
    let q3 = load_row(buf, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_row(buf, point - 3 * stride, p2);
    store_row(buf, point - 2 * stride, p1);
    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
    store_row(buf, point + stride, q1);
    store_row(buf, point + 2 * stride, q2);
}

/// Normal horizontal inner filter (subblock edge): vertical edge at x0, 16 rows
#[rite]
pub(crate) fn normal_h_filter16_inner_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let base = y_start * stride + x0;
    let p3 = load_col(buf, base - 4, stride);
    let p2 = load_col(buf, base - 3, stride);
    let mut p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let mut q1 = load_col(buf, base + 1, stride);
    let q2 = load_col(buf, base + 2, stride);
    let q3 = load_col(buf, base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_col(buf, base - 2, stride, p1);
    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
    store_col(buf, base + 1, stride, q1);
}

/// Normal horizontal edge filter (macroblock edge): vertical edge at x0, 16 rows
#[rite]
pub(crate) fn normal_h_filter16_edge_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let base = y_start * stride + x0;
    let p3 = load_col(buf, base - 4, stride);
    let mut p2 = load_col(buf, base - 3, stride);
    let mut p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let mut q1 = load_col(buf, base + 1, stride);
    let mut q2 = load_col(buf, base + 2, stride);
    let q3 = load_col(buf, base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_col(buf, base - 3, stride, p2);
    store_col(buf, base - 2, stride, p1);
    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
    store_col(buf, base + 1, stride, q1);
    store_col(buf, base + 2, stride, q2);
}

// =============================================================================
// UV (chroma) filters: pack U+V 8-pixel halves into 16-wide registers
// =============================================================================

/// Load 8 bytes from U and 8 bytes from V, packed into one u8x16
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_uv_col(u_buf: &[u8], v_buf: &[u8], u_base: usize, v_base: usize, stride: usize) -> v128 {
    u8x16(
        u_buf[u_base],
        u_buf[u_base + stride],
        u_buf[u_base + stride * 2],
        u_buf[u_base + stride * 3],
        u_buf[u_base + stride * 4],
        u_buf[u_base + stride * 5],
        u_buf[u_base + stride * 6],
        u_buf[u_base + stride * 7],
        v_buf[v_base],
        v_buf[v_base + stride],
        v_buf[v_base + stride * 2],
        v_buf[v_base + stride * 3],
        v_buf[v_base + stride * 4],
        v_buf[v_base + stride * 5],
        v_buf[v_base + stride * 6],
        v_buf[v_base + stride * 7],
    )
}

/// Store 8 bytes to U and 8 bytes to V from packed u8x16
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_uv_col(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_base: usize,
    v_base: usize,
    stride: usize,
    v: v128,
) {
    u_buf[u_base] = u8x16_extract_lane::<0>(v);
    u_buf[u_base + stride] = u8x16_extract_lane::<1>(v);
    u_buf[u_base + stride * 2] = u8x16_extract_lane::<2>(v);
    u_buf[u_base + stride * 3] = u8x16_extract_lane::<3>(v);
    u_buf[u_base + stride * 4] = u8x16_extract_lane::<4>(v);
    u_buf[u_base + stride * 5] = u8x16_extract_lane::<5>(v);
    u_buf[u_base + stride * 6] = u8x16_extract_lane::<6>(v);
    u_buf[u_base + stride * 7] = u8x16_extract_lane::<7>(v);
    v_buf[v_base] = u8x16_extract_lane::<8>(v);
    v_buf[v_base + stride] = u8x16_extract_lane::<9>(v);
    v_buf[v_base + stride * 2] = u8x16_extract_lane::<10>(v);
    v_buf[v_base + stride * 3] = u8x16_extract_lane::<11>(v);
    v_buf[v_base + stride * 4] = u8x16_extract_lane::<12>(v);
    v_buf[v_base + stride * 5] = u8x16_extract_lane::<13>(v);
    v_buf[v_base + stride * 6] = u8x16_extract_lane::<14>(v);
    v_buf[v_base + stride * 7] = u8x16_extract_lane::<15>(v);
}

/// Load UV row: 8 bytes from U row + 8 bytes from V row packed into u8x16
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_uv_row(u_buf: &[u8], v_buf: &[u8], u_off: usize, v_off: usize) -> v128 {
    u8x16(
        u_buf[u_off],
        u_buf[u_off + 1],
        u_buf[u_off + 2],
        u_buf[u_off + 3],
        u_buf[u_off + 4],
        u_buf[u_off + 5],
        u_buf[u_off + 6],
        u_buf[u_off + 7],
        v_buf[v_off],
        v_buf[v_off + 1],
        v_buf[v_off + 2],
        v_buf[v_off + 3],
        v_buf[v_off + 4],
        v_buf[v_off + 5],
        v_buf[v_off + 6],
        v_buf[v_off + 7],
    )
}

/// Store UV row
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_uv_row(u_buf: &mut [u8], v_buf: &mut [u8], u_off: usize, v_off: usize, v: v128) {
    u_buf[u_off] = u8x16_extract_lane::<0>(v);
    u_buf[u_off + 1] = u8x16_extract_lane::<1>(v);
    u_buf[u_off + 2] = u8x16_extract_lane::<2>(v);
    u_buf[u_off + 3] = u8x16_extract_lane::<3>(v);
    u_buf[u_off + 4] = u8x16_extract_lane::<4>(v);
    u_buf[u_off + 5] = u8x16_extract_lane::<5>(v);
    u_buf[u_off + 6] = u8x16_extract_lane::<6>(v);
    u_buf[u_off + 7] = u8x16_extract_lane::<7>(v);
    v_buf[v_off] = u8x16_extract_lane::<8>(v);
    v_buf[v_off + 1] = u8x16_extract_lane::<9>(v);
    v_buf[v_off + 2] = u8x16_extract_lane::<10>(v);
    v_buf[v_off + 3] = u8x16_extract_lane::<11>(v);
    v_buf[v_off + 4] = u8x16_extract_lane::<12>(v);
    v_buf[v_off + 5] = u8x16_extract_lane::<13>(v);
    v_buf[v_off + 6] = u8x16_extract_lane::<14>(v);
    v_buf[v_off + 7] = u8x16_extract_lane::<15>(v);
}

/// Normal vertical UV edge filter
#[rite]
pub(crate) fn normal_v_filter_uv_edge_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_uv_row(u_buf, v_buf, point - 4 * stride, point - 4 * stride);
    let mut p2 = load_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride);
    let mut p1 = load_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride);
    let mut p0 = load_uv_row(u_buf, v_buf, point - stride, point - stride);
    let mut q0 = load_uv_row(u_buf, v_buf, point, point);
    let mut q1 = load_uv_row(u_buf, v_buf, point + stride, point + stride);
    let mut q2 = load_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride);
    let q3 = load_uv_row(u_buf, v_buf, point + 3 * stride, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride, p2);
    store_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride, p1);
    store_uv_row(u_buf, v_buf, point - stride, point - stride, p0);
    store_uv_row(u_buf, v_buf, point, point, q0);
    store_uv_row(u_buf, v_buf, point + stride, point + stride, q1);
    store_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride, q2);
}

/// Normal vertical UV inner filter
#[rite]
pub(crate) fn normal_v_filter_uv_inner_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_uv_row(u_buf, v_buf, point - 4 * stride, point - 4 * stride);
    let p2 = load_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride);
    let mut p1 = load_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride);
    let mut p0 = load_uv_row(u_buf, v_buf, point - stride, point - stride);
    let mut q0 = load_uv_row(u_buf, v_buf, point, point);
    let mut q1 = load_uv_row(u_buf, v_buf, point + stride, point + stride);
    let q2 = load_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride);
    let q3 = load_uv_row(u_buf, v_buf, point + 3 * stride, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride, p1);
    store_uv_row(u_buf, v_buf, point - stride, point - stride, p0);
    store_uv_row(u_buf, v_buf, point, point, q0);
    store_uv_row(u_buf, v_buf, point + stride, point + stride, q1);
}

/// Normal horizontal UV edge filter
#[rite]
pub(crate) fn normal_h_filter_uv_edge_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let u_base = y_start * stride + x0;
    let v_base = y_start * stride + x0;
    let p3 = load_uv_col(u_buf, v_buf, u_base - 4, v_base - 4, stride);
    let mut p2 = load_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride);
    let mut p1 = load_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride);
    let mut p0 = load_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride);
    let mut q0 = load_uv_col(u_buf, v_buf, u_base, v_base, stride);
    let mut q1 = load_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride);
    let mut q2 = load_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride);
    let q3 = load_uv_col(u_buf, v_buf, u_base + 3, v_base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride, p2);
    store_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride, p1);
    store_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride, p0);
    store_uv_col(u_buf, v_buf, u_base, v_base, stride, q0);
    store_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride, q1);
    store_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride, q2);
}

/// Normal horizontal UV inner filter
#[rite]
pub(crate) fn normal_h_filter_uv_inner_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let u_base = y_start * stride + x0;
    let v_base = y_start * stride + x0;
    let p3 = load_uv_col(u_buf, v_buf, u_base - 4, v_base - 4, stride);
    let p2 = load_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride);
    let mut p1 = load_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride);
    let mut p0 = load_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride);
    let mut q0 = load_uv_col(u_buf, v_buf, u_base, v_base, stride);
    let mut q1 = load_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride);
    let q2 = load_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride);
    let q3 = load_uv_col(u_buf, v_buf, u_base + 3, v_base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride, p1);
    store_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride, p0);
    store_uv_col(u_buf, v_buf, u_base, v_base, stride, q0);
    store_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride, q1);
}

// ============================================================================
// Dispatch: precomputed filter params and #[arcane] entry points
// ============================================================================

/// Precomputed filter parameters for a single macroblock.
/// Computed by `calculate_filter_parameters` before entering the filter loop.
#[derive(Clone, Copy)]
pub(crate) struct MbFilterParams {
    pub filter_level: u8,
    pub interior_limit: u8,
    pub hev_threshold: u8,
    pub mbedge_limit: u8,
    pub sub_bedge_limit: u8,
    pub do_subblock_filtering: bool,
}

// ============================================================================
// Scalar fallback filter functions (used when no SIMD token is available)
// ============================================================================

/// Apply simple horizontal filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn simple_filter_horizontal_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let y0 = y_start + y;
        simple_segment_horizontal(edge_limit, &mut buf[y0 * stride + x0 - 4..][..8]);
    }
}

/// Apply simple vertical filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn simple_filter_vertical_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        simple_segment_vertical(edge_limit, buf, point, stride);
    }
}

/// Apply normal vertical macroblock filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_mb_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Apply normal vertical subblock filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_sub_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Apply normal horizontal macroblock filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_mb_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let row = y_start + y;
        macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal subblock filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_sub_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let row = y_start + y;
        subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal macroblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..8 {
        let row = y_start + y;
        macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal subblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..8 {
        let row = y_start + y;
        subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal vertical macroblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

/// Apply normal vertical subblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

// ============================================================================
// Single #[arcane] entry point for the entire filter-row loop.
//
// All arch-specific filter functions are #[rite] (target_feature + inline),
// so they get inlined into this single #[arcane] boundary. This eliminates
// the per-call dispatch overhead (~7.6M instructions per decode for a
// 1024x1024 image).
// ============================================================================

/// Filter a full row of macroblocks using SIMD, with a single target_feature boundary.
///
/// All `#[rite]` filter functions inline into this one target_feature region,
/// eliminating per-call dispatch overhead.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::X64V3Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        // Filter across left of macroblock (horizontal filter on vertical edge)
        if mbx > 0 {
            if filter_type {
                simple_h_filter16(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                normal_h_filter16_edge(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                normal_h_filter_uv_edge(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        // Filter across vertical subblocks
        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    simple_h_filter16(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                // Use fused 3-edge horizontal filter for luma subblocks
                normal_h_filter16i(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
                normal_h_filter_uv_inner(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        // Filter across top of macroblock (vertical filter on horizontal edge)
        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                simple_v_filter16(_token, cache_y, point, cache_y_stride, mbedge_limit_i);
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                normal_v_filter16_edge(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_edge(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        // Filter across horizontal subblock edges
        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    simple_v_filter16(_token, cache_y, point, cache_y_stride, sub_bedge_limit_i);
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    normal_v_filter16_inner(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_inner(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}

/// NEON filter row entry point.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::NeonToken,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        if mbx > 0 {
            if filter_type {
                simple_h_filter16_neon(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                normal_h_filter16_edge_neon(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                normal_h_filter_uv_edge_neon(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    simple_h_filter16_neon(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for x in (4usize..16 - 3).step_by(4) {
                    normal_h_filter16_inner_neon(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                normal_h_filter_uv_inner_neon(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                simple_v_filter16_neon(_token, cache_y, point, cache_y_stride, mbedge_limit_i);
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                normal_v_filter16_edge_neon(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_edge_neon(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    simple_v_filter16_neon(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    normal_v_filter16_inner_neon(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_inner_neon(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}

/// WASM SIMD128 filter row entry point.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::Wasm128Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        if mbx > 0 {
            if filter_type {
                simple_h_filter16_wasm(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                normal_h_filter16_edge_wasm(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                normal_h_filter_uv_edge_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    simple_h_filter16_wasm(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for x in (4usize..16 - 3).step_by(4) {
                    normal_h_filter16_inner_wasm(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                normal_h_filter_uv_inner_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                simple_v_filter16_wasm(_token, cache_y, point, cache_y_stride, mbedge_limit_i);
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                normal_v_filter16_edge_wasm(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_edge_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    simple_v_filter16_wasm(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    normal_v_filter16_inner_wasm(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                normal_v_filter_uv_inner_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}
