//! Utilities for doing the YUV -> RGB conversion
//! The images are encoded in the Y'CbCr format as detailed here: <https://en.wikipedia.org/wiki/YCbCr>
//! so need to be converted to RGB to be displayed

// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]
//! To do the YUV -> RGB conversion we need to first decide how to map the yuv values to the pixels
//! The y buffer is the same size as the pixel buffer so that maps 1-1 but the
//! u and v buffers are half the size of the pixel buffer so we need to scale it up
//! The simple way to upscale is just to take each u/v value and associate it with the 4
//! pixels around it e.g. for a 4x4 image:
//!
//! ||||||
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! ||||||
//!
//! |||||||
//! |uu|vv|
//! |uu|vv|
//! |||||||
//!
//! Then each of the 2x2 pixels would match the u/v from the same quadrant
//!
//! However fancy upsampling is the default for libwebp which does a little more work to make the values smoother
//! It interpolates u and v so that for e.g. the pixel 1 down and 1 from the left the u value
//! would be (9*u0 + 3*u1 + 3*u2 + u3 + 8) / 16 and similar for the other pixels
//! The edges are mirrored, so for the pixel 1 down and 0 from the left it uses (9*u0 + 3*u2 + 3*u0 + u2 + 8) / 16

use alloc::vec;
use alloc::vec::Vec;

use archmage::prelude::*;

#[cfg(target_arch = "aarch64")]
use archmage::intrinsics::aarch64 as simd_mem;
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem;

/// `_mm_mulhi_epu16` emulation
fn mulhi(v: u8, coeff: u16) -> i32 {
    ((u32::from(v) * u32::from(coeff)) >> 8) as i32
}

/// This function has been rewritten to encourage auto-vectorization.
///
/// Based on [src/dsp/yuv.h](https://github.com/webmproject/libwebp/blob/8534f53960befac04c9631e6e50d21dcb42dfeaf/src/dsp/yuv.h#L79)
/// from the libwebp source.
/// ```text
/// const YUV_FIX2: i32 = 6;
/// const YUV_MASK2: i32 = (256 << YUV_FIX2) - 1;
/// fn clip(v: i32) -> u8 {
///     if (v & !YUV_MASK2) == 0 {
///         (v >> YUV_FIX2) as u8
///     } else if v < 0 {
///         0
///     } else {
///         255
///     }
/// }
/// ```
// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
fn clip(v: i32) -> u8 {
    const YUV_FIX2: i32 = 6;
    (v >> YUV_FIX2).max(0).min(255) as u8
}

#[inline(always)]
pub(crate) fn yuv_to_r(y: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234)
}

#[inline(always)]
pub(crate) fn yuv_to_g(y: u8, u: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708)
}

#[inline(always)]
pub(crate) fn yuv_to_b(y: u8, u: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685)
}

#[inline]
pub(crate) fn get_fancy_chroma_value(main: u8, secondary1: u8, secondary2: u8, tertiary: u8) -> u8 {
    let val0 = u16::from(main);
    let val1 = u16::from(secondary1);
    let val2 = u16::from(secondary2);
    let val3 = u16::from(tertiary);
    ((9 * val0 + 3 * val1 + 3 * val2 + val3 + 8) / 16) as u8
}

#[inline]
#[allow(dead_code)]
pub(crate) fn set_pixel(rgb: &mut [u8], y: u8, u: u8, v: u8) {
    rgb[0] = yuv_to_r(y, v);
    rgb[1] = yuv_to_g(y, u, v);
    rgb[2] = yuv_to_b(y, u);
}

/// Simple conversion, not currently used but could add a config to allow for using the simple
#[allow(unused)]
pub(crate) fn fill_rgb_buffer_simple<const BPP: usize>(
    buffer: &mut [u8],
    y_buffer: &[u8],
    u_buffer: &[u8],
    v_buffer: &[u8],
    width: usize,
    chroma_width: usize,
    buffer_width: usize,
) {
    let u_row_twice_iter = u_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| core::iter::repeat_n(n, 2));
    let v_row_twice_iter = v_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| core::iter::repeat_n(n, 2));

    for (((row, y_row), u_row), v_row) in buffer
        .chunks_exact_mut(width * BPP)
        .zip(y_buffer.chunks_exact(buffer_width))
        .zip(u_row_twice_iter)
        .zip(v_row_twice_iter)
    {
        fill_rgba_row_simple::<BPP>(
            &y_row[..width],
            &u_row[..chroma_width],
            &v_row[..chroma_width],
            row,
        );
    }
}

fn fill_rgba_row_simple<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Use SIMD for RGB (BPP=3) if available and row is wide enough
    if BPP == 3 && y_vec.len() >= 8 {
        incant!(
            fill_rgba_row_simple_dispatch(y_vec, u_vec, v_vec, rgba),
            [v3, neon, wasm128, scalar]
        );
        return;
    }
    fill_rgba_row_simple_scalar::<BPP>(y_vec, u_vec, v_vec, rgba);
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn fill_rgba_row_simple_dispatch_v3(
    _token: X64V3Token,
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    fill_rgba_row_simple_simd::<3>(y_vec, u_vec, v_vec, rgba);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fill_rgba_row_simple_dispatch_neon(
    token: NeonToken,
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    yuv420_to_rgb_row_neon(token, y_vec, u_vec, v_vec, rgba);
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn fill_rgba_row_simple_dispatch_wasm128(
    token: Wasm128Token,
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    yuv420_to_rgb_row_wasm(token, y_vec, u_vec, v_vec, rgba);
}

#[inline(always)]
fn fill_rgba_row_simple_dispatch_scalar(
    _token: ScalarToken,
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    fill_rgba_row_simple_scalar::<3>(y_vec, u_vec, v_vec, rgba);
}

#[cfg(target_arch = "x86_64")]
fn fill_rgba_row_simple_simd<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Use the new row-based SIMD function that processes 32 pixels at a time
    yuv420_to_rgb_row(y_vec, u_vec, v_vec, rgba);
}

fn fill_rgba_row_simple_scalar<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Fill 2 pixels per iteration: these pixels share `u` and `v` components
    let mut rgb_chunks = rgba.chunks_exact_mut(BPP * 2);
    let mut y_chunks = y_vec.chunks_exact(2);
    let mut u_iter = u_vec.iter();
    let mut v_iter = v_vec.iter();

    for (((rgb, y), &u), &v) in (&mut rgb_chunks)
        .zip(&mut y_chunks)
        .zip(&mut u_iter)
        .zip(&mut v_iter)
    {
        let coeffs = [
            mulhi(v, 26149),
            mulhi(u, 6419),
            mulhi(v, 13320),
            mulhi(u, 33050),
        ];

        let get_r = |y: u8| clip(mulhi(y, 19077) + coeffs[0] - 14234);
        let get_g = |y: u8| clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
        let get_b = |y: u8| clip(mulhi(y, 19077) + coeffs[3] - 17685);

        let rgb1 = &mut rgb[0..3];
        rgb1[0] = get_r(y[0]);
        rgb1[1] = get_g(y[0]);
        rgb1[2] = get_b(y[0]);

        let rgb2 = &mut rgb[BPP..];
        rgb2[0] = get_r(y[1]);
        rgb2[1] = get_g(y[1]);
        rgb2[2] = get_b(y[1]);
    }

    let remainder = rgb_chunks.into_remainder();
    if remainder.len() >= 3
        && let (Some(&y), Some(&u), Some(&v)) = (
            y_chunks.remainder().iter().next(),
            u_iter.next(),
            v_iter.next(),
        )
    {
        let coeffs = [
            mulhi(v, 26149),
            mulhi(u, 6419),
            mulhi(v, 13320),
            mulhi(u, 33050),
        ];

        remainder[0] = clip(mulhi(y, 19077) + coeffs[0] - 14234);
        remainder[1] = clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
        remainder[2] = clip(mulhi(y, 19077) + coeffs[3] - 17685);
    }
}

// constants used for yuv -> rgb conversion, using ones from libwebp
const YUV_FIX: i32 = 16;
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

// ---------------------------------------------------------------------------
// Gamma-corrected chroma downsampling tables (from libwebp picture_csp_enc.c)
//
// libwebp uses gamma = 0.80 (not sRGB) to compensate for resolution loss
// during 4:2:0 chroma subsampling. Each RGB channel is converted to a
// linearish space before averaging, then converted back.
//
// GAMMA_FIX=12 => kGammaScale = 4095, GAMMA_TAB_FIX=7, GAMMA_TAB_SIZE=32.
// Forward: GammaToLinear[v] = round(pow(v/255, 0.80) * 4095)
// Inverse: LinearToGamma[i] = round(255 * pow(i * 128/4095, 1.25))

/// sRGB byte -> linear^0.80 (scale 0..4095). 256 entries.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) const GAMMA_TO_LINEAR_TAB: [u16; 256] = [
    0, 49, 85, 117, 147, 176, 204, 231, 257, 282, 307, 331, 355, 379, 402, 425, 447, 469, 491, 513,
    534, 556, 577, 598, 618, 639, 659, 679, 699, 719, 739, 759, 778, 798, 817, 836, 855, 874, 893,
    912, 930, 949, 967, 986, 1004, 1022, 1040, 1059, 1077, 1094, 1112, 1130, 1148, 1165, 1183,
    1200, 1218, 1235, 1252, 1270, 1287, 1304, 1321, 1338, 1355, 1372, 1389, 1406, 1422, 1439, 1456,
    1472, 1489, 1505, 1522, 1538, 1555, 1571, 1587, 1604, 1620, 1636, 1652, 1668, 1684, 1700, 1716,
    1732, 1748, 1764, 1780, 1796, 1812, 1827, 1843, 1859, 1874, 1890, 1905, 1921, 1937, 1952, 1967,
    1983, 1998, 2014, 2029, 2044, 2059, 2075, 2090, 2105, 2120, 2135, 2151, 2166, 2181, 2196, 2211,
    2226, 2241, 2256, 2270, 2285, 2300, 2315, 2330, 2345, 2359, 2374, 2389, 2403, 2418, 2433, 2447,
    2462, 2477, 2491, 2506, 2520, 2535, 2549, 2564, 2578, 2592, 2607, 2621, 2636, 2650, 2664, 2679,
    2693, 2707, 2721, 2736, 2750, 2764, 2778, 2792, 2806, 2820, 2835, 2849, 2863, 2877, 2891, 2905,
    2919, 2933, 2947, 2961, 2975, 2988, 3002, 3016, 3030, 3044, 3058, 3072, 3085, 3099, 3113, 3127,
    3140, 3154, 3168, 3182, 3195, 3209, 3222, 3236, 3250, 3263, 3277, 3291, 3304, 3318, 3331, 3345,
    3358, 3372, 3385, 3399, 3412, 3426, 3439, 3452, 3466, 3479, 3493, 3506, 3519, 3533, 3546, 3559,
    3573, 3586, 3599, 3612, 3626, 3639, 3652, 3665, 3678, 3692, 3705, 3718, 3731, 3744, 3757, 3771,
    3784, 3797, 3810, 3823, 3836, 3849, 3862, 3875, 3888, 3901, 3914, 3927, 3940, 3953, 3966, 3979,
    3992, 4005, 4018, 4031, 4044, 4056, 4069, 4082, 4095,
];

/// Linear^0.80 -> gamma-space byte. 33 entries (GAMMA_TAB_SIZE + 1).
/// Indexed at steps of 128 across the [0..4095] linear range.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) const LINEAR_TO_GAMMA_TAB: [u8; 33] = [
    0, 3, 8, 13, 19, 25, 31, 38, 45, 52, 60, 67, 75, 83, 91, 99, 107, 116, 124, 133, 142, 151, 160,
    169, 178, 187, 197, 206, 216, 226, 235, 245, 255,
];

/// Convert an sRGB byte to linear^0.80 fixed-point (0..4095).
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn gamma_to_linear(v: u8) -> u32 {
    GAMMA_TO_LINEAR_TAB[v as usize] as u32
}

/// Convert a linear^0.80 value (0..4095) back to an sRGB byte (0..255)
/// using interpolation in the inverse table.
///
/// The table has 33 entries at steps of 128. We interpolate between adjacent
/// entries using the 7-bit fractional part.
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn linear_to_gamma(v: u32) -> u8 {
    let tab_idx = (v >> 7) as usize; // 0..32
    let frac = v & 0x7F; // 7-bit fraction
    let v0 = LINEAR_TO_GAMMA_TAB[tab_idx] as u32;
    let v1 = LINEAR_TO_GAMMA_TAB[tab_idx + 1] as u32;
    ((v0 * (128 - frac) + v1 * frac + 64) >> 7) as u8
}

/// Average 4 byte values in gamma-corrected (linear^0.80) space.
/// Converts to linear, averages, converts back to sRGB byte.
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn gamma_avg_4(a: u8, b: u8, c: u8, d: u8) -> u8 {
    let sum = gamma_to_linear(a) + gamma_to_linear(b) + gamma_to_linear(c) + gamma_to_linear(d);
    // Integer division by 4 with rounding
    linear_to_gamma((sum + 2) >> 2)
}

/// Average 2 byte values in gamma-corrected (linear^0.80) space.
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn gamma_avg_2(a: u8, b: u8) -> u8 {
    let sum = gamma_to_linear(a) + gamma_to_linear(b);
    linear_to_gamma((sum + 1) >> 1)
}

/// converts the whole image to yuv data and adds values on the end to make it match the macroblock sizes
/// downscales the u/v data as well so it's half the width and height of the y data
// zenyuv path replaces this in production; kept for test_helpers parity testing.
#[allow(dead_code)]
pub(crate) fn convert_image_yuv<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Process pairs of rows for chroma downsampling (2x2 averaging)
    let row_pairs = height / 2;
    let odd_height = height & 1 != 0;
    let col_pairs = width / 2;
    let odd_width = width & 1 != 0;

    for row_pair in 0..row_pairs {
        let src_row1 = row_pair * 2;
        let src_row2 = src_row1 + 1;
        let chroma_row = row_pair;

        // Process pairs of columns
        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            // Get 4 RGB pixels for 2x2 block
            let rgb1 = &image_data[(src_row1 * stride + src_col1) * BPP..][..BPP];
            let rgb2 = &image_data[(src_row1 * stride + src_col2) * BPP..][..BPP];
            let rgb3 = &image_data[(src_row2 * stride + src_col1) * BPP..][..BPP];
            let rgb4 = &image_data[(src_row2 * stride + src_col2) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row1 * luma_width + src_col1] = rgb_to_y(rgb1);
            y_bytes[src_row1 * luma_width + src_col2] = rgb_to_y(rgb2);
            y_bytes[src_row2 * luma_width + src_col1] = rgb_to_y(rgb3);
            y_bytes[src_row2 * luma_width + src_col2] = rgb_to_y(rgb4);

            // Convert to U/V with gamma-corrected chroma downsampling
            let (u, v) = gamma_downsample_uv_4(rgb1, rgb2, rgb3, rgb4);
            u_bytes[chroma_row * chroma_width + chroma_col] = u;
            v_bytes[chroma_row * chroma_width + chroma_col] = v;
        }

        // Handle last column if width is odd
        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            // Get 2 RGB pixels (vertical pair for odd-width edge)
            let rgb1 = &image_data[(src_row1 * stride + src_col) * BPP..][..BPP];
            let rgb3 = &image_data[(src_row2 * stride + src_col) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row1 * luma_width + src_col] = rgb_to_y(rgb1);
            y_bytes[src_row2 * luma_width + src_col] = rgb_to_y(rgb3);

            // Convert to U/V with gamma-corrected averaging of 2 pixels
            let (u, v) = gamma_downsample_uv_2(rgb1, rgb3);
            u_bytes[chroma_row * chroma_width + chroma_col] = u;
            v_bytes[chroma_row * chroma_width + chroma_col] = v;
        }
    }

    // Handle last row if height is odd
    if odd_height {
        let src_row = height - 1;
        let chroma_row = row_pairs;

        // Process pairs of columns
        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            // Get 2 RGB pixels (horizontal pair for odd-height edge)
            let rgb1 = &image_data[(src_row * stride + src_col1) * BPP..][..BPP];
            let rgb2 = &image_data[(src_row * stride + src_col2) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row * luma_width + src_col1] = rgb_to_y(rgb1);
            y_bytes[src_row * luma_width + src_col2] = rgb_to_y(rgb2);

            // Convert to U/V with gamma-corrected averaging of 2 pixels
            let (u, v) = gamma_downsample_uv_2(rgb1, rgb2);
            u_bytes[chroma_row * chroma_width + chroma_col] = u;
            v_bytes[chroma_row * chroma_width + chroma_col] = v;
        }

        // Handle corner case: both width and height are odd
        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            // Single pixel — no averaging needed, just convert directly
            let rgb = &image_data[(src_row * stride + src_col) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row * luma_width + src_col] = rgb_to_y(rgb);

            // Convert to U/V (single pixel, no downsampling needed)
            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_single(rgb[0], rgb[1], rgb[2]);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_single(rgb[0], rgb[1], rgb[2]);
        }
    }

    // Replicate edge pixels to fill macroblock padding
    // Horizontal padding for Y
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }

    // Vertical padding for Y (including horizontal padding area)
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }

    // Horizontal padding for U/V
    let chroma_height = height.div_ceil(2);
    let actual_chroma_width = width.div_ceil(2);
    for y in 0..chroma_height {
        let last_u = u_bytes[y * chroma_width + actual_chroma_width - 1];
        let last_v = v_bytes[y * chroma_width + actual_chroma_width - 1];
        for x in actual_chroma_width..chroma_width {
            u_bytes[y * chroma_width + x] = last_u;
            v_bytes[y * chroma_width + x] = last_v;
        }
    }

    // Vertical padding for U/V
    for y in chroma_height..(mb_height * 8) {
        for x in 0..chroma_width {
            u_bytes[y * chroma_width + x] = u_bytes[(chroma_height - 1) * chroma_width + x];
            v_bytes[y * chroma_width + x] = v_bytes[(chroma_height - 1) * chroma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

pub(crate) fn convert_image_y<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let u_bytes = vec![127u8; chroma_size];
    let v_bytes = vec![127u8; chroma_size];

    // Process all source rows
    for y in 0..height {
        let src_row = &image_data[y * stride * BPP..y * stride * BPP + width * BPP];
        for x in 0..width {
            y_bytes[y * luma_width + x] = src_row[x * BPP];
        }
    }

    // Replicate edge pixels to fill macroblock padding
    // Horizontal padding for Y
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }

    // Vertical padding for Y (including horizontal padding area)
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

// values come from libwebp
// Y = 0.2568 * R + 0.5041 * G + 0.0979 * B + 16
// U = -0.1482 * R - 0.2910 * G + 0.4392 * B + 128
// V = 0.4392 * R - 0.3678 * G - 0.0714 * B + 128

// this is converted to 16 bit fixed point by multiplying by 2^16
// and shifting back

// Encoder-only functions below (used when std feature is enabled)
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_y(rgb: &[u8]) -> u8 {
    let luma = 16839 * i32::from(rgb[0]) + 33059 * i32::from(rgb[1]) + 6420 * i32::from(rgb[2]);
    ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
}

/// Compute U from a single pixel (no averaging).
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_u_single(r: u8, g: u8, b: u8) -> u8 {
    let u = -9719 * i32::from(r) - 19081 * i32::from(g) + 28800 * i32::from(b) + (128 << YUV_FIX);
    ((u + YUV_HALF) >> YUV_FIX) as u8
}

/// Compute V from a single pixel (no averaging).
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_v_single(r: u8, g: u8, b: u8) -> u8 {
    let v = 28800 * i32::from(r) - 24116 * i32::from(g) - 4684 * i32::from(b) + (128 << YUV_FIX);
    ((v + YUV_HALF) >> YUV_FIX) as u8
}

/// Get the chroma-downsampled U value for a 2x2 pixel block using
/// gamma-corrected averaging (gamma=0.80, matching libwebp).
///
/// Each R/G/B channel is averaged in linear^0.80 space before the YUV
/// matrix is applied to the averaged RGB values.
#[allow(dead_code)] // Used by codec.rs (behind `zencodec` feature)
pub(crate) fn rgb_to_u_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let r = gamma_avg_4(rgb1[0], rgb2[0], rgb3[0], rgb4[0]);
    let g = gamma_avg_4(rgb1[1], rgb2[1], rgb3[1], rgb4[1]);
    let b = gamma_avg_4(rgb1[2], rgb2[2], rgb3[2], rgb4[2]);
    rgb_to_u_single(r, g, b)
}

/// Get the chroma-downsampled V value for a 2x2 pixel block using
/// gamma-corrected averaging (gamma=0.80, matching libwebp).
#[allow(dead_code)] // Used by codec.rs (behind `zencodec` feature)
pub(crate) fn rgb_to_v_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let r = gamma_avg_4(rgb1[0], rgb2[0], rgb3[0], rgb4[0]);
    let g = gamma_avg_4(rgb1[1], rgb2[1], rgb3[1], rgb4[1]);
    let b = gamma_avg_4(rgb1[2], rgb2[2], rgb3[2], rgb4[2]);
    rgb_to_v_single(r, g, b)
}

/// Compute gamma-corrected U/V for a 2x2 pixel block (4 pixels).
/// Returns (u, v) with each R/G/B channel averaged in linear^0.80 space.
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn gamma_downsample_uv_4(p1: &[u8], p2: &[u8], p3: &[u8], p4: &[u8]) -> (u8, u8) {
    let r = gamma_avg_4(p1[0], p2[0], p3[0], p4[0]);
    let g = gamma_avg_4(p1[1], p2[1], p3[1], p4[1]);
    let b = gamma_avg_4(p1[2], p2[2], p3[2], p4[2]);
    (rgb_to_u_single(r, g, b), rgb_to_v_single(r, g, b))
}

/// Compute gamma-corrected U/V for a 1x2 or 2x1 pixel pair.
/// Returns (u, v) with each R/G/B channel averaged in linear^0.80 space.
#[inline(always)]
#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn gamma_downsample_uv_2(p1: &[u8], p2: &[u8]) -> (u8, u8) {
    let r = gamma_avg_2(p1[0], p2[0]);
    let g = gamma_avg_2(p1[1], p2[1]);
    let b = gamma_avg_2(p1[2], p2[2]);
    (rgb_to_u_single(r, g, b), rgb_to_v_single(r, g, b))
}

/// Convert image to YUV420 using sharp (iterative) chroma downsampling.
///
/// This produces higher-quality chroma planes at the cost of being slower.
/// Uses the `yuv` crate's sharp YUV implementation with BT.601 matrix and sRGB gamma.
#[allow(dead_code)]
pub(crate) fn convert_image_sharp_yuv(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
) -> (
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
) {
    convert_image_sharp_yuv_with_config(
        image_data,
        color,
        width,
        height,
        stride,
        zenyuv::SharpYuvConfig::default(),
    )
}

/// Convert image to YUV420 using sharp (iterative) chroma downsampling
/// with an explicit [`SharpYuvConfig`](zenyuv::SharpYuvConfig).
pub(crate) fn convert_image_sharp_yuv_with_config(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
    config: zenyuv::SharpYuvConfig,
) -> (
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
) {
    use crate::encoder::PixelLayout;

    // Sharp YUV only applies to RGB/RGBA/BGR/BGRA inputs (chroma subsampling matters).
    // For grayscale, fall back to standard conversion.
    match color {
        PixelLayout::L8 => return convert_image_y::<1>(image_data, width, height, stride),
        PixelLayout::La8 => return convert_image_y::<2>(image_data, width, height, stride),
        PixelLayout::Yuv420 => {
            unreachable!("sharp YUV should not be called with Yuv420 input");
        }
        PixelLayout::Argb8 => {
            unreachable!("sharp YUV should not be called with Argb8 input");
        }
        _ => {}
    }

    zenyuv_encode_planes(
        image_data,
        color,
        width,
        height,
        stride,
        YuvEncodeMode::SharpWith(config),
    )
}

/// Fast non-sharp RGB/RGBA/BGR/BGRA → YUV420 conversion using zenyuv for Y
/// (SIMD) and scalar gamma-corrected chroma downsampling for U/V.
///
/// The Y plane is computed by zenyuv's SIMD kernel (AVX2/NEON/WASM SIMD128),
/// which is ~2-3x faster than the pure scalar path. U/V use zenwebp's existing
/// gamma-corrected formula to match libwebp's default chroma quality.
pub(crate) fn convert_image_yuv_fast(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    zenyuv_encode_planes(image_data, color, width, height, stride, YuvEncodeMode::Box)
}

/// Return a tightly-packed RGB buffer (`width * height * 3` bytes). Uses garb's
/// strided variants, which process the whole frame in a single SIMD pass.
///
/// If the source is already tightly-packed Rgb8 (stride == width), returns a
/// borrowed view and avoids the copy.
fn to_tight_rgb<'a>(
    src: &'a [u8],
    color: crate::encoder::PixelLayout,
    w: usize,
    h: usize,
    src_stride_bytes: usize,
) -> alloc::borrow::Cow<'a, [u8]> {
    use alloc::borrow::Cow;

    use crate::encoder::PixelLayout;
    let tight_rgb_stride = w * 3;
    if color == PixelLayout::Rgb8 && src_stride_bytes == tight_rgb_stride {
        return Cow::Borrowed(&src[..tight_rgb_stride * h]);
    }

    let mut rgb = alloc::vec![0u8; w * h * 3];
    match color {
        PixelLayout::Rgb8 => {
            // Strided RGB → tight RGB: row copies.
            for y in 0..h {
                rgb[y * tight_rgb_stride..(y + 1) * tight_rgb_stride].copy_from_slice(
                    &src[y * src_stride_bytes..y * src_stride_bytes + tight_rgb_stride],
                );
            }
        }
        PixelLayout::Bgr8 => {
            garb::bytes::bgr_to_rgb_strided(
                src,
                &mut rgb,
                w,
                h,
                src_stride_bytes,
                tight_rgb_stride,
            )
            .expect("validated sizes");
        }
        PixelLayout::Rgba8 => {
            garb::bytes::rgba_to_rgb_strided(
                src,
                &mut rgb,
                w,
                h,
                src_stride_bytes,
                tight_rgb_stride,
            )
            .expect("validated sizes");
        }
        PixelLayout::Bgra8 => {
            garb::bytes::bgra_to_rgb_strided(
                src,
                &mut rgb,
                w,
                h,
                src_stride_bytes,
                tight_rgb_stride,
            )
            .expect("validated sizes");
        }
        _ => unreachable!(),
    }
    Cow::Owned(rgb)
}

/// Copy a tight `src_w x src_h` plane into a `dst_w x dst_h` buffer with edge replication
/// on the right and bottom borders.
fn pad_plane(src: &[u8], dst: &mut [u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) {
    for y in 0..src_h {
        let sr = &src[y * src_w..(y + 1) * src_w];
        let dr = &mut dst[y * dst_w..(y + 1) * dst_w];
        dr[..src_w].copy_from_slice(sr);
        if dst_w > src_w {
            let last = sr[src_w - 1];
            for x in src_w..dst_w {
                dr[x] = last;
            }
        }
    }
    if dst_h > src_h {
        // Replicate the last written row
        let (filled, rest) = dst.split_at_mut(src_h * dst_w);
        let last_row = &filled[(src_h - 1) * dst_w..];
        for chunk in rest.chunks_exact_mut(dst_w) {
            chunk.copy_from_slice(last_row);
        }
    }
}

/// Replicate rows [src_h..dst_h) by copying the last written row. Used when an
/// image already has mb-aligned width but height < mb-aligned height.
fn pad_plane_vertical(dst: &mut [u8], row_w: usize, src_h: usize, dst_h: usize) {
    if dst_h <= src_h {
        return;
    }
    let (filled, rest) = dst.split_at_mut(src_h * row_w);
    let last_row = &filled[(src_h - 1) * row_w..src_h * row_w];
    for chunk in rest.chunks_exact_mut(row_w) {
        chunk.copy_from_slice(last_row);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum YuvEncodeMode {
    /// Non-sharp: zenyuv for Y (SIMD), gamma-corrected scalar for chroma.
    /// Matches libwebp's default chroma quality.
    Box,
    /// Sharp YUV: zenyuv's iterative chroma optimization with explicit config.
    SharpWith(zenyuv::SharpYuvConfig),
}

/// Unified zenyuv-backed encoder for RGB/RGBA/BGR/BGRA → YUV420 (mb-aligned).
///
/// Optimizations over the naive path:
/// - For Rgb8 with tight stride, avoids the RGB intermediate copy entirely.
/// - When the image width is already a multiple of 16 (mb-aligned), writes Y/U/V
///   directly into the mb-aligned output buffers (no tight intermediate). Only
///   vertical padding needs a copy.
/// - For [`YuvEncodeMode::Box`], overwrites zenyuv's linear-averaged chroma with
///   our gamma-corrected scalar computation for libwebp-parity perceptual quality.
fn zenyuv_encode_planes(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
    mode: YuvEncodeMode,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use crate::encoder::PixelLayout;
    use zenyuv::{Matrix, Range, YuvContext};

    let w = usize::from(width);
    let h = usize::from(height);
    let mb_width = w.div_ceil(16);
    let mb_height = h.div_ceil(16);
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let luma_height = 16 * mb_height;
    let chroma_height = 8 * mb_height;
    let y_size = luma_width * luma_height;
    let chroma_size = chroma_width * chroma_height;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let src_bpp: usize = match color {
        PixelLayout::Rgb8 | PixelLayout::Bgr8 => 3,
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => 4,
        _ => unreachable!("zenyuv_encode_planes: unsupported layout {:?}", color),
    };
    let src_stride_bytes = stride * src_bpp;
    let rgb = to_tight_rgb(image_data, color, w, h, src_stride_bytes);
    let rgb: &[u8] = &rgb;

    // Allocate mb-aligned output buffers up-front.
    let mut y_bytes = alloc::vec![0u8; y_size];
    let mut u_bytes = alloc::vec![0u8; chroma_size];
    let mut v_bytes = alloc::vec![0u8; chroma_size];

    // Fast path: when w is already mb-aligned, luma rows match and chroma rows
    // match (cw = w/2 = chroma_width). We can write directly into the mb-aligned
    // buffers and only need vertical padding.
    let mb_aligned_width = w == luma_width;

    let mut ctx = YuvContext::new(Range::Limited, Matrix::Bt601);

    // Two-pass: zenyuv Y-only (fast maddubs SIMD, no chroma compute)
    // + our SIMD gamma chroma overwrite.
    if mb_aligned_width {
        ctx.encode_420_y_only_u8(rgb, &mut y_bytes[..w * h], w, h);
        pad_plane_vertical(&mut y_bytes, luma_width, h, luma_height);
    } else {
        let mut y_tight = alloc::vec![0u8; w * h];
        ctx.encode_420_y_only_u8(rgb, &mut y_tight, w, h);
        pad_plane(&y_tight, &mut y_bytes, w, h, luma_width, luma_height);
    }
    gamma_chroma_overwrite(
        rgb,
        w,
        h,
        &mut u_bytes,
        &mut v_bytes,
        chroma_width,
        chroma_height,
    );

    // Sharp path: refine the gamma-corrected chroma via Newton-step iteration.
    // The gamma chroma matches the decoder's model, so the iteration starts from
    // a correct baseline and can only improve. The iteration minimizes
    // ||RGB_original - RGB_reconstructed|| using the BT.601-Limited inverse matrix.
    if let YuvEncodeMode::SharpWith(config) = mode {
        if mb_aligned_width {
            // Tight Y is the first w*h bytes of the mb-aligned buffer.
            // Tight chroma is cw-strided, which equals chroma_width when aligned.
            zenyuv::sharp::refine_chroma_420_u8(
                rgb,
                &y_bytes[..w * h],
                &mut u_bytes[..cw * ch],
                &mut v_bytes[..cw * ch],
                w,
                h,
                Range::Limited,
                Matrix::Bt601,
                &config,
            );
            // Step 4: Refine Y to compensate for chroma-induced luma error.
            if config.refine_y {
                zenyuv::sharp::refine_y_420_u8(
                    rgb,
                    &mut y_bytes[..w * h],
                    &u_bytes[..cw * ch],
                    &v_bytes[..cw * ch],
                    w,
                    h,
                    Range::Limited,
                    Matrix::Bt601,
                );
            }
            // Re-pad luma and chroma vertically after refinement.
            pad_plane_vertical(&mut y_bytes, luma_width, h, luma_height);
            pad_plane_vertical(&mut u_bytes, chroma_width, ch, chroma_height);
            pad_plane_vertical(&mut v_bytes, chroma_width, ch, chroma_height);
        } else {
            // Need tight Y and tight chroma for the refine function.
            // y_bytes is mb-aligned (luma_width stride), extract tight (w stride).
            let mut y_tight = alloc::vec![0u8; w * h];
            for row in 0..h {
                y_tight[row * w..(row + 1) * w]
                    .copy_from_slice(&y_bytes[row * luma_width..row * luma_width + w]);
            }
            let mut u_tight = alloc::vec![0u8; cw * ch];
            let mut v_tight = alloc::vec![0u8; cw * ch];
            for row in 0..ch {
                u_tight[row * cw..(row + 1) * cw]
                    .copy_from_slice(&u_bytes[row * chroma_width..row * chroma_width + cw]);
                v_tight[row * cw..(row + 1) * cw]
                    .copy_from_slice(&v_bytes[row * chroma_width..row * chroma_width + cw]);
            }
            zenyuv::sharp::refine_chroma_420_u8(
                rgb,
                &y_tight,
                &mut u_tight,
                &mut v_tight,
                w,
                h,
                Range::Limited,
                Matrix::Bt601,
                &config,
            );
            // Step 4: Refine Y to compensate for chroma-induced luma error.
            if config.refine_y {
                zenyuv::sharp::refine_y_420_u8(
                    rgb,
                    &mut y_tight,
                    &u_tight,
                    &v_tight,
                    w,
                    h,
                    Range::Limited,
                    Matrix::Bt601,
                );
            }
            // Copy refined Y back to mb-aligned buffer with padding.
            for row in 0..h {
                y_bytes[row * luma_width..row * luma_width + w]
                    .copy_from_slice(&y_tight[row * w..(row + 1) * w]);
            }
            // Copy refined chroma back to mb-aligned buffers with padding.
            for row in 0..ch {
                u_bytes[row * chroma_width..row * chroma_width + cw]
                    .copy_from_slice(&u_tight[row * cw..(row + 1) * cw]);
                v_bytes[row * chroma_width..row * chroma_width + cw]
                    .copy_from_slice(&v_tight[row * cw..(row + 1) * cw]);
            }
            // Horizontal + vertical padding for chroma.
            for row in 0..ch {
                let last_u = u_bytes[row * chroma_width + cw - 1];
                let last_v = v_bytes[row * chroma_width + cw - 1];
                for x in cw..chroma_width {
                    u_bytes[row * chroma_width + x] = last_u;
                    v_bytes[row * chroma_width + x] = last_v;
                }
            }
            // Horizontal padding for luma.
            for row in 0..h {
                let last_y = y_bytes[row * luma_width + w - 1];
                for x in w..luma_width {
                    y_bytes[row * luma_width + x] = last_y;
                }
            }
            pad_plane_vertical(&mut y_bytes, luma_width, h, luma_height);
            pad_plane_vertical(&mut u_bytes, chroma_width, ch, chroma_height);
            pad_plane_vertical(&mut v_bytes, chroma_width, ch, chroma_height);
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

/// Build the dense 4096-entry inverse-gamma LUT. Computed lazily via `OnceLock`
/// from the existing 33-entry interpolated formula; eliminates per-pixel
/// interpolation arithmetic in the gamma-corrected chroma path.
#[cfg(feature = "std")]
fn linear_to_gamma_dense() -> &'static [u8; 4096] {
    use std::sync::OnceLock;
    static LUT: OnceLock<alloc::boxed::Box<[u8; 4096]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = alloc::boxed::Box::new([0u8; 4096]);
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = linear_to_gamma(i as u32);
        }
        t
    })
}

/// SIMD kernel generic over archmage Token: process all full row-pairs and
/// bulk col-pairs for gamma-corrected U/V chroma downsampling. Uses the
/// scalar-LUT-bookend pattern (see `~/work/claudehints/topics/rust-defaults.md`).
///
/// Tried fusing Y computation into this kernel — the i32x8 Y matrix was ~27%
/// slower than zenyuv's maddubs-based Y kernel, so we keep them split.
/// `#[magetypes]` generates `_v3`/`_neon`/`_wasm128`/`_scalar` variants;
/// `incant!` at the call site dispatches to the best available.
#[cfg(feature = "std")]
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn gamma_chroma_rows_generic(
    token: Token,
    rgb: &[u8],
    w: usize,
    row_pairs: usize,
    bulk_col_pairs: usize,
    u_bytes: &mut [u8],
    v_bytes: &mut [u8],
    chroma_width: usize,
    gamma_lut: &[u16; 256],
    inv_gamma_dense: &[u8; 4096],
) {
    #[allow(non_camel_case_types)]
    type u16x8 = magetypes::simd::generic::u16x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = magetypes::simd::generic::i32x8<Token>;

    let two = u16x8::splat(token, 2);
    let uv_bias = (128i32 << YUV_FIX) + YUV_HALF;
    let u_r = i32x8::splat(token, -9719);
    let u_g = i32x8::splat(token, -19081);
    let u_b = i32x8::splat(token, 28800);
    let v_r = i32x8::splat(token, 28800);
    let v_g = i32x8::splat(token, -24116);
    let v_b = i32x8::splat(token, -4684);
    let uv_bias_v = i32x8::splat(token, uv_bias);

    for row_pair in 0..row_pairs {
        let r1 = row_pair * 2;
        let r2 = r1 + 1;
        let top_row_off = r1 * w * 3;
        let bot_row_off = r2 * w * 3;
        let mut cp = 0;
        while cp < bulk_col_pairs {
            let col0 = cp * 2;
            let top: &[u8; 48] = (&rgb[top_row_off + col0 * 3..top_row_off + col0 * 3 + 48])
                .try_into()
                .unwrap();
            let bot: &[u8; 48] = (&rgb[bot_row_off + col0 * 3..bot_row_off + col0 * 3 + 48])
                .try_into()
                .unwrap();

            // ── Stage 1: scalar gather in ──
            let mut r_te = [0u16; 8];
            let mut r_to = [0u16; 8];
            let mut r_be = [0u16; 8];
            let mut r_bo = [0u16; 8];
            let mut g_te = [0u16; 8];
            let mut g_to = [0u16; 8];
            let mut g_be = [0u16; 8];
            let mut g_bo = [0u16; 8];
            let mut b_te = [0u16; 8];
            let mut b_to = [0u16; 8];
            let mut b_be = [0u16; 8];
            let mut b_bo = [0u16; 8];
            for i in 0..8 {
                let ei = 2 * i;
                let oi = ei + 1;
                r_te[i] = gamma_lut[top[ei * 3] as usize];
                r_to[i] = gamma_lut[top[oi * 3] as usize];
                r_be[i] = gamma_lut[bot[ei * 3] as usize];
                r_bo[i] = gamma_lut[bot[oi * 3] as usize];
                g_te[i] = gamma_lut[top[ei * 3 + 1] as usize];
                g_to[i] = gamma_lut[top[oi * 3 + 1] as usize];
                g_be[i] = gamma_lut[bot[ei * 3 + 1] as usize];
                g_bo[i] = gamma_lut[bot[oi * 3 + 1] as usize];
                b_te[i] = gamma_lut[top[ei * 3 + 2] as usize];
                b_to[i] = gamma_lut[top[oi * 3 + 2] as usize];
                b_be[i] = gamma_lut[bot[ei * 3 + 2] as usize];
                b_bo[i] = gamma_lut[bot[oi * 3 + 2] as usize];
            }

            // ── Stage 2: SIMD average ── (max sum = 4*4095 fits in u16)
            let r_sum = u16x8::from_array(token, r_te)
                + u16x8::from_array(token, r_to)
                + u16x8::from_array(token, r_be)
                + u16x8::from_array(token, r_bo);
            let g_sum = u16x8::from_array(token, g_te)
                + u16x8::from_array(token, g_to)
                + u16x8::from_array(token, g_be)
                + u16x8::from_array(token, g_bo);
            let b_sum = u16x8::from_array(token, b_te)
                + u16x8::from_array(token, b_to)
                + u16x8::from_array(token, b_be)
                + u16x8::from_array(token, b_bo);
            let r_avg_v = (r_sum + two).shr_logical_const::<2>();
            let g_avg_v = (g_sum + two).shr_logical_const::<2>();
            let b_avg_v = (b_sum + two).shr_logical_const::<2>();

            let mut r_avg = [0u16; 8];
            let mut g_avg = [0u16; 8];
            let mut b_avg = [0u16; 8];
            r_avg_v.store(&mut r_avg);
            g_avg_v.store(&mut g_avg);
            b_avg_v.store(&mut b_avg);

            // ── Stage 3: scalar gather out (dense LUT, no interpolation) ──
            let mut r_u8 = [0i32; 8];
            let mut g_u8 = [0i32; 8];
            let mut b_u8 = [0i32; 8];
            for i in 0..8 {
                r_u8[i] = inv_gamma_dense[(r_avg[i] as usize) & 0xFFF] as i32;
                g_u8[i] = inv_gamma_dense[(g_avg[i] as usize) & 0xFFF] as i32;
                b_u8[i] = inv_gamma_dense[(b_avg[i] as usize) & 0xFFF] as i32;
            }

            // ── Stage 4: SIMD YCbCr matrix (i32 fixed-point, YUV_FIX=16) ──
            let r_v = i32x8::from_array(token, r_u8);
            let g_v = i32x8::from_array(token, g_u8);
            let b_v = i32x8::from_array(token, b_u8);
            let u_vals = uv_bias_v + r_v * u_r + g_v * u_g + b_v * u_b;
            let v_vals = uv_bias_v + r_v * v_r + g_v * v_g + b_v * v_b;

            let mut u_i32 = [0i32; 8];
            let mut v_i32 = [0i32; 8];
            u_vals.shr_arithmetic_const::<16>().store(&mut u_i32);
            v_vals.shr_arithmetic_const::<16>().store(&mut v_i32);

            let idx = row_pair * chroma_width + cp;
            for i in 0..8 {
                u_bytes[idx + i] = u_i32[i].clamp(0, 255) as u8;
                v_bytes[idx + i] = v_i32[i].clamp(0, 255) as u8;
            }

            cp += 8;
        }
    }
}

/// Overwrite the U/V chroma planes with gamma-corrected values. Leaves Y
/// untouched. Called after zenyuv has populated all three planes — we just
/// replace the linear-averaged chroma with the gamma-corrected equivalent.
///
/// SIMD fast path via `yuv420_rows_generic` processes 8 chroma outputs per
/// iteration but skips Y writes via a stride of 0 (its Y output is discarded
/// below — cache-only, no effect on correctness since zenyuv already wrote
/// the correct Y). Scalar fallback handles tail cols and odd rows.
fn gamma_chroma_overwrite(
    rgb: &[u8],
    w: usize,
    h: usize,
    u_bytes: &mut [u8],
    v_bytes: &mut [u8],
    chroma_width: usize,
    chroma_height_mb: usize,
) {
    let src_chroma_width = w.div_ceil(2);
    let src_chroma_height = h.div_ceil(2);
    let row_pairs = h / 2;
    let odd_height = h & 1 != 0;
    let col_pairs = w / 2;
    let odd_width = w & 1 != 0;

    let px = |x: usize, y: usize| -> &[u8] { &rgb[(y * w + x) * 3..(y * w + x) * 3 + 3] };

    #[cfg(feature = "std")]
    let simd_col_pairs = {
        use archmage::incant;
        let bulk_col_pairs = col_pairs & !7;
        if bulk_col_pairs > 0 {
            incant!(
                gamma_chroma_rows_generic(
                    rgb,
                    w,
                    row_pairs,
                    bulk_col_pairs,
                    u_bytes,
                    v_bytes,
                    chroma_width,
                    &GAMMA_TO_LINEAR_TAB,
                    linear_to_gamma_dense(),
                ),
                [v3, neon, wasm128, scalar]
            );
        }
        bulk_col_pairs
    };
    #[cfg(not(feature = "std"))]
    let simd_col_pairs = 0usize;

    // Scalar tail for UV only — Y is already correct from zenyuv.
    for row_pair in 0..row_pairs {
        let r1 = row_pair * 2;
        let r2 = r1 + 1;
        for col_pair in simd_col_pairs..col_pairs {
            let c1 = col_pair * 2;
            let c2 = c1 + 1;
            let (u, v) = gamma_downsample_uv_4(px(c1, r1), px(c2, r1), px(c1, r2), px(c2, r2));
            let idx = row_pair * chroma_width + col_pair;
            u_bytes[idx] = u;
            v_bytes[idx] = v;
        }
        if odd_width {
            let c = w - 1;
            let (u, v) = gamma_downsample_uv_2(px(c, r1), px(c, r2));
            let idx = row_pair * chroma_width + col_pairs;
            u_bytes[idx] = u;
            v_bytes[idx] = v;
        }
    }
    if odd_height {
        let r = h - 1;
        for col_pair in 0..col_pairs {
            let c1 = col_pair * 2;
            let c2 = c1 + 1;
            let (u, v) = gamma_downsample_uv_2(px(c1, r), px(c2, r));
            let idx = row_pairs * chroma_width + col_pair;
            u_bytes[idx] = u;
            v_bytes[idx] = v;
        }
        if odd_width {
            let c = w - 1;
            let p = px(c, r);
            let idx = row_pairs * chroma_width + col_pairs;
            u_bytes[idx] = rgb_to_u_single(p[0], p[1], p[2]);
            v_bytes[idx] = rgb_to_v_single(p[0], p[1], p[2]);
        }
    }

    // Edge replication for UV (Y's edges are already handled by caller).
    for y in 0..src_chroma_height {
        let last_u = u_bytes[y * chroma_width + src_chroma_width - 1];
        let last_v = v_bytes[y * chroma_width + src_chroma_width - 1];
        for x in src_chroma_width..chroma_width {
            u_bytes[y * chroma_width + x] = last_u;
            v_bytes[y * chroma_width + x] = last_v;
        }
    }
    pad_plane_vertical(u_bytes, chroma_width, src_chroma_height, chroma_height_mb);
    pad_plane_vertical(v_bytes, chroma_width, src_chroma_height, chroma_height_mb);
}

/// Convert BGR/BGRA image data to YUV420 with macroblock alignment.
///
/// Same as `convert_image_yuv` but reads pixels as B,G,R(,A) instead of R,G,B(,A).
/// BPP=3 for BGR, BPP=4 for BGRA.
// zenyuv path replaces this in production; kept for symmetry with convert_image_yuv.
#[allow(dead_code)]
pub(crate) fn convert_image_yuv_bgr<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Helper to convert BGR pixel slice to RGB-ordered slice for the rgb_to_* functions.
    // We swap B and R so the existing rgb_to_y/u/v functions work correctly.
    #[inline(always)]
    fn bgr_to_rgb(bgr: &[u8]) -> [u8; 4] {
        // Return [R, G, B, ...] from [B, G, R, ...]
        [bgr[2], bgr[1], bgr[0], 0]
    }

    let row_pairs = height / 2;
    let odd_height = height & 1 != 0;
    let col_pairs = width / 2;
    let odd_width = width & 1 != 0;

    for row_pair in 0..row_pairs {
        let src_row1 = row_pair * 2;
        let src_row2 = src_row1 + 1;
        let chroma_row = row_pair;

        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            let rgb1 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col1) * BPP..]);
            let rgb2 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col2) * BPP..]);
            let rgb3 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col1) * BPP..]);
            let rgb4 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col2) * BPP..]);

            y_bytes[src_row1 * luma_width + src_col1] = rgb_to_y(&rgb1);
            y_bytes[src_row1 * luma_width + src_col2] = rgb_to_y(&rgb2);
            y_bytes[src_row2 * luma_width + src_col1] = rgb_to_y(&rgb3);
            y_bytes[src_row2 * luma_width + src_col2] = rgb_to_y(&rgb4);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb2, &rgb3, &rgb4);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb2, &rgb3, &rgb4);
        }

        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            let rgb1 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col) * BPP..]);
            let rgb3 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col) * BPP..]);

            y_bytes[src_row1 * luma_width + src_col] = rgb_to_y(&rgb1);
            y_bytes[src_row2 * luma_width + src_col] = rgb_to_y(&rgb3);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb1, &rgb3, &rgb3);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb1, &rgb3, &rgb3);
        }
    }

    if odd_height {
        let src_row = height - 1;
        let chroma_row = row_pairs;

        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            let rgb1 = bgr_to_rgb(&image_data[(src_row * stride + src_col1) * BPP..]);
            let rgb2 = bgr_to_rgb(&image_data[(src_row * stride + src_col2) * BPP..]);

            y_bytes[src_row * luma_width + src_col1] = rgb_to_y(&rgb1);
            y_bytes[src_row * luma_width + src_col2] = rgb_to_y(&rgb2);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb2, &rgb1, &rgb2);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb2, &rgb1, &rgb2);
        }

        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            let rgb = bgr_to_rgb(&image_data[(src_row * stride + src_col) * BPP..]);

            y_bytes[src_row * luma_width + src_col] = rgb_to_y(&rgb);

            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(&rgb, &rgb, &rgb, &rgb);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(&rgb, &rgb, &rgb, &rgb);
        }
    }

    // Replicate edge pixels to fill macroblock padding (same as convert_image_yuv)
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }
    let chroma_height = height.div_ceil(2);
    let actual_chroma_width = width.div_ceil(2);
    for y in 0..chroma_height {
        let last_u = u_bytes[y * chroma_width + actual_chroma_width - 1];
        let last_v = v_bytes[y * chroma_width + actual_chroma_width - 1];
        for x in actual_chroma_width..chroma_width {
            u_bytes[y * chroma_width + x] = last_u;
            v_bytes[y * chroma_width + x] = last_v;
        }
    }
    for y in chroma_height..(mb_height * 8) {
        for x in 0..chroma_width {
            u_bytes[y * chroma_width + x] = u_bytes[(chroma_height - 1) * chroma_width + x];
            v_bytes[y * chroma_width + x] = v_bytes[(chroma_height - 1) * chroma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

/// Import raw YUV 4:2:0 planes into macroblock-aligned buffers.
///
/// Copies Y/U/V planes with macroblock padding (edge replication).
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn import_yuv420_planes(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u16,
    height: u16,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = usize::from(width);
    let h = usize::from(height);
    let mb_width = w.div_ceil(16);
    let mb_height = h.div_ceil(16);
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let y_size = luma_width * 16 * mb_height;
    let chroma_size = chroma_width * 8 * mb_height;

    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);

    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Copy Y plane with horizontal padding
    for y in 0..h {
        let src_start = y * w;
        let dst_start = y * luma_width;
        y_bytes[dst_start..dst_start + w].copy_from_slice(&y_plane[src_start..src_start + w]);
        let last_y = y_bytes[dst_start + w - 1];
        for x in w..luma_width {
            y_bytes[dst_start + x] = last_y;
        }
    }
    // Vertical padding for Y
    for y in h..(mb_height * 16) {
        let src_row = (h - 1) * luma_width;
        let dst_row = y * luma_width;
        y_bytes.copy_within(src_row..src_row + luma_width, dst_row);
    }

    // Copy U/V planes with horizontal padding
    for y in 0..uv_h {
        let src_start = y * uv_w;
        let dst_start = y * chroma_width;
        u_bytes[dst_start..dst_start + uv_w].copy_from_slice(&u_plane[src_start..src_start + uv_w]);
        v_bytes[dst_start..dst_start + uv_w].copy_from_slice(&v_plane[src_start..src_start + uv_w]);
        let last_u = u_bytes[dst_start + uv_w - 1];
        let last_v = v_bytes[dst_start + uv_w - 1];
        for x in uv_w..chroma_width {
            u_bytes[dst_start + x] = last_u;
            v_bytes[dst_start + x] = last_v;
        }
    }
    // Vertical padding for U/V
    for y in uv_h..(mb_height * 8) {
        let src_row = (uv_h - 1) * chroma_width;
        let dst_row = y * chroma_width;
        u_bytes.copy_within(src_row..src_row + chroma_width, dst_row);
        v_bytes.copy_within(src_row..src_row + chroma_width, dst_row);
    }

    (y_bytes, u_bytes, v_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_conversions() {
        let (y, u, v) = (203, 40, 42);

        assert_eq!(yuv_to_r(y, v), 80);
        assert_eq!(yuv_to_g(y, u, v), 255);
        assert_eq!(yuv_to_b(y, u), 40);
    }
}

// ============================================================================
// SSE2 YUV->RGB conversion (x86_64)
// ============================================================================

// YUV to RGB conversion constants (14-bit fixed-point, matching libwebp).
// R = (19077 * y             + 26149 * v - 14234) >> 6
// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
// B = (19077 * y + 33050 * u             - 17685) >> 6

/// Load 8 bytes into the upper 8 bits of 16-bit words (equivalent to << 8).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn load_hi_16(_token: X64V3Token, src: &[u8; 8]) -> __m128i {
    let zero = _mm_setzero_si128();
    // Load 8 bytes as i64 then convert to __m128i
    let val = i64::from_le_bytes(*src);
    let data = _mm_cvtsi64_si128(val);
    _mm_unpacklo_epi8(zero, data)
}

/// Load 4 U/V bytes and replicate each to get 8 values for 4:2:0.
/// Result: [u0,u0,u1,u1,u2,u2,u3,u3] in upper 8 bits of 16-bit words.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn load_uv_hi_8(_token: X64V3Token, src: &[u8; 4]) -> __m128i {
    let zero = _mm_setzero_si128();
    // Load 4 bytes as i32
    let val = i32::from_le_bytes(*src);
    let tmp0 = _mm_cvtsi32_si128(val);
    // Unpack to 16-bit with zeros in low bytes: [0,u0,0,u1,0,u2,0,u3,...]
    let tmp1 = _mm_unpacklo_epi8(zero, tmp0);
    // Replicate: [0,u0,0,u0,0,u1,0,u1,...]
    _mm_unpacklo_epi16(tmp1, tmp1)
}

/// Convert 8 YUV444 pixels to R, G, B (16-bit results).
/// Input Y, U, V are in upper 8 bits of 16-bit words.
/// Output R, G, B are signed 16-bit values (will be clamped later).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn convert_yuv444_to_rgb(
    _token: X64V3Token,
    y: __m128i,
    u: __m128i,
    v: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let k19077 = _mm_set1_epi16(19077);
    let k26149 = _mm_set1_epi16(26149);
    let k14234 = _mm_set1_epi16(14234);
    // 33050 doesn't fit in signed i16, use unsigned arithmetic
    let k33050 = _mm_set1_epi16(33050u16 as i16);
    let k17685 = _mm_set1_epi16(17685);
    let k6419 = _mm_set1_epi16(6419);
    let k13320 = _mm_set1_epi16(13320);
    let k8708 = _mm_set1_epi16(8708);

    // Y contribution (same for all channels)
    let y1 = _mm_mulhi_epu16(y, k19077);

    // R = Y1 + V*26149 - 14234
    let r0 = _mm_mulhi_epu16(v, k26149);
    let r1 = _mm_sub_epi16(y1, k14234);
    let r2 = _mm_add_epi16(r1, r0);

    // G = Y1 - U*6419 - V*13320 + 8708
    let g0 = _mm_mulhi_epu16(u, k6419);
    let g1 = _mm_mulhi_epu16(v, k13320);
    let g2 = _mm_add_epi16(y1, k8708);
    let g3 = _mm_add_epi16(g0, g1);
    let g4 = _mm_sub_epi16(g2, g3);

    // B = Y1 + U*33050 - 17685 (careful with unsigned arithmetic)
    let b0 = _mm_mulhi_epu16(u, k33050);
    let b1 = _mm_adds_epu16(b0, y1);
    let b2 = _mm_subs_epu16(b1, k17685);

    // Final shift by 6
    // R and G can be negative, use arithmetic shift
    // B is always positive (due to unsigned ops), use logical shift
    let r = _mm_srai_epi16(r2, 6);
    let g = _mm_srai_epi16(g4, 6);
    let b = _mm_srli_epi16(b2, 6);

    (r, g, b)
}

/// Pack R, G, B, A (8 pixels each, 16-bit) into 32 bytes of RGBA output.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
#[allow(dead_code)]
fn pack_and_store_rgba(
    _token: X64V3Token,
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
    dst: &mut [u8; 32],
) {
    let rb = _mm_packus_epi16(r, b);
    let ga = _mm_packus_epi16(g, a);
    let rg = _mm_unpacklo_epi8(rb, ga);
    let ba = _mm_unpackhi_epi8(rb, ga);
    let rgba_lo = _mm_unpacklo_epi16(rg, ba);
    let rgba_hi = _mm_unpackhi_epi16(rg, ba);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), rgba_lo);
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(),
        rgba_hi,
    );
}

/// Helper macro for VP8PlanarTo24b - splits even/odd bytes
macro_rules! planar_to_24b_helper {
    ($in0:expr, $in1:expr, $in2:expr, $in3:expr, $in4:expr, $in5:expr,
     $out0:expr, $out1:expr, $out2:expr, $out3:expr, $out4:expr, $out5:expr) => {
        let v_mask = _mm_set1_epi16(0x00ff);
        // Take even bytes (lower 8 bits of each 16-bit word)
        $out0 = _mm_packus_epi16(_mm_and_si128($in0, v_mask), _mm_and_si128($in1, v_mask));
        $out1 = _mm_packus_epi16(_mm_and_si128($in2, v_mask), _mm_and_si128($in3, v_mask));
        $out2 = _mm_packus_epi16(_mm_and_si128($in4, v_mask), _mm_and_si128($in5, v_mask));
        // Take odd bytes (upper 8 bits of each 16-bit word)
        $out3 = _mm_packus_epi16(_mm_srli_epi16($in0, 8), _mm_srli_epi16($in1, 8));
        $out4 = _mm_packus_epi16(_mm_srli_epi16($in2, 8), _mm_srli_epi16($in3, 8));
        $out5 = _mm_packus_epi16(_mm_srli_epi16($in4, 8), _mm_srli_epi16($in5, 8));
    };
}

/// Convert planar RRRR...GGGG...BBBB... to interleaved RGBRGBRGB...
/// Input: 6 registers (R0, R1, G0, G1, B0, B1) with 16 bytes each = 32 R, 32 G, 32 B
/// Output: 6 registers with 96 bytes of interleaved RGB
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn planar_to_24b(
    _token: X64V3Token,
    in0: __m128i,
    in1: __m128i,
    in2: __m128i,
    in3: __m128i,
    in4: __m128i,
    in5: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // 5 passes of the permutation to convert:
    // Input: r0r1r2r3... | r16r17... | g0g1g2g3... | g16g17... | b0b1b2b3... | b16b17...
    // Output: r0g0b0r1g1b1... (interleaved RGB)

    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5);
    let (mut o0, mut o1, mut o2, mut o3, mut o4, mut o5);

    // Pass 1
    planar_to_24b_helper!(in0, in1, in2, in3, in4, in5, t0, t1, t2, t3, t4, t5);
    // Pass 2
    planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
    // Pass 3
    planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);
    // Pass 4
    planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
    // Pass 5
    planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);

    (t0, t1, t2, t3, t4, t5)
}

/// Convert 32 YUV444 pixels to 96 bytes of RGB.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn yuv444_to_rgb_32(
    _token: X64V3Token,
    y: &[u8; 32],
    u: &[u8; 32],
    v: &[u8; 32],
    dst: &mut [u8; 96],
) {
    // Process 4 groups of 8 pixels
    let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[..8]).unwrap());
    let u0 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[..8]).unwrap());
    let v0 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[..8]).unwrap());
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y0, u0, v0);

    let y1 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[8..16]).unwrap());
    let u1 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[8..16]).unwrap());
    let v1 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[8..16]).unwrap());
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y1, u1, v1);

    let y2 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[16..24]).unwrap());
    let u2 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[16..24]).unwrap());
    let v2 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[16..24]).unwrap());
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y2, u2, v2);

    let y3 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[24..32]).unwrap());
    let u3 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[24..32]).unwrap());
    let v3 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[24..32]).unwrap());
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y3, u3, v3);

    // Pack to 8-bit and arrange as RRRRGGGGBBBB
    let rgb0 = _mm_packus_epi16(r0, r1); // R0-R15
    let rgb1 = _mm_packus_epi16(r2, r3); // R16-R31
    let rgb2 = _mm_packus_epi16(g0, g1); // G0-G15
    let rgb3 = _mm_packus_epi16(g2, g3); // G16-G31
    let rgb4 = _mm_packus_epi16(b0, b1); // B0-B15
    let rgb5 = _mm_packus_epi16(b2, b3); // B16-B31

    // Interleave to RGBRGBRGB...
    let (out0, out1, out2, out3, out4, out5) =
        planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store 96 bytes
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[32..48]).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[48..64]).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[64..80]).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[80..96]).unwrap(), out5);
}

/// Convert 32 YUV420 pixels (32 Y, 16 U, 16 V) to 96 bytes of RGB.
/// Each U/V value is replicated for 2 adjacent Y pixels.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn yuv420_to_rgb_32(
    _token: X64V3Token,
    y: &[u8; 32],
    u: &[u8; 16],
    v: &[u8; 16],
    dst: &mut [u8; 96],
) {
    // Process 4 groups of 8 pixels, with U/V replication
    let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[..8]).unwrap());
    let u0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[..4]).unwrap());
    let v0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[..4]).unwrap());
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y0, u0, v0);

    let y1 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[8..16]).unwrap());
    let u1 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[4..8]).unwrap());
    let v1 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[4..8]).unwrap());
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y1, u1, v1);

    let y2 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[16..24]).unwrap());
    let u2 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[8..12]).unwrap());
    let v2 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[8..12]).unwrap());
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y2, u2, v2);

    let y3 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[24..32]).unwrap());
    let u3 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[12..16]).unwrap());
    let v3 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[12..16]).unwrap());
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y3, u3, v3);

    // Pack to 8-bit and arrange as RRRRGGGGBBBB
    let rgb0 = _mm_packus_epi16(r0, r1);
    let rgb1 = _mm_packus_epi16(r2, r3);
    let rgb2 = _mm_packus_epi16(g0, g1);
    let rgb3 = _mm_packus_epi16(g2, g3);
    let rgb4 = _mm_packus_epi16(b0, b1);
    let rgb5 = _mm_packus_epi16(b2, b3);

    // Interleave to RGBRGBRGB...
    let (out0, out1, out2, out3, out4, out5) =
        planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store 96 bytes
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[32..48]).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[48..64]).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[64..80]).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[80..96]).unwrap(), out5);
}

/// Scalar fallback for YUV to RGB conversion (single pixel).
#[inline]
fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    fn mulhi(val: u8, coeff: u16) -> i32 {
        ((u32::from(val) * u32::from(coeff)) >> 8) as i32
    }
    fn clip(v: i32) -> u8 {
        (v >> 6).clamp(0, 255) as u8
    }
    let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
    let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
    let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
    (r, g, b)
}

/// Convert a row of YUV420 to RGB using SIMD.
/// Processes 32 pixels at a time, with scalar fallback for remainder.
///
/// Requires SSE2. y must have `len` elements, u/v must have `len/2` elements.
/// dst must have `len * 3` bytes.
#[cfg(target_arch = "x86_64")]
pub fn yuv420_to_rgb_row(y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    yuv420_to_rgb_row_inner(token, y, u, v, dst);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn yuv420_to_rgb_row_inner(_token: X64V3Token, y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    let len = y.len();
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(u.len() >= len.div_ceil(2));
    assert!(v.len() >= len.div_ceil(2));
    assert!(dst.len() >= len * 3);

    let mut n = 0usize;

    // Process 32 pixels at a time
    while n + 32 <= len {
        let y_arr = <&[u8; 32]>::try_from(&y[n..n + 32]).unwrap();
        let u_arr = <&[u8; 16]>::try_from(&u[n / 2..n / 2 + 16]).unwrap();
        let v_arr = <&[u8; 16]>::try_from(&v[n / 2..n / 2 + 16]).unwrap();
        let dst_arr = <&mut [u8; 96]>::try_from(&mut dst[n * 3..n * 3 + 96]).unwrap();
        yuv420_to_rgb_32(_token, y_arr, u_arr, v_arr, dst_arr);
        n += 32;
    }

    // Scalar fallback for remainder
    while n < len {
        let y_val = y[n];
        let u_val = u[n / 2];
        let v_val = v[n / 2];
        let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
        dst[n * 3] = r;
        dst[n * 3 + 1] = g;
        dst[n * 3 + 2] = b;
        n += 1;
    }
}

/// Convert a row of YUV420 to RGBA using SIMD.
/// Alpha is set to 255.
///
/// Requires SSE2. y must have `len` elements, u/v must have `len/2` elements.
/// dst must have `len * 4` bytes.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn yuv420_to_rgba_row(y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    yuv420_to_rgba_row_inner(token, y, u, v, dst);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn yuv420_to_rgba_row_inner(_token: X64V3Token, y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    let len = y.len();
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(u.len() >= len.div_ceil(2));
    assert!(v.len() >= len.div_ceil(2));
    assert!(dst.len() >= len * 4);

    let k_alpha = _mm_set1_epi16(255);
    let mut n = 0usize;

    // Process 8 pixels at a time for RGBA
    while n + 8 <= len {
        let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[n..n + 8]).unwrap());
        let u0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[n / 2..n / 2 + 4]).unwrap());
        let v0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[n / 2..n / 2 + 4]).unwrap());
        let (r, g, b) = convert_yuv444_to_rgb(_token, y0, u0, v0);
        pack_and_store_rgba(
            _token,
            r,
            g,
            b,
            k_alpha,
            <&mut [u8; 32]>::try_from(&mut dst[n * 4..n * 4 + 32]).unwrap(),
        );

        n += 8;
    }

    // Scalar fallback for remainder
    while n < len {
        let y_val = y[n];
        let u_val = u[n / 2];
        let v_val = v[n / 2];
        let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
        dst[n * 4] = r;
        dst[n * 4 + 1] = g;
        dst[n * 4 + 2] = b;
        dst[n * 4 + 3] = 255;
        n += 1;
    }
}

// =============================================================================
// Fancy upsampling (bilinear interpolation of chroma)
// =============================================================================

/// Compute fancy chroma interpolation for 16 pixels using SIMD.
/// Formula: (9*a + 3*b + 3*c + d + 8) / 16
///
/// Uses libwebp's efficient approach with _mm_avg_epu8.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn fancy_upsample_16(
    _token: X64V3Token,
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i) {
    let one = _mm_set1_epi8(1);

    // s = (a + d + 1) / 2
    let s = _mm_avg_epu8(a, d);
    // t = (b + c + 1) / 2
    let t = _mm_avg_epu8(b, c);
    // st = s ^ t
    let st = _mm_xor_si128(s, t);

    // ad = a ^ d, bc = b ^ c
    let ad = _mm_xor_si128(a, d);
    let bc = _mm_xor_si128(b, c);

    // k = (a + b + c + d) / 4 with proper rounding
    // k = (s + t + 1) / 2 - ((a^d) | (b^c) | (s^t)) & 1
    let t1 = _mm_or_si128(ad, bc);
    let t2 = _mm_or_si128(t1, st);
    let t3 = _mm_and_si128(t2, one);
    let t4 = _mm_avg_epu8(s, t);
    let k = _mm_sub_epi8(t4, t3);

    // diag1 = (a + 3*b + 3*c + d) / 8 then (9*a + 3*b + 3*c + d) / 16 = (a + diag1 + 1) / 2
    // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
    let tmp1 = _mm_avg_epu8(k, t);
    let tmp2 = _mm_and_si128(bc, st);
    let tmp3 = _mm_xor_si128(k, t);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m1 = _mm_sub_epi8(tmp1, tmp5);

    // diag2 = (3*a + b + c + 3*d) / 8 then (3*a + 9*b + c + 3*d) / 16 = (b + diag2 + 1) / 2
    // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
    let tmp1 = _mm_avg_epu8(k, s);
    let tmp2 = _mm_and_si128(ad, st);
    let tmp3 = _mm_xor_si128(k, s);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m2 = _mm_sub_epi8(tmp1, tmp5);

    // Final results
    let diag1 = _mm_avg_epu8(a, m1); // (9*a + 3*b + 3*c + d + 8) / 16
    let diag2 = _mm_avg_epu8(b, m2); // (3*a + 9*b + c + 3*d + 8) / 16

    (diag1, diag2)
}

/// Upsample 32 chroma pixels from two rows.
/// Input: 17 pixels from each row (r1, r2)
/// Output: 32 upsampled pixels for top row, 32 for bottom row
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn upsample_32_pixels(_token: X64V3Token, r1: &[u8; 17], r2: &[u8; 17], out: &mut [u8; 128]) {
    let one = _mm_set1_epi8(1);

    let a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r1[..16]).unwrap());
    let b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r1[1..17]).unwrap());
    let c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r2[..16]).unwrap());
    let d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r2[1..17]).unwrap());

    let s = _mm_avg_epu8(a, d);
    let t = _mm_avg_epu8(b, c);
    let st = _mm_xor_si128(s, t);
    let ad = _mm_xor_si128(a, d);
    let bc = _mm_xor_si128(b, c);

    let t1 = _mm_or_si128(ad, bc);
    let t2 = _mm_or_si128(t1, st);
    let t3 = _mm_and_si128(t2, one);
    let t4 = _mm_avg_epu8(s, t);
    let k = _mm_sub_epi8(t4, t3);

    // m1 for diag1
    let tmp1 = _mm_avg_epu8(k, t);
    let tmp2 = _mm_and_si128(bc, st);
    let tmp3 = _mm_xor_si128(k, t);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let diag1 = _mm_sub_epi8(tmp1, tmp5);

    // m2 for diag2
    let tmp1 = _mm_avg_epu8(k, s);
    let tmp2 = _mm_and_si128(ad, st);
    let tmp3 = _mm_xor_si128(k, s);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let diag2 = _mm_sub_epi8(tmp1, tmp5);

    // Pack alternating pixels for top row: (9a+3b+3c+d)/16, (3a+9b+c+3d)/16
    let t_a = _mm_avg_epu8(a, diag1);
    let t_b = _mm_avg_epu8(b, diag2);
    let t_1 = _mm_unpacklo_epi8(t_a, t_b);
    let t_2 = _mm_unpackhi_epi8(t_a, t_b);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[..16]).unwrap(), t_1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[16..32]).unwrap(), t_2);

    // Pack for bottom row: roles of diag1/diag2 swapped
    let b_a = _mm_avg_epu8(c, diag2);
    let b_b = _mm_avg_epu8(d, diag1);
    let b_1 = _mm_unpacklo_epi8(b_a, b_b);
    let b_2 = _mm_unpackhi_epi8(b_a, b_b);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[64..80]).unwrap(), b_1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[80..96]).unwrap(), b_2);
}

/// Process 8 pixel pairs with fancy upsampling and YUV->RGB conversion.
/// This is the main entry point for fancy upsampling used by yuv.rs.
///
/// Requires SSE2. All input slices must have the required minimum lengths.
#[cfg(target_arch = "x86_64")]
/// Process 8 pixel pairs with fancy upsampling and YUV->RGB conversion.
/// This version accepts a pre-summoned token to avoid repeated summon() calls in loops.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub fn fancy_upsample_8_pairs_with_token(
    token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    fancy_upsample_8_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Used in tests
pub fn fancy_upsample_8_pairs(
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    fancy_upsample_8_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

/// Optimized inner function taking fixed-size arrays to eliminate bounds checks.
/// Takes: 16 Y bytes, 9 U/V bytes per row (for overlapping window), 48 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_8_pairs_inner_opt(
    _token: X64V3Token,
    y_row: &[u8; 16],
    u_row_1: &[u8; 9],
    u_row_2: &[u8; 9],
    v_row_1: &[u8; 9],
    v_row_2: &[u8; 9],
    rgb: &mut [u8; 48],
) {
    // Load 8 chroma values from fixed-size arrays - no bounds checks needed
    // Using a macro instead of a nested function because macros expand at call site,
    // so _mm_cvtsi64_si128 is called within the #[arcane] function's target_feature context.
    // A nested function wouldn't inherit target_feature and would fail to compile.
    macro_rules! load_8_from_9 {
        ($arr:expr, 0) => {{
            let bytes: [u8; 8] = [
                $arr[0], $arr[1], $arr[2], $arr[3], $arr[4], $arr[5], $arr[6], $arr[7],
            ];
            let val = i64::from_le_bytes(bytes);
            _mm_cvtsi64_si128(val)
        }};
        ($arr:expr, 1) => {{
            let bytes: [u8; 8] = [
                $arr[1], $arr[2], $arr[3], $arr[4], $arr[5], $arr[6], $arr[7], $arr[8],
            ];
            let val = i64::from_le_bytes(bytes);
            _mm_cvtsi64_si128(val)
        }};
    }

    let u_a = load_8_from_9!(u_row_1, 0);
    let u_b = load_8_from_9!(u_row_1, 1);
    let u_c = load_8_from_9!(u_row_2, 0);
    let u_d = load_8_from_9!(u_row_2, 1);

    let v_a = load_8_from_9!(v_row_1, 0);
    let v_b = load_8_from_9!(v_row_1, 1);
    let v_c = load_8_from_9!(v_row_2, 0);
    let v_d = load_8_from_9!(v_row_2, 1);

    // Compute fancy upsampled U/V
    let (u_diag1, u_diag2) = fancy_upsample_16(_token, u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(_token, v_a, v_b, v_c, v_d);

    // Interleave: [diag1[0], diag2[0], diag1[1], diag2[1], ...]
    let u_interleaved = _mm_unpacklo_epi8(u_diag1, u_diag2);
    let v_interleaved = _mm_unpacklo_epi8(v_diag1, v_diag2);

    // Load Y directly from fixed-size array - no bounds check
    let y_vec = simd_mem::_mm_loadu_si128(y_row);
    let zero = _mm_setzero_si128();

    // Process first 8 pixels
    let y_lo = _mm_unpacklo_epi8(zero, y_vec);
    let u_lo = _mm_unpacklo_epi8(zero, u_interleaved);
    let v_lo = _mm_unpacklo_epi8(zero, v_interleaved);
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y_lo, u_lo, v_lo);

    // Process second 8 pixels
    let y_hi = _mm_unpackhi_epi8(zero, y_vec);
    let u_hi = _mm_unpackhi_epi8(zero, u_interleaved);
    let v_hi = _mm_unpackhi_epi8(zero, v_interleaved);
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y_hi, u_hi, v_hi);

    // Pack to 8-bit
    let r8 = _mm_packus_epi16(r0, r1);
    let g8 = _mm_packus_epi16(g0, g1);
    let b8 = _mm_packus_epi16(b0, b1);

    // Interleave RGB using partial planar_to_24b
    let rgb0 = r8;
    let rgb1 = _mm_setzero_si128();
    let rgb2 = g8;
    let rgb3 = _mm_setzero_si128();
    let rgb4 = b8;
    let rgb5 = _mm_setzero_si128();

    let (out0, out1, out2, _, _, _) = planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store to fixed-size output - use split_at_mut for non-overlapping borrows
    let (rgb_0, rest) = rgb.split_at_mut(16);
    let (rgb_1, rgb_2) = rest.split_at_mut(16);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_0).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_1).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_2).unwrap(), out2);
}

/// Wrapper that converts slices to fixed-size arrays with a single bounds check.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_8_pairs_inner(
    _token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // Single bounds check at entry - arrays are then passed by reference
    let y: &[u8; 16] = y_row[..16].try_into().unwrap();
    let u1: &[u8; 9] = u_row_1[..9].try_into().unwrap();
    let u2: &[u8; 9] = u_row_2[..9].try_into().unwrap();
    let v1: &[u8; 9] = v_row_1[..9].try_into().unwrap();
    let v2: &[u8; 9] = v_row_2[..9].try_into().unwrap();
    let out: &mut [u8; 48] = (&mut rgb[..48]).try_into().unwrap();

    fancy_upsample_8_pairs_inner_opt(_token, y, u1, u2, v1, v2, out);
}

/// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV->RGB conversion.
/// Processes 2x the pixels of fancy_upsample_8_pairs, reducing loop overhead.
/// Takes: 32 Y bytes, 17 U/V bytes per row (for overlapping window), 96 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub fn fancy_upsample_16_pairs_with_token(
    token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    fancy_upsample_16_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

/// Optimized inner function for 32 Y pixels (16 chroma pairs).
/// Takes: 32 Y bytes, 17 U/V bytes per row, 96 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_16_pairs_inner(
    _token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // Single bounds check at entry
    let y: &[u8; 32] = y_row[..32].try_into().unwrap();
    let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
    let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
    let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
    let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
    let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

    fancy_upsample_16_pairs_inner_opt(_token, y, u1, u2, v1, v2, out);
}

/// Core SIMD implementation for 32 Y pixels (16 chroma pairs).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_16_pairs_inner_opt(
    _token: X64V3Token,
    y_row: &[u8; 32],
    u_row_1: &[u8; 17],
    u_row_2: &[u8; 17],
    v_row_1: &[u8; 17],
    v_row_2: &[u8; 17],
    rgb: &mut [u8; 96],
) {
    // Load 16 U/V values for each position (a=offset0, b=offset1)
    let u_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_1[0..16]).unwrap());
    let u_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_1[1..17]).unwrap());
    let u_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_2[0..16]).unwrap());
    let u_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_2[1..17]).unwrap());

    let v_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_1[0..16]).unwrap());
    let v_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_1[1..17]).unwrap());
    let v_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_2[0..16]).unwrap());
    let v_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_2[1..17]).unwrap());

    // Compute fancy upsampled U/V - produces 16 diag1 + 16 diag2 values each
    let (u_diag1, u_diag2) = fancy_upsample_16(_token, u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(_token, v_a, v_b, v_c, v_d);

    // Interleave diag1/diag2 to get 32 U and 32 V values
    // First 16: unpacklo gives [d1[0],d2[0],d1[1],d2[1],...d1[7],d2[7]]
    // Second 16: unpackhi gives [d1[8],d2[8],...d1[15],d2[15]]
    let u_lo = _mm_unpacklo_epi8(u_diag1, u_diag2); // U values for Y[0..16]
    let u_hi = _mm_unpackhi_epi8(u_diag1, u_diag2); // U values for Y[16..32]
    let v_lo = _mm_unpacklo_epi8(v_diag1, v_diag2);
    let v_hi = _mm_unpackhi_epi8(v_diag1, v_diag2);

    // Load 32 Y values
    let y_0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y_row[0..16]).unwrap());
    let y_1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y_row[16..32]).unwrap());

    let zero = _mm_setzero_si128();

    // Process first 16 Y pixels (4 groups of 8 for YUV conversion)
    // Group 0: Y[0..8], U_lo[0..8], V_lo[0..8]
    let y_0_lo = _mm_unpacklo_epi8(zero, y_0);
    let u_0_lo = _mm_unpacklo_epi8(zero, u_lo);
    let v_0_lo = _mm_unpacklo_epi8(zero, v_lo);
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y_0_lo, u_0_lo, v_0_lo);

    // Group 1: Y[8..16], U_lo[8..16], V_lo[8..16]
    let y_0_hi = _mm_unpackhi_epi8(zero, y_0);
    let u_0_hi = _mm_unpackhi_epi8(zero, u_lo);
    let v_0_hi = _mm_unpackhi_epi8(zero, v_lo);
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y_0_hi, u_0_hi, v_0_hi);

    // Group 2: Y[16..24], U_hi[0..8], V_hi[0..8]
    let y_1_lo = _mm_unpacklo_epi8(zero, y_1);
    let u_1_lo = _mm_unpacklo_epi8(zero, u_hi);
    let v_1_lo = _mm_unpacklo_epi8(zero, v_hi);
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y_1_lo, u_1_lo, v_1_lo);

    // Group 3: Y[24..32], U_hi[8..16], V_hi[8..16]
    let y_1_hi = _mm_unpackhi_epi8(zero, y_1);
    let u_1_hi = _mm_unpackhi_epi8(zero, u_hi);
    let v_1_hi = _mm_unpackhi_epi8(zero, v_hi);
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y_1_hi, u_1_hi, v_1_hi);

    // Pack R/G/B to 8-bit: each packus gives 16 bytes
    let r_0 = _mm_packus_epi16(r0, r1); // R for Y[0..16]
    let r_1 = _mm_packus_epi16(r2, r3); // R for Y[16..32]
    let g_0 = _mm_packus_epi16(g0, g1);
    let g_1 = _mm_packus_epi16(g2, g3);
    let b_0 = _mm_packus_epi16(b0, b1);
    let b_1 = _mm_packus_epi16(b2, b3);

    // Interleave RGB using planar_to_24b (32 pixels -> 96 bytes)
    let (out0, out1, out2, out3, out4, out5) = planar_to_24b(_token, r_0, r_1, g_0, g_1, b_0, b_1);

    // Store 96 bytes
    let (rgb_0, rest) = rgb.split_at_mut(16);
    let (rgb_1, rest) = rest.split_at_mut(16);
    let (rgb_2, rest) = rest.split_at_mut(16);
    let (rgb_3, rest) = rest.split_at_mut(16);
    let (rgb_4, rgb_5) = rest.split_at_mut(16);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_0).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_1).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_2).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_3).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_4).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_5).unwrap(), out5);
}

#[cfg(test)]
mod tests_simd {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_yuv_to_rgb_matches_scalar() {
        if X64V3Token::summon().is_none() {
            return;
        }

        let test_cases: [(u8, u8, u8); 8] = [
            (128, 128, 128),
            (255, 128, 128),
            (0, 128, 128),
            (203, 40, 42),
            (77, 34, 97),
            (162, 101, 167),
            (202, 84, 150),
            (185, 101, 167),
        ];

        let y: Vec<u8> = test_cases.iter().map(|(y, _, _)| *y).collect();
        let u: Vec<u8> = test_cases.iter().map(|(_, u, _)| *u).collect();
        let v: Vec<u8> = test_cases.iter().map(|(_, _, v)| *v).collect();

        let u_420: Vec<u8> = u.iter().step_by(2).copied().collect();
        let v_420: Vec<u8> = v.iter().step_by(2).copied().collect();

        let mut rgb_simd = vec![0u8; 24];
        yuv420_to_rgb_row(&y, &u_420, &v_420, &mut rgb_simd);

        for i in 0..8 {
            let y_val = y[i];
            let u_val = u_420[i / 2];
            let v_val = v_420[i / 2];
            let (r_scalar, g_scalar, b_scalar) = yuv_to_rgb_scalar(y_val, u_val, v_val);

            assert_eq!(rgb_simd[i * 3], r_scalar);
            assert_eq!(rgb_simd[i * 3 + 1], g_scalar);
            assert_eq!(rgb_simd[i * 3 + 2], b_scalar);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_yuv_to_rgb_32_pixels() {
        if X64V3Token::summon().is_none() {
            return;
        }

        let y: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
        let u: Vec<u8> = (0..16).map(|i| (128 + i * 4) as u8).collect();
        let v: Vec<u8> = (0..16).map(|i| (128 - i * 4) as u8).collect();

        let mut rgb_simd = vec![0u8; 96];
        yuv420_to_rgb_row(&y, &u, &v, &mut rgb_simd);

        for i in 0..32 {
            let y_val = y[i];
            let u_val = u[i / 2];
            let v_val = v[i / 2];
            let (r_scalar, g_scalar, b_scalar) = yuv_to_rgb_scalar(y_val, u_val, v_val);

            assert_eq!(rgb_simd[i * 3], r_scalar);
            assert_eq!(rgb_simd[i * 3 + 1], g_scalar);
            assert_eq!(rgb_simd[i * 3 + 2], b_scalar);
        }
    }

    fn get_fancy_chroma_value(main: u8, secondary1: u8, secondary2: u8, tertiary: u8) -> u8 {
        let val0 = u16::from(main);
        let val1 = u16::from(secondary1);
        let val2 = u16::from(secondary2);
        let val3 = u16::from(tertiary);
        ((9 * val0 + 3 * val1 + 3 * val2 + val3 + 8) / 16) as u8
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_fancy_upsample_8_pairs() {
        if X64V3Token::summon().is_none() {
            return;
        }

        let y_row: [u8; 16] = [
            77, 162, 202, 185, 28, 13, 199, 182, 135, 147, 164, 135, 66, 27, 171, 130,
        ];
        let u_row_1: [u8; 9] = [34, 101, 84, 123, 163, 90, 110, 140, 120];
        let u_row_2: [u8; 9] = [123, 163, 133, 150, 100, 80, 95, 105, 115];
        let v_row_1: [u8; 9] = [97, 167, 150, 149, 23, 45, 67, 89, 100];
        let v_row_2: [u8; 9] = [149, 23, 86, 100, 120, 55, 75, 95, 110];

        let mut rgb_simd = [0u8; 48];
        fancy_upsample_8_pairs(
            &y_row,
            &u_row_1,
            &u_row_2,
            &v_row_1,
            &v_row_2,
            &mut rgb_simd,
        );

        let mut rgb_scalar = [0u8; 48];
        for i in 0..8 {
            let u_diag1 =
                get_fancy_chroma_value(u_row_1[i], u_row_1[i + 1], u_row_2[i], u_row_2[i + 1]);
            let v_diag1 =
                get_fancy_chroma_value(v_row_1[i], v_row_1[i + 1], v_row_2[i], v_row_2[i + 1]);
            let u_diag2 =
                get_fancy_chroma_value(u_row_1[i + 1], u_row_1[i], u_row_2[i + 1], u_row_2[i]);
            let v_diag2 =
                get_fancy_chroma_value(v_row_1[i + 1], v_row_1[i], v_row_2[i + 1], v_row_2[i]);

            let (r1, g1, b1) = yuv_to_rgb_scalar(y_row[i * 2], u_diag1, v_diag1);
            let (r2, g2, b2) = yuv_to_rgb_scalar(y_row[i * 2 + 1], u_diag2, v_diag2);

            rgb_scalar[i * 6] = r1;
            rgb_scalar[i * 6 + 1] = g1;
            rgb_scalar[i * 6 + 2] = b1;
            rgb_scalar[i * 6 + 3] = r2;
            rgb_scalar[i * 6 + 4] = g2;
            rgb_scalar[i * 6 + 5] = b2;
        }

        for i in 0..16 {
            assert_eq!(rgb_simd[i * 3], rgb_scalar[i * 3]);
            assert_eq!(rgb_simd[i * 3 + 1], rgb_scalar[i * 3 + 1]);
            assert_eq!(rgb_simd[i * 3 + 2], rgb_scalar[i * 3 + 2]);
        }
    }
}

// ============================================================================
// NEON YUV->RGB conversion (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod yuv_neon_impl {
    use super::*;

    // YUV to RGB conversion constants (matching libwebp's upsampling_neon.c):
    // These are used with vqdmulhq_lane_s16 which computes (a*b*2) >> 16
    // So effective multiply is coefficient/32768.
    //
    // R = (19077 * y             + 26149 * v - 14234) >> 6
    // G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
    // B = (19077 * y + 33050 * u             - 17685) >> 6
    //
    // With vqdmulhq: val << 7 gives val*128, then vqdmulhq(val*128, coeff) = (val*128*coeff*2) >> 16
    // = (val * coeff) >> 8, which matches mulhi(val, coeff).
    // Then shift right by 6 for final result.
    const K_COEFFS1: [i16; 4] = [19077, 26149, 6419, 13320];

    // Rounders for YUV→RGB (same as libwebp)
    const R_ROUNDER: i16 = -14234;
    const G_ROUNDER: i16 = 8708;
    const B_ROUNDER: i16 = -17685;

    // B channel uses 33050 which doesn't fit in i16 for vqdmulhq.
    // libwebp splits it: 33050 = 32768 + 282, and 32768 * x / 32768 = x
    // So B0 = vqdmulhq_n_s16(U0, 282) and then B3 = B2 + U0
    // (the U0 addition accounts for the 32768/32768 = 1.0 factor)
    const B_MULT_EXTRA: i16 = 282;

    /// Convert 16 YUV444 pixels to RGB and store as interleaved RGBRGB...
    /// Uses NEON vst3q_u8 for hardware-accelerated interleaving.
    ///
    /// Input: 16 Y values, 16 U values (upsampled), 16 V values (upsampled)
    /// Output: 48 bytes of interleaved RGB

    #[rite]
    fn convert_and_store_rgb16_neon(
        _token: NeonToken,
        y_vals: uint8x16_t,
        u_vals: uint8x16_t,
        v_vals: uint8x16_t,
        rgb: &mut [u8; 48],
    ) {
        let coeff1 = simd_mem::vld1_s16(&K_COEFFS1);
        let r_rounder = vdupq_n_s16(R_ROUNDER);
        let g_rounder = vdupq_n_s16(G_ROUNDER);
        let b_rounder = vdupq_n_s16(B_ROUNDER);

        // Process low 8 pixels
        let y_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(y_vals), 7));
        let u_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(u_vals), 7));
        let v_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(v_vals), 7));

        let y1_lo = vqdmulhq_lane_s16(y_lo, coeff1, 0); // Y * 19077
        let r0_lo = vqdmulhq_lane_s16(v_lo, coeff1, 1); // V * 26149
        let g0_lo = vqdmulhq_lane_s16(u_lo, coeff1, 2); // U * 6419
        let g1_lo = vqdmulhq_lane_s16(v_lo, coeff1, 3); // V * 13320
        let b0_lo = vqdmulhq_n_s16(u_lo, B_MULT_EXTRA); // U * 282

        let r1_lo = vqaddq_s16(y1_lo, r_rounder);
        let g2_lo = vqaddq_s16(y1_lo, g_rounder);
        let b1_lo = vqaddq_s16(y1_lo, b_rounder);

        let r2_lo = vqaddq_s16(r0_lo, r1_lo);
        let g3_lo = vqaddq_s16(g0_lo, g1_lo);
        let b2_lo = vqaddq_s16(b0_lo, b1_lo);
        let g4_lo = vqsubq_s16(g2_lo, g3_lo);
        let b3_lo = vqaddq_s16(b2_lo, u_lo); // + U accounts for 32768/32768

        let r_lo = vqshrun_n_s16(r2_lo, 6);
        let g_lo = vqshrun_n_s16(g4_lo, 6);
        let b_lo = vqshrun_n_s16(b3_lo, 6);

        // Process high 8 pixels
        let y_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(y_vals), 7));
        let u_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(u_vals), 7));
        let v_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(v_vals), 7));

        let y1_hi = vqdmulhq_lane_s16(y_hi, coeff1, 0);
        let r0_hi = vqdmulhq_lane_s16(v_hi, coeff1, 1);
        let g0_hi = vqdmulhq_lane_s16(u_hi, coeff1, 2);
        let g1_hi = vqdmulhq_lane_s16(v_hi, coeff1, 3);
        let b0_hi = vqdmulhq_n_s16(u_hi, B_MULT_EXTRA);

        let r1_hi = vqaddq_s16(y1_hi, r_rounder);
        let g2_hi = vqaddq_s16(y1_hi, g_rounder);
        let b1_hi = vqaddq_s16(y1_hi, b_rounder);

        let r2_hi = vqaddq_s16(r0_hi, r1_hi);
        let g3_hi = vqaddq_s16(g0_hi, g1_hi);
        let b2_hi = vqaddq_s16(b0_hi, b1_hi);
        let g4_hi = vqsubq_s16(g2_hi, g3_hi);
        let b3_hi = vqaddq_s16(b2_hi, u_hi);

        let r_hi = vqshrun_n_s16(r2_hi, 6);
        let g_hi = vqshrun_n_s16(g4_hi, 6);
        let b_hi = vqshrun_n_s16(b3_hi, 6);

        // Combine low/high halves
        let r16 = vcombine_u8(r_lo, r_hi);
        let g16 = vcombine_u8(g_lo, g_hi);
        let b16 = vcombine_u8(b_lo, b_hi);

        // Interleaved store: RGBRGBRGB... (48 bytes for 16 pixels)
        let rgb_val = uint8x16x3_t(r16, g16, b16);
        simd_mem::vst3q_u8(rgb, rgb_val);
    }

    /// Upsample 8 chroma pixels from two rows into 16 upsampled values.
    /// Implements libwebp's UPSAMPLE_16PIXELS macro using NEON.
    ///
    /// Input: 9 pixels from each of two rows (a[0..9], c[0..9])
    /// The overlap (9 vs 8) provides the neighbor for bilinear filtering.
    ///
    /// Output: 16 upsampled values stored interleaved (diag1[0], diag2[0], diag1[1], ...)
    /// where diag1[i] = (9*a[i] + 3*b[i] + 3*c[i] + d[i] + 8) / 16
    ///       diag2[i] = (9*b[i] + 3*a[i] + 3*d[i] + c[i] + 8) / 16
    /// a=row1[0..8], b=row1[1..9], c=row2[0..8], d=row2[1..9]

    #[rite]
    fn upsample_16pixels_neon(
        _token: NeonToken,
        a: uint8x8_t,
        b: uint8x8_t,
        c: uint8x8_t,
        d: uint8x8_t,
    ) -> uint8x16_t {
        // Compute (a + b + c + d)
        let ad = vaddl_u8(a, d); // u16
        let bc = vaddl_u8(b, c); // u16
        let abcd = vaddq_u16(ad, bc); // u16

        // 3a + b + c + 3d = abcd + 2*ad
        let al = vaddq_u16(abcd, vshlq_n_u16(ad, 1));
        // a + 3b + 3c + d = abcd + 2*bc
        let bl = vaddq_u16(abcd, vshlq_n_u16(bc, 1));

        // Divide by 8 (shift right 3)
        let diag2 = vshrn_n_u16(al, 3); // (3a + b + c + 3d) / 8
        let diag1 = vshrn_n_u16(bl, 3); // (a + 3b + 3c + d) / 8

        // Final result with rounding: vrhadd = (a + b + 1) / 2
        let a_out = vrhadd_u8(a, diag1); // (9a + 3b + 3c + d + 8) / 16
        let b_out = vrhadd_u8(b, diag2); // (3a + 9b + c + 3d + 8) / 16

        // Interleave A and B: [A[0], B[0], A[1], B[1], ...]
        // vzip_u8 on D-registers gives two D-registers; combine into Q-register
        let interleaved = vzip_u8(a_out, b_out);
        vcombine_u8(interleaved.0, interleaved.1)
    }

    /// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV→RGB.
    /// This is the NEON equivalent of fancy_upsample_16_pairs_with_token.
    ///
    /// Takes: 32 Y bytes, 17 U/V bytes per row (overlapping window), 96 RGB bytes output.

    #[arcane]
    pub(crate) fn fancy_upsample_16_pairs_neon(
        _token: NeonToken,
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        rgb: &mut [u8],
    ) {
        // Bounds check at entry
        let y: &[u8; 32] = y_row[..32].try_into().unwrap();
        let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
        let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
        let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
        let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
        let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

        fancy_upsample_16_pairs_inner_neon(_token, y, u1, u2, v1, v2, out);
    }

    #[rite]
    fn fancy_upsample_16_pairs_inner_neon(
        _token: NeonToken,
        y_row: &[u8; 32],
        u_row_1: &[u8; 17],
        u_row_2: &[u8; 17],
        v_row_1: &[u8; 17],
        v_row_2: &[u8; 17],
        rgb: &mut [u8; 96],
    ) {
        // Load U rows: a=row1[0..8], b=row1[1..9], then second batch [8..16], [9..17]
        let u_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[0..8]).unwrap());
        let u_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[1..9]).unwrap());
        let u_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[0..8]).unwrap());
        let u_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[1..9]).unwrap());

        let u_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[8..16]).unwrap());
        let u_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[9..17]).unwrap());
        let u_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[8..16]).unwrap());
        let u_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[9..17]).unwrap());

        // Same for V
        let v_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[0..8]).unwrap());
        let v_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[1..9]).unwrap());
        let v_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[0..8]).unwrap());
        let v_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[1..9]).unwrap());

        let v_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[8..16]).unwrap());
        let v_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[9..17]).unwrap());
        let v_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[8..16]).unwrap());
        let v_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[9..17]).unwrap());

        // Upsample: 8 chroma → 16 Y-aligned values for each batch
        let u_up0 = upsample_16pixels_neon(_token, u_a0, u_b0, u_c0, u_d0);
        let u_up1 = upsample_16pixels_neon(_token, u_a1, u_b1, u_c1, u_d1);
        let v_up0 = upsample_16pixels_neon(_token, v_a0, v_b0, v_c0, v_d0);
        let v_up1 = upsample_16pixels_neon(_token, v_a1, v_b1, v_c1, v_d1);

        // Load Y
        let y0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&y_row[0..16]).unwrap());
        let y1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&y_row[16..32]).unwrap());

        // Convert and store first 16 pixels
        let (rgb_0, rgb_1) = rgb.split_at_mut(48);
        convert_and_store_rgb16_neon(
            _token,
            y0,
            u_up0,
            v_up0,
            <&mut [u8; 48]>::try_from(rgb_0).unwrap(),
        );
        // Convert and store second 16 pixels
        convert_and_store_rgb16_neon(
            _token,
            y1,
            u_up1,
            v_up1,
            <&mut [u8; 48]>::try_from(rgb_1).unwrap(),
        );
    }

    /// Process 8 pixel pairs (16 Y pixels) with fancy upsampling and YUV→RGB.
    /// This is the NEON equivalent of fancy_upsample_8_pairs_with_token.
    ///
    /// Takes: 16 Y bytes, 9 U/V bytes per row (overlapping window), 48 RGB bytes output.

    #[arcane]
    pub(crate) fn fancy_upsample_8_pairs_neon(
        _token: NeonToken,
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        rgb: &mut [u8],
    ) {
        let y: &[u8; 16] = y_row[..16].try_into().unwrap();
        let u1: &[u8; 9] = u_row_1[..9].try_into().unwrap();
        let u2: &[u8; 9] = u_row_2[..9].try_into().unwrap();
        let v1: &[u8; 9] = v_row_1[..9].try_into().unwrap();
        let v2: &[u8; 9] = v_row_2[..9].try_into().unwrap();
        let out: &mut [u8; 48] = (&mut rgb[..48]).try_into().unwrap();

        fancy_upsample_8_pairs_inner_neon(_token, y, u1, u2, v1, v2, out);
    }

    #[rite]
    fn fancy_upsample_8_pairs_inner_neon(
        _token: NeonToken,
        y_row: &[u8; 16],
        u_row_1: &[u8; 9],
        u_row_2: &[u8; 9],
        v_row_1: &[u8; 9],
        v_row_2: &[u8; 9],
        rgb: &mut [u8; 48],
    ) {
        let u_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[0..8]).unwrap());
        let u_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[1..9]).unwrap());
        let u_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[0..8]).unwrap());
        let u_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[1..9]).unwrap());

        let v_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[0..8]).unwrap());
        let v_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[1..9]).unwrap());
        let v_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[0..8]).unwrap());
        let v_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[1..9]).unwrap());

        let u_up = upsample_16pixels_neon(_token, u_a, u_b, u_c, u_d);
        let v_up = upsample_16pixels_neon(_token, v_a, v_b, v_c, v_d);

        let y_vec = simd_mem::vld1q_u8(y_row);

        convert_and_store_rgb16_neon(_token, y_vec, u_up, v_up, rgb);
    }

    /// Convert a row of YUV420 to RGB using NEON.
    /// Simple (non-fancy) upsampling: each U/V value maps to 2 adjacent Y pixels.
    /// Processes 16 pixels at a time.

    #[arcane]
    pub(crate) fn yuv420_to_rgb_row_neon(
        _token: NeonToken,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        dst: &mut [u8],
    ) {
        let len = y.len();
        assert!(u.len() >= len.div_ceil(2));
        assert!(v.len() >= len.div_ceil(2));
        assert!(dst.len() >= len * 3);

        let mut n = 0usize;

        // Process 16 pixels at a time
        while n + 16 <= len {
            let y_arr = <&[u8; 16]>::try_from(&y[n..n + 16]).unwrap();
            let u_arr = <&[u8; 8]>::try_from(&u[n / 2..n / 2 + 8]).unwrap();
            let v_arr = <&[u8; 8]>::try_from(&v[n / 2..n / 2 + 8]).unwrap();
            let dst_arr = <&mut [u8; 48]>::try_from(&mut dst[n * 3..n * 3 + 48]).unwrap();

            let y_vec = simd_mem::vld1q_u8(y_arr);

            // Replicate each U/V for 2 adjacent Y pixels: [u0,u0,u1,u1,...u7,u7]
            let u_d = simd_mem::vld1_u8(u_arr);
            let v_d = simd_mem::vld1_u8(v_arr);
            let u_zip = vzip_u8(u_d, u_d);
            let v_zip = vzip_u8(v_d, v_d);
            let u_dup = vcombine_u8(u_zip.0, u_zip.1);
            let v_dup = vcombine_u8(v_zip.0, v_zip.1);

            convert_and_store_rgb16_neon(_token, y_vec, u_dup, v_dup, dst_arr);
            n += 16;
        }

        // Scalar fallback for remainder
        while n < len {
            let y_val = y[n];
            let u_val = u[n / 2];
            let v_val = v[n / 2];
            let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
            dst[n * 3] = r;
            dst[n * 3 + 1] = g;
            dst[n * 3 + 2] = b;
            n += 1;
        }
    }

    /// Scalar fallback for YUV to RGB conversion (single pixel).
    #[inline]
    fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        fn mulhi(val: u8, coeff: u16) -> i32 {
            ((u32::from(val) * u32::from(coeff)) >> 8) as i32
        }
        fn clip(v: i32) -> u8 {
            (v >> 6).clamp(0, 255) as u8
        }
        let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
        let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
        let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
        (r, g, b)
    }
} // mod yuv_neon_impl

#[cfg(target_arch = "aarch64")]
pub(crate) use yuv_neon_impl::*;

// ============================================================================
// WASM SIMD128 YUV->RGB conversion (wasm32)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod yuv_wasm_impl {
    use super::*;

    // YUV to RGB conversion constants (matching libwebp):
    // R = (19077 * y             + 26149 * v - 14234) >> 6
    // G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
    // B = (19077 * y + 33050 * u             - 17685) >> 6
    const Y_COEFF: i16 = 19077;
    const V_TO_R: i16 = 26149;
    const U_TO_G: i16 = 6419;
    const V_TO_G: i16 = 13320;
    const R_ROUNDER: i16 = -14234;
    const G_ROUNDER: i16 = 8708;
    const B_ROUNDER: i16 = -17685;
    // B channel: 33050 = 32768 + 282. 32768/32768 = 1.0, handled by adding U directly.
    const B_MULT_EXTRA: i16 = 282;

    // =============================================================================
    // Load/store helpers
    // =============================================================================

    #[inline(always)]
    fn load_u8x16(a: &[u8; 16]) -> v128 {
        u8x16(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
            a[14], a[15],
        )
    }

    #[inline(always)]
    fn load_u8x8_low(a: &[u8; 8]) -> v128 {
        u8x16(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], 0, 0, 0, 0, 0, 0, 0, 0,
        )
    }

    // =============================================================================
    // Core YUV → RGB conversion (8 pixels at a time, i16 arithmetic)
    // =============================================================================

    /// Multiply-high for i16: (a * b) >> 8, using extending multiply
    /// This matches libwebp's mulhi pattern.
    #[inline(always)]
    fn mulhi_i16x8(a: v128, coeff: i16) -> v128 {
        let coeff_v = i16x8_splat(coeff);
        let lo = i32x4_extmul_low_i16x8(a, coeff_v);
        let hi = i32x4_extmul_high_i16x8(a, coeff_v);
        let lo_shifted = i32x4_shr(lo, 8);
        let hi_shifted = i32x4_shr(hi, 8);
        i16x8_narrow_i32x4(lo_shifted, hi_shifted)
    }

    /// Convert 8 YUV444 pixels to 8 RGB pixels (returned as three i16x8 vectors clamped to u8 range)
    #[inline(always)]
    fn convert_yuv_to_rgb_8(y: v128, u: v128, v: v128) -> (v128, v128, v128) {
        let y1 = mulhi_i16x8(y, Y_COEFF);
        let r0 = mulhi_i16x8(v, V_TO_R);
        let g0 = mulhi_i16x8(u, U_TO_G);
        let g1 = mulhi_i16x8(v, V_TO_G);
        let b0 = mulhi_i16x8(u, B_MULT_EXTRA);

        let r_round = i16x8_splat(R_ROUNDER);
        let g_round = i16x8_splat(G_ROUNDER);
        let b_round = i16x8_splat(B_ROUNDER);

        let r1 = i16x8_add_sat(y1, r_round);
        let g2 = i16x8_add_sat(y1, g_round);
        let b1 = i16x8_add_sat(y1, b_round);

        let r2 = i16x8_add_sat(r0, r1);
        let g3 = i16x8_add_sat(g0, g1);
        let b2 = i16x8_add_sat(b0, b1);
        let g4 = i16x8_sub_sat(g2, g3);
        let b3 = i16x8_add_sat(b2, u); // + U accounts for 32768/32768

        // Shift right 6 and clamp to [0, 255]
        let r = i16x8_shr(r2, 6);
        let g = i16x8_shr(g4, 6);
        let b = i16x8_shr(b3, 6);

        (r, g, b)
    }

    /// Convert 16 YUV444 pixels to RGB and store as interleaved RGBRGB... (48 bytes)
    #[inline(always)]
    fn convert_and_store_rgb16(y_vals: v128, u_vals: v128, v_vals: v128, rgb: &mut [u8; 48]) {
        // Process low 8 pixels: extend u8 to i16 (zero-extend, then treat as signed)
        let y_lo = u16x8_extend_low_u8x16(y_vals);
        let u_lo = u16x8_extend_low_u8x16(u_vals);
        let v_lo = u16x8_extend_low_u8x16(v_vals);
        let (r_lo, g_lo, b_lo) = convert_yuv_to_rgb_8(y_lo, u_lo, v_lo);

        // Process high 8 pixels
        let y_hi = u16x8_extend_high_u8x16(y_vals);
        let u_hi = u16x8_extend_high_u8x16(u_vals);
        let v_hi = u16x8_extend_high_u8x16(v_vals);
        let (r_hi, g_hi, b_hi) = convert_yuv_to_rgb_8(y_hi, u_hi, v_hi);

        // Pack i16 → u8 with saturation
        let r16 = u8x16_narrow_i16x8(r_lo, r_hi);
        let g16 = u8x16_narrow_i16x8(g_lo, g_hi);
        let b16 = u8x16_narrow_i16x8(b_lo, b_hi);

        // Interleave RGB: we need [R0,G0,B0, R1,G1,B1, ...]
        // WASM doesn't have vst3q_u8, so we extract lanes individually.
        // We process in chunks of 16 bytes (which holds ~5.3 RGB triplets).
        // Instead, interleave by extracting lanes.
        for i in 0..16 {
            rgb[i * 3] = u8x16_extract_lane_runtime(r16, i);
            rgb[i * 3 + 1] = u8x16_extract_lane_runtime(g16, i);
            rgb[i * 3 + 2] = u8x16_extract_lane_runtime(b16, i);
        }
    }

    /// Extract a lane from u8x16 at runtime index
    #[inline(always)]
    fn u8x16_extract_lane_runtime(v: v128, i: usize) -> u8 {
        match i {
            0 => u8x16_extract_lane::<0>(v),
            1 => u8x16_extract_lane::<1>(v),
            2 => u8x16_extract_lane::<2>(v),
            3 => u8x16_extract_lane::<3>(v),
            4 => u8x16_extract_lane::<4>(v),
            5 => u8x16_extract_lane::<5>(v),
            6 => u8x16_extract_lane::<6>(v),
            7 => u8x16_extract_lane::<7>(v),
            8 => u8x16_extract_lane::<8>(v),
            9 => u8x16_extract_lane::<9>(v),
            10 => u8x16_extract_lane::<10>(v),
            11 => u8x16_extract_lane::<11>(v),
            12 => u8x16_extract_lane::<12>(v),
            13 => u8x16_extract_lane::<13>(v),
            14 => u8x16_extract_lane::<14>(v),
            15 => u8x16_extract_lane::<15>(v),
            _ => 0,
        }
    }

    // =============================================================================
    // Upsampling
    // =============================================================================

    /// Upsample 8 chroma pixels from two rows into 16 upsampled values.
    /// Implements libwebp's UPSAMPLE_16PIXELS using WASM SIMD.
    ///
    /// diag1[i] = (9*a[i] + 3*b[i] + 3*c[i] + d[i] + 8) / 16
    /// diag2[i] = (9*b[i] + 3*a[i] + 3*d[i] + c[i] + 8) / 16
    #[inline(always)]
    fn upsample_16pixels(a: v128, b: v128, c: v128, d: v128) -> v128 {
        // a, b, c, d are u8 values in low 8 lanes (zero-extended to u16 for arithmetic)
        let a16 = u16x8_extend_low_u8x16(a);
        let b16 = u16x8_extend_low_u8x16(b);
        let c16 = u16x8_extend_low_u8x16(c);
        let d16 = u16x8_extend_low_u8x16(d);

        // ad = a + d, bc = b + c
        let ad = i16x8_add(a16, d16);
        let bc = i16x8_add(b16, c16);
        let abcd = i16x8_add(ad, bc);

        // al = abcd + 2*ad = 3a+b+c+3d, bl = abcd + 2*bc = a+3b+3c+d
        let al = i16x8_add(abcd, i16x8_shl(ad, 1));
        let bl = i16x8_add(abcd, i16x8_shl(bc, 1));

        // Divide by 8
        let diag2 = u16x8_shr(al, 3); // (3a + b + c + 3d) / 8
        let diag1 = u16x8_shr(bl, 3); // (a + 3b + 3c + d) / 8

        // Pack back to u8
        let diag2_u8 = u8x16_narrow_i16x8(diag2, diag2);
        let diag1_u8 = u8x16_narrow_i16x8(diag1, diag1);

        // Final rounding average with original values
        let a_out = u8x16_avgr(a, diag1_u8); // (9a + 3b + 3c + d + 8) / 16
        let b_out = u8x16_avgr(b, diag2_u8); // (3a + 9b + c + 3d + 8) / 16

        // Interleave: [A[0], B[0], A[1], B[1], ...]
        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a_out, b_out)
    }

    // =============================================================================
    // Public entry points
    // =============================================================================

    /// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV→RGB.
    #[arcane]
    pub(crate) fn fancy_upsample_16_pairs_wasm(
        _token: Wasm128Token,
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        rgb: &mut [u8],
    ) {
        let y: &[u8; 32] = y_row[..32].try_into().unwrap();
        let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
        let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
        let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
        let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
        let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

        // Load U rows: a=row1[0..8], b=row1[1..9], etc.
        let u_a0 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[0..8]).unwrap());
        let u_b0 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[1..9]).unwrap());
        let u_c0 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[0..8]).unwrap());
        let u_d0 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[1..9]).unwrap());

        let u_a1 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[8..16]).unwrap());
        let u_b1 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[9..17]).unwrap());
        let u_c1 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[8..16]).unwrap());
        let u_d1 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[9..17]).unwrap());

        let v_a0 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[0..8]).unwrap());
        let v_b0 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[1..9]).unwrap());
        let v_c0 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[0..8]).unwrap());
        let v_d0 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[1..9]).unwrap());

        let v_a1 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[8..16]).unwrap());
        let v_b1 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[9..17]).unwrap());
        let v_c1 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[8..16]).unwrap());
        let v_d1 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[9..17]).unwrap());

        let u_up0 = upsample_16pixels(u_a0, u_b0, u_c0, u_d0);
        let u_up1 = upsample_16pixels(u_a1, u_b1, u_c1, u_d1);
        let v_up0 = upsample_16pixels(v_a0, v_b0, v_c0, v_d0);
        let v_up1 = upsample_16pixels(v_a1, v_b1, v_c1, v_d1);

        let y0 = load_u8x16(<&[u8; 16]>::try_from(&y[0..16]).unwrap());
        let y1 = load_u8x16(<&[u8; 16]>::try_from(&y[16..32]).unwrap());

        let (rgb_0, rgb_1) = out.split_at_mut(48);
        convert_and_store_rgb16(y0, u_up0, v_up0, <&mut [u8; 48]>::try_from(rgb_0).unwrap());
        convert_and_store_rgb16(y1, u_up1, v_up1, <&mut [u8; 48]>::try_from(rgb_1).unwrap());
    }

    /// Process 8 pixel pairs (16 Y pixels) with fancy upsampling and YUV→RGB.
    #[arcane]
    pub(crate) fn fancy_upsample_8_pairs_wasm(
        _token: Wasm128Token,
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        rgb: &mut [u8],
    ) {
        let y: &[u8; 16] = y_row[..16].try_into().unwrap();
        let u1: &[u8; 9] = u_row_1[..9].try_into().unwrap();
        let u2: &[u8; 9] = u_row_2[..9].try_into().unwrap();
        let v1: &[u8; 9] = v_row_1[..9].try_into().unwrap();
        let v2: &[u8; 9] = v_row_2[..9].try_into().unwrap();
        let out: &mut [u8; 48] = (&mut rgb[..48]).try_into().unwrap();

        let u_a = load_u8x8_low(<&[u8; 8]>::try_from(&u1[0..8]).unwrap());
        let u_b = load_u8x8_low(<&[u8; 8]>::try_from(&u1[1..9]).unwrap());
        let u_c = load_u8x8_low(<&[u8; 8]>::try_from(&u2[0..8]).unwrap());
        let u_d = load_u8x8_low(<&[u8; 8]>::try_from(&u2[1..9]).unwrap());

        let v_a = load_u8x8_low(<&[u8; 8]>::try_from(&v1[0..8]).unwrap());
        let v_b = load_u8x8_low(<&[u8; 8]>::try_from(&v1[1..9]).unwrap());
        let v_c = load_u8x8_low(<&[u8; 8]>::try_from(&v2[0..8]).unwrap());
        let v_d = load_u8x8_low(<&[u8; 8]>::try_from(&v2[1..9]).unwrap());

        let u_up = upsample_16pixels(u_a, u_b, u_c, u_d);
        let v_up = upsample_16pixels(v_a, v_b, v_c, v_d);
        let y_vec = load_u8x16(y);

        convert_and_store_rgb16(y_vec, u_up, v_up, out);
    }

    /// Convert a row of YUV420 to RGB using WASM SIMD.
    /// Simple (non-fancy) upsampling: each U/V value maps to 2 adjacent Y pixels.
    #[arcane]
    pub(crate) fn yuv420_to_rgb_row_wasm(
        _token: Wasm128Token,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        dst: &mut [u8],
    ) {
        let len = y.len();
        assert!(u.len() >= len.div_ceil(2));
        assert!(v.len() >= len.div_ceil(2));
        assert!(dst.len() >= len * 3);

        let mut n = 0usize;

        while n + 16 <= len {
            let y_arr = <&[u8; 16]>::try_from(&y[n..n + 16]).unwrap();
            let u_arr = <&[u8; 8]>::try_from(&u[n / 2..n / 2 + 8]).unwrap();
            let v_arr = <&[u8; 8]>::try_from(&v[n / 2..n / 2 + 8]).unwrap();
            let dst_arr = <&mut [u8; 48]>::try_from(&mut dst[n * 3..n * 3 + 48]).unwrap();

            let y_vec = load_u8x16(y_arr);

            // Replicate each U/V for 2 adjacent Y pixels: [u0,u0,u1,u1,...u7,u7]
            let u_lo = load_u8x8_low(u_arr);
            let v_lo = load_u8x8_low(v_arr);
            let u_dup = i8x16_shuffle::<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>(u_lo, u_lo);
            let v_dup = i8x16_shuffle::<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>(v_lo, v_lo);

            convert_and_store_rgb16(y_vec, u_dup, v_dup, dst_arr);
            n += 16;
        }

        // Scalar fallback for remainder
        while n < len {
            let y_val = y[n];
            let u_val = u[n / 2];
            let v_val = v[n / 2];
            let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
            dst[n * 3] = r;
            dst[n * 3 + 1] = g;
            dst[n * 3 + 2] = b;
            n += 1;
        }
    }

    /// Scalar fallback for YUV to RGB conversion (single pixel).
    #[inline]
    fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        fn mulhi(val: u8, coeff: u16) -> i32 {
            ((u32::from(val) * u32::from(coeff)) >> 8) as i32
        }
        fn clip(v: i32) -> u8 {
            (v >> 6).clamp(0, 255) as u8
        }
        let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
        let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
        let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
        (r, g, b)
    }
} // mod yuv_wasm_impl

#[cfg(target_arch = "wasm32")]
pub(crate) use yuv_wasm_impl::*;
