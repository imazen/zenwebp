//! Fused fancy-upsample + YUV→RGB kernel, bit-exact with libwebp.
//!
//! Replaces the two-stage pipeline (upsample chroma → convert YUV444 → RGB)
//! with a single pass that keeps everything in registers. Uses archmage SIMD
//! with `incant!` dispatch across V3 (AVX2+FMA), NEON, and WASM128 tiers.
//!
//! ## Exact math
//!
//! Fancy upsample weights (per libwebp):
//! ```text
//! (9*near + 3*h_neighbor + 3*v_neighbor + diag + 8) >> 4
//! ```
//!
//! YUV→RGB (BT.601, libwebp `src/dsp/yuv.h`):
//! ```text
//! mulhi(v, c) = (v as u32 * c as u32) >> 8
//! R = clip((mulhi(Y,19077) + mulhi(V,26149) - 14234) >> 6)
//! G = clip((mulhi(Y,19077) - mulhi(U,6419) - mulhi(V,13320) + 8708) >> 6)
//! B = clip((mulhi(Y,19077) + mulhi(U,33050) - 17685) >> 6)
//! clip(x) = x.clamp(0,255) as u8
//! ```

use alloc::vec::Vec;
use archmage::prelude::*;

use crate::decoder::yuv::{get_fancy_chroma_value, yuv_to_b, yuv_to_g, yuv_to_r};

// ============================================================================
// Public API
// ============================================================================

/// Convert full-frame YUV420 (with fancy chroma upsampling) to RGB or RGBA,
/// bit-exact with libwebp. `bpp` must be 3 (RGB) or 4 (RGBA).
///
/// Buffer layout:
/// - `y_stride` = macroblock-aligned luma width (typically `mbwidth * 16`)
/// - `uv_stride` = macroblock-aligned chroma width (typically `mbwidth * 8`)
/// - `ybuf` has `y_stride * mbheight * 16` bytes
/// - `ubuf`/`vbuf` have `uv_stride * (mbheight * 8 + 1)` bytes
///
/// `output` is resized to `width * height * bpp`.
pub(crate) fn yuv420_to_rgb_exact(
    ybuf: &[u8],
    ubuf: &[u8],
    vbuf: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    output: &mut Vec<u8>,
    bpp: usize,
) {
    debug_assert!(bpp == 3 || bpp == 4);
    output.resize(width * height * bpp, 0);

    if height == 0 || width == 0 {
        return;
    }

    let chroma_width = (width + 1) / 2;

    // First row: only one chroma row (mirror top = bottom)
    fill_1uv_row(
        &mut output[..width * bpp],
        &ybuf[..width],
        &ubuf[..chroma_width],
        &vbuf[..chroma_width],
        bpp,
    );

    // Interior: process pairs of Y rows sharing two adjacent chroma rows.
    // Y row 1,2 share chroma rows 0,1. Y row 3,4 share chroma rows 1,2. Etc.
    // For Y row at index r:
    //   chroma_near = chroma row (r / 2)
    //   chroma_far  = chroma row ((r + 1) / 2) for odd r, ((r - 1) / 2) for even r
    //
    // Matching the existing fill_rgb_buffer_fancy pattern:
    //   Pairs of Y rows are iterated. For each pair (y_row_1, y_row_2):
    //     y_row_1 uses u_row_1=near, u_row_2=far
    //     y_row_2 uses u_row_2=near, u_row_1=far  (swapped)
    let mut y_row_idx = 1;
    let mut uv_row_idx = 0;

    while y_row_idx + 1 < height {
        let y0_start = y_row_idx * y_stride;
        let y1_start = (y_row_idx + 1) * y_stride;
        let uv0_start = uv_row_idx * uv_stride;
        let uv1_start = (uv_row_idx + 1) * uv_stride;
        let rgb0_start = y_row_idx * width * bpp;
        let rgb1_start = (y_row_idx + 1) * width * bpp;

        let y_row_0 = &ybuf[y0_start..y0_start + width];
        let y_row_1 = &ybuf[y1_start..y1_start + width];
        let u_near = &ubuf[uv0_start..uv0_start + chroma_width];
        let u_far = &ubuf[uv1_start..uv1_start + chroma_width];
        let v_near = &vbuf[uv0_start..uv0_start + chroma_width];
        let v_far = &vbuf[uv1_start..uv1_start + chroma_width];

        // First Y row of pair: near=top chroma, far=bottom chroma
        incant!(
            fill_2uv_row(
                &mut output[rgb0_start..rgb0_start + width * bpp],
                y_row_0,
                u_near,
                u_far,
                v_near,
                v_far,
                bpp,
            ),
            [v3, neon, wasm128, scalar]
        );

        // Second Y row of pair: near=bottom chroma, far=top chroma (swapped)
        incant!(
            fill_2uv_row(
                &mut output[rgb1_start..rgb1_start + width * bpp],
                y_row_1,
                u_far,
                u_near,
                v_far,
                v_near,
                bpp,
            ),
            [v3, neon, wasm128, scalar]
        );

        y_row_idx += 2;
        uv_row_idx += 1;
    }

    // If height is even, the last row has only one chroma row (mirror bottom)
    if y_row_idx < height {
        let y_start = y_row_idx * y_stride;
        let uv_start = uv_row_idx * uv_stride;
        let rgb_start = y_row_idx * width * bpp;

        fill_1uv_row(
            &mut output[rgb_start..rgb_start + width * bpp],
            &ybuf[y_start..y_start + width],
            &ubuf[uv_start..uv_start + chroma_width],
            &vbuf[uv_start..uv_start + chroma_width],
            bpp,
        );
    }
}

/// Convert visible cache rows from one MB row directly to RGB output.
///
/// This is the streaming alternative to `yuv420_to_rgb_exact`: instead of
/// building full-frame Y/U/V buffers and converting at the end, each MB row's
/// cache is converted immediately after filtering.
///
/// # Arguments
/// - `cache_y/u/v`: the row cache buffers (with extra rows + 16/8 MB rows + padding)
/// - `cache_y_stride / cache_uv_stride`: stride between rows in cache
/// - `extra_y_rows`: number of extra Y rows at top of cache (filter context, typically 8)
/// - `mby`: current macroblock row index (0-based)
/// - `mbheight`: total number of macroblock rows
/// - `width / height`: visible pixel dimensions
/// - `output`: pre-allocated RGB/RGBA output buffer (`width * height * bpp` bytes)
/// - `bpp`: 3 (RGB) or 4 (RGBA)
/// - `prev_last_u_row / prev_last_v_row`: the last visible UV row from the
///   previous MB row's cache, used as the "far" chroma reference for the first
///   even Y row at the MB boundary
#[allow(clippy::too_many_arguments)]
pub(super) fn convert_cache_rows_to_rgb(
    cache_y: &[u8],
    cache_u: &[u8],
    cache_v: &[u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    mby: usize,
    mbheight: usize,
    width: usize,
    height: usize,
    output: &mut [u8],
    bpp: usize,
    prev_last_u_row: &[u8],
    prev_last_v_row: &[u8],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let is_first_row = mby == 0;
    let is_last_row = mby == mbheight - 1;

    // Determine which cache rows to output and their full-image Y row positions.
    let (src_y_start, num_y_rows, dst_img_y_row) = if is_first_row && is_last_row {
        (extra_y_rows, 16usize, 0usize)
    } else if is_first_row {
        (extra_y_rows, 16 - extra_y_rows, 0usize)
    } else if is_last_row {
        (0usize, extra_y_rows + 16, mby * 16 - extra_y_rows)
    } else {
        (0usize, 16usize, mby * 16 - extra_y_rows)
    };

    let chroma_width = (width + 1) / 2;

    // Clamp num_y_rows to not exceed the image height
    let num_y_rows = num_y_rows.min(height - dst_img_y_row);

    // The first image UV row stored in the cache at row 0.
    // For mby=0: the extra rows are 128-initialized (not real data), so
    //   image UV row 0 starts at cache UV row extra_uv_rows.
    //   Cache row 0 doesn't correspond to a real image UV row.
    // For mby>0: after rotate_extra_rows, cache UV row 0 =
    //   image UV row (mby * 8 - extra_uv_rows).
    let first_cache_img_uv = if is_first_row {
        // For first MB row, only rows starting at extra_uv_rows are real
        // image data. Map cache_uv_row = extra_uv_rows -> image UV 0.
        0isize - extra_uv_rows as isize
    } else {
        (mby * 8 - extra_uv_rows) as isize
    };

    // Process each visible Y row
    for i in 0..num_y_rows {
        let cache_y_row = src_y_start + i;
        let img_y_row = dst_img_y_row + i;
        let rgb_offset = img_y_row * width * bpp;

        let y_start = cache_y_row * cache_y_stride;
        let y_row = &cache_y[y_start..y_start + width];
        let rgb_row = &mut output[rgb_offset..rgb_offset + width * bpp];

        // Map image UV row to cache UV row:
        // cache_uv_row = img_uv_row - first_cache_img_uv
        // img_y_row fits in isize (max 16383, well below isize::MAX).
        let img_uv_row = (img_y_row / 2) as isize;
        let cache_uv_row_signed = img_uv_row - first_cache_img_uv;
        debug_assert!(cache_uv_row_signed >= 0, "cache UV row underflow");
        let cache_uv_row = cache_uv_row_signed as usize;

        // The full-frame fancy upsampler uses 1-UV mode (mirror) for:
        //   - Y row 0 (always)
        //   - The last Y row only when height is even (the unpaired remainder)
        // All other rows use 2-UV mode (fancy interpolation between adjacent chroma rows).
        let use_1uv = img_y_row == 0 || (img_y_row == height - 1 && height % 2 == 0);

        if use_1uv {
            // Edge row: single chroma row (mirrored)
            let uv_start = cache_uv_row * cache_uv_stride;
            fill_1uv_row(
                rgb_row,
                y_row,
                &cache_u[uv_start..uv_start + chroma_width],
                &cache_v[uv_start..uv_start + chroma_width],
                bpp,
            );
        } else if img_y_row % 2 == 1 {
            // Odd Y row: near = UV[y/2], far = UV[y/2 + 1]
            // Both are forward in the cache, always available.
            let near_start = cache_uv_row * cache_uv_stride;
            let far_start = (cache_uv_row + 1) * cache_uv_stride;

            fill_2uv_dispatch(
                rgb_row,
                y_row,
                &cache_u[near_start..near_start + chroma_width],
                &cache_u[far_start..far_start + chroma_width],
                &cache_v[near_start..near_start + chroma_width],
                &cache_v[far_start..far_start + chroma_width],
                bpp,
            );
        } else {
            // Even Y row (>0, not last): near = UV[y/2], far = UV[y/2 - 1]
            let near_start = cache_uv_row * cache_uv_stride;

            // Check if the "far" UV row is before the cache start.
            // This happens at MB row boundaries where the previous MB row's
            // last visible UV row has been rotated out.
            let img_far_uv = img_uv_row - 1;
            let cache_far_uv = img_far_uv - first_cache_img_uv;

            if cache_far_uv < 0 {
                // Far UV row is from the previous MB row — use saved boundary data
                fill_2uv_dispatch(
                    rgb_row,
                    y_row,
                    &cache_u[near_start..near_start + chroma_width],
                    &prev_last_u_row[..chroma_width],
                    &cache_v[near_start..near_start + chroma_width],
                    &prev_last_v_row[..chroma_width],
                    bpp,
                );
            } else {
                // cache_far_uv >= 0 is guaranteed by the if/else above.
                debug_assert!(cache_far_uv >= 0);
                let far_start = cache_far_uv as usize * cache_uv_stride;
                fill_2uv_dispatch(
                    rgb_row,
                    y_row,
                    &cache_u[near_start..near_start + chroma_width],
                    &cache_u[far_start..far_start + chroma_width],
                    &cache_v[near_start..near_start + chroma_width],
                    &cache_v[far_start..far_start + chroma_width],
                    bpp,
                );
            }
        }
    }
}

/// Dispatch 2-UV row conversion: use fused kernel for RGB (bpp=3), non-fused for RGBA.
#[inline(always)]
fn fill_2uv_dispatch(
    rgb_row: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    if bpp == 3 {
        crate::decoder::yuv_fused::fused_row_2uv(rgb_row, y_row, u_near, u_far, v_near, v_far);
    } else {
        incant!(
            fill_2uv_row(rgb_row, y_row, u_near, u_far, v_near, v_far, bpp),
            [v3, neon, wasm128, scalar]
        );
    }
}

/// Convert visible cache rows from one MB row to a strip buffer (relative offset 0).
///
/// Same logic as `convert_cache_rows_to_rgb`, but writes to a strip buffer
/// starting at byte 0 instead of an absolute position in a full-frame buffer.
///
/// Returns `(y_start, num_rows)` — the absolute Y row in the image where this
/// strip begins and how many rows were written.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn convert_cache_rows_to_strip(
    cache_y: &[u8],
    cache_u: &[u8],
    cache_v: &[u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    mby: usize,
    mbheight: usize,
    width: usize,
    height: usize,
    strip: &mut [u8],
    bpp: usize,
    prev_last_u_row: &[u8],
    prev_last_v_row: &[u8],
) -> (usize, usize) {
    let extra_uv_rows = extra_y_rows / 2;
    let is_first_row = mby == 0;
    let is_last_row = mby == mbheight - 1;

    let (src_y_start, num_y_rows, dst_img_y_row) = if is_first_row && is_last_row {
        (extra_y_rows, 16usize, 0usize)
    } else if is_first_row {
        (extra_y_rows, 16 - extra_y_rows, 0usize)
    } else if is_last_row {
        (0usize, extra_y_rows + 16, mby * 16 - extra_y_rows)
    } else {
        (0usize, 16usize, mby * 16 - extra_y_rows)
    };

    let chroma_width = (width + 1) / 2;
    let num_y_rows = num_y_rows.min(height - dst_img_y_row);

    let first_cache_img_uv = if is_first_row {
        0isize - extra_uv_rows as isize
    } else {
        (mby * 8 - extra_uv_rows) as isize
    };

    for i in 0..num_y_rows {
        let cache_y_row = src_y_start + i;
        let img_y_row = dst_img_y_row + i;
        // Write to strip at relative offset instead of absolute
        let rgb_offset = i * width * bpp;

        let y_start = cache_y_row * cache_y_stride;
        let y_row = &cache_y[y_start..y_start + width];
        let rgb_row = &mut strip[rgb_offset..rgb_offset + width * bpp];

        let img_uv_row = (img_y_row / 2) as isize;
        let cache_uv_row_signed = img_uv_row - first_cache_img_uv;
        debug_assert!(cache_uv_row_signed >= 0, "cache UV row underflow");
        let cache_uv_row = cache_uv_row_signed as usize;

        let use_1uv = img_y_row == 0 || (img_y_row == height - 1 && height % 2 == 0);

        if use_1uv {
            let uv_start = cache_uv_row * cache_uv_stride;
            fill_1uv_row(
                rgb_row,
                y_row,
                &cache_u[uv_start..uv_start + chroma_width],
                &cache_v[uv_start..uv_start + chroma_width],
                bpp,
            );
        } else if img_y_row % 2 == 1 {
            let near_start = cache_uv_row * cache_uv_stride;
            let far_start = (cache_uv_row + 1) * cache_uv_stride;
            incant!(
                fill_2uv_row(
                    rgb_row,
                    y_row,
                    &cache_u[near_start..near_start + chroma_width],
                    &cache_u[far_start..far_start + chroma_width],
                    &cache_v[near_start..near_start + chroma_width],
                    &cache_v[far_start..far_start + chroma_width],
                    bpp,
                ),
                [v3, neon, wasm128, scalar]
            );
        } else {
            let near_start = cache_uv_row * cache_uv_stride;
            let img_far_uv = img_uv_row - 1;
            let cache_far_uv = img_far_uv - first_cache_img_uv;

            if cache_far_uv < 0 {
                incant!(
                    fill_2uv_row(
                        rgb_row,
                        y_row,
                        &cache_u[near_start..near_start + chroma_width],
                        &prev_last_u_row[..chroma_width],
                        &cache_v[near_start..near_start + chroma_width],
                        &prev_last_v_row[..chroma_width],
                        bpp,
                    ),
                    [v3, neon, wasm128, scalar]
                );
            } else {
                debug_assert!(cache_far_uv >= 0);
                let far_start = cache_far_uv as usize * cache_uv_stride;
                incant!(
                    fill_2uv_row(
                        rgb_row,
                        y_row,
                        &cache_u[near_start..near_start + chroma_width],
                        &cache_u[far_start..far_start + chroma_width],
                        &cache_v[near_start..near_start + chroma_width],
                        &cache_v[far_start..far_start + chroma_width],
                        bpp,
                    ),
                    [v3, neon, wasm128, scalar]
                );
            }
        }
    }

    (dst_img_y_row, num_y_rows)
}

// ============================================================================
// Edge row (1 chroma row — first/last row of image)
// ============================================================================

/// Process one Y row with a single chroma row (edge: top or bottom of image).
/// The chroma row serves as both near and far (mirrored).
fn fill_1uv_row(rgb: &mut [u8], y_row: &[u8], u_row: &[u8], v_row: &[u8], bpp: usize) {
    let width = y_row.len();
    if width == 0 {
        return;
    }

    // First pixel
    write_pixel(&mut rgb[..bpp], y_row[0], u_row[0], v_row[0]);

    // Interior: pairs
    let mut yx = 1;
    let mut cx = 0;
    while yx + 1 < width && cx + 1 < u_row.len() {
        // Left pixel of pair
        let u = get_fancy_chroma_value(u_row[cx], u_row[cx + 1], u_row[cx], u_row[cx + 1]);
        let v = get_fancy_chroma_value(v_row[cx], v_row[cx + 1], v_row[cx], v_row[cx + 1]);
        write_pixel(&mut rgb[yx * bpp..(yx + 1) * bpp], y_row[yx], u, v);

        // Right pixel of pair
        let u = get_fancy_chroma_value(u_row[cx + 1], u_row[cx], u_row[cx + 1], u_row[cx]);
        let v = get_fancy_chroma_value(v_row[cx + 1], v_row[cx], v_row[cx + 1], v_row[cx]);
        write_pixel(
            &mut rgb[(yx + 1) * bpp..(yx + 2) * bpp],
            y_row[yx + 1],
            u,
            v,
        );

        yx += 2;
        cx += 1;
    }

    // Last pixel (odd width)
    if yx < width {
        let lc = u_row.len() - 1;
        write_pixel(
            &mut rgb[yx * bpp..(yx + 1) * bpp],
            y_row[yx],
            u_row[lc],
            v_row[lc],
        );
    }
}

// ============================================================================
// Interior row — scalar fallback
// ============================================================================

/// Scalar: process one Y row using two chroma rows (near + far).
fn fill_2uv_row_scalar(
    _token: archmage::ScalarToken,
    rgb: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    fill_2uv_row_generic(rgb, y_row, u_near, u_far, v_near, v_far, bpp);
}

/// Shared scalar implementation (used by all non-SIMD tiers and as tail handler).
#[inline(always)]
fn fill_2uv_row_generic(
    rgb: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    let width = y_row.len();
    let chroma_width = u_near.len();
    if width == 0 {
        return;
    }

    // First pixel (mirror left edge)
    {
        let u = get_fancy_chroma_value(u_near[0], u_near[0], u_far[0], u_far[0]);
        let v = get_fancy_chroma_value(v_near[0], v_near[0], v_far[0], v_far[0]);
        write_pixel(&mut rgb[..bpp], y_row[0], u, v);
    }

    // Interior: pairs
    let mut yx = 1;
    let mut cx = 0;
    while yx + 1 < width && cx + 1 < chroma_width {
        {
            let u = get_fancy_chroma_value(u_near[cx], u_near[cx + 1], u_far[cx], u_far[cx + 1]);
            let v = get_fancy_chroma_value(v_near[cx], v_near[cx + 1], v_far[cx], v_far[cx + 1]);
            write_pixel(&mut rgb[yx * bpp..(yx + 1) * bpp], y_row[yx], u, v);
        }
        {
            let u = get_fancy_chroma_value(u_near[cx + 1], u_near[cx], u_far[cx + 1], u_far[cx]);
            let v = get_fancy_chroma_value(v_near[cx + 1], v_near[cx], v_far[cx + 1], v_far[cx]);
            write_pixel(
                &mut rgb[(yx + 1) * bpp..(yx + 2) * bpp],
                y_row[yx + 1],
                u,
                v,
            );
        }
        yx += 2;
        cx += 1;
    }

    // Last pixel (mirror right edge)
    if yx < width {
        let lc = chroma_width - 1;
        let u = get_fancy_chroma_value(u_near[lc], u_near[lc], u_far[lc], u_far[lc]);
        let v = get_fancy_chroma_value(v_near[lc], v_near[lc], v_far[lc], v_far[lc]);
        write_pixel(&mut rgb[yx * bpp..(yx + 1) * bpp], y_row[yx], u, v);
    }
}

// ============================================================================
// x86_64 V3 (AVX2+FMA)
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn fill_2uv_row_v3(
    token: archmage::X64V3Token,
    rgb: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    x86_impl::fill_2uv_row_arcane(token, rgb, y_row, u_near, u_far, v_near, v_far, bpp);
}

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use archmage::prelude::*;

    use archmage::intrinsics::x86_64 as simd_mem;

    use super::write_pixel;
    use crate::decoder::yuv::get_fancy_chroma_value;

    /// Fancy upsample 16 chroma samples: produces (diag1, diag2).
    /// Interleave diag1/diag2 to get 32 luma-aligned chroma values.
    #[rite(v3, import_intrinsics)]
    fn fancy_upsample_16(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> (__m128i, __m128i) {
        let one = _mm_set1_epi8(1);
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

        // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
        let tmp0 = _mm_avg_epu8(k, t);
        let tmp1 = _mm_and_si128(bc, st);
        let tmp2 = _mm_xor_si128(k, t);
        let tmp3 = _mm_or_si128(tmp1, tmp2);
        let tmp4 = _mm_and_si128(tmp3, one);
        let m1 = _mm_sub_epi8(tmp0, tmp4);

        // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
        let tmp0 = _mm_avg_epu8(k, s);
        let tmp1 = _mm_and_si128(ad, st);
        let tmp2 = _mm_xor_si128(k, s);
        let tmp3 = _mm_or_si128(tmp1, tmp2);
        let tmp4 = _mm_and_si128(tmp3, one);
        let m2 = _mm_sub_epi8(tmp0, tmp4);

        let diag1 = _mm_avg_epu8(a, m1);
        let diag2 = _mm_avg_epu8(b, m2);
        (diag1, diag2)
    }

    /// Convert 8 YUV444 pixels to (R, G, B) as signed i16.
    /// Input values are u8 zero-extended into the high byte of each i16 lane
    /// (i.e. `_mm_unpacklo_epi8(zero, val)`).
    #[rite(v3, import_intrinsics)]
    fn yuv_to_rgb_8(y: __m128i, u: __m128i, v: __m128i) -> (__m128i, __m128i, __m128i) {
        let k19077 = _mm_set1_epi16(19077);
        let k26149 = _mm_set1_epi16(26149);
        let k14234 = _mm_set1_epi16(14234);
        let k33050 = _mm_set1_epi16(33050u16 as i16);
        let k17685 = _mm_set1_epi16(17685);
        let k6419 = _mm_set1_epi16(6419);
        let k13320 = _mm_set1_epi16(13320);
        let k8708 = _mm_set1_epi16(8708);

        let y1 = _mm_mulhi_epu16(y, k19077);

        // R = (Y1 + V*26149 - 14234) >> 6
        let r0 = _mm_mulhi_epu16(v, k26149);
        let r1 = _mm_sub_epi16(y1, k14234);
        let r2 = _mm_add_epi16(r1, r0);
        let r = _mm_srai_epi16(r2, 6);

        // G = (Y1 - U*6419 - V*13320 + 8708) >> 6
        let g0 = _mm_mulhi_epu16(u, k6419);
        let g1 = _mm_mulhi_epu16(v, k13320);
        let g2 = _mm_add_epi16(y1, k8708);
        let g3 = _mm_add_epi16(g0, g1);
        let g4 = _mm_sub_epi16(g2, g3);
        let g = _mm_srai_epi16(g4, 6);

        // B = (Y1 + U*33050 - 17685) >> 6
        // 33050 > 32767 so use unsigned saturating add/sub
        let b0 = _mm_mulhi_epu16(u, k33050);
        let b1 = _mm_adds_epu16(b0, y1);
        let b2 = _mm_subs_epu16(b1, k17685);
        let b = _mm_srli_epi16(b2, 6);

        (r, g, b)
    }

    /// Interleave planar R,G,B → packed 24-bit RGB (libwebp VP8PlanarTo24b).
    macro_rules! planar_to_24b_step {
        ($in0:expr, $in1:expr, $in2:expr, $in3:expr, $in4:expr, $in5:expr,
         $out0:expr, $out1:expr, $out2:expr, $out3:expr, $out4:expr, $out5:expr) => {
            let v_mask = _mm_set1_epi16(0x00ff);
            $out0 = _mm_packus_epi16(_mm_and_si128($in0, v_mask), _mm_and_si128($in1, v_mask));
            $out1 = _mm_packus_epi16(_mm_and_si128($in2, v_mask), _mm_and_si128($in3, v_mask));
            $out2 = _mm_packus_epi16(_mm_and_si128($in4, v_mask), _mm_and_si128($in5, v_mask));
            $out3 = _mm_packus_epi16(_mm_srli_epi16($in0, 8), _mm_srli_epi16($in1, 8));
            $out4 = _mm_packus_epi16(_mm_srli_epi16($in2, 8), _mm_srli_epi16($in3, 8));
            $out5 = _mm_packus_epi16(_mm_srli_epi16($in4, 8), _mm_srli_epi16($in5, 8));
        };
    }

    #[rite(v3, import_intrinsics)]
    fn planar_to_24b(
        in0: __m128i,
        in1: __m128i,
        in2: __m128i,
        in3: __m128i,
        in4: __m128i,
        in5: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
        let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5);
        let (mut o0, mut o1, mut o2, mut o3, mut o4, mut o5);
        planar_to_24b_step!(in0, in1, in2, in3, in4, in5, t0, t1, t2, t3, t4, t5);
        planar_to_24b_step!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
        planar_to_24b_step!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);
        planar_to_24b_step!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
        planar_to_24b_step!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);
        (t0, t1, t2, t3, t4, t5)
    }

    /// Upsample + convert + store 32 Y pixels (16 chroma pairs) → 96 bytes RGB.
    #[rite(v3, import_intrinsics)]
    fn process_32_rgb(
        y: &[u8; 32],
        u_near: &[u8; 17],
        u_far: &[u8; 17],
        v_near: &[u8; 17],
        v_far: &[u8; 17],
        rgb: &mut [u8; 96],
    ) {
        let u_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_near[0..16]).unwrap());
        let u_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_near[1..17]).unwrap());
        let u_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_far[0..16]).unwrap());
        let u_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_far[1..17]).unwrap());
        let v_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_near[0..16]).unwrap());
        let v_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_near[1..17]).unwrap());
        let v_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_far[0..16]).unwrap());
        let v_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_far[1..17]).unwrap());

        let (u_d1, u_d2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
        let (v_d1, v_d2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

        let u_lo = _mm_unpacklo_epi8(u_d1, u_d2);
        let u_hi = _mm_unpackhi_epi8(u_d1, u_d2);
        let v_lo = _mm_unpacklo_epi8(v_d1, v_d2);
        let v_hi = _mm_unpackhi_epi8(v_d1, v_d2);

        let y0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y[0..16]).unwrap());
        let y1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y[16..32]).unwrap());
        let zero = _mm_setzero_si128();

        let (r0, g0, b0) = yuv_to_rgb_8(
            _mm_unpacklo_epi8(zero, y0),
            _mm_unpacklo_epi8(zero, u_lo),
            _mm_unpacklo_epi8(zero, v_lo),
        );
        let (r1, g1, b1) = yuv_to_rgb_8(
            _mm_unpackhi_epi8(zero, y0),
            _mm_unpackhi_epi8(zero, u_lo),
            _mm_unpackhi_epi8(zero, v_lo),
        );
        let (r2, g2, b2) = yuv_to_rgb_8(
            _mm_unpacklo_epi8(zero, y1),
            _mm_unpacklo_epi8(zero, u_hi),
            _mm_unpacklo_epi8(zero, v_hi),
        );
        let (r3, g3, b3) = yuv_to_rgb_8(
            _mm_unpackhi_epi8(zero, y1),
            _mm_unpackhi_epi8(zero, u_hi),
            _mm_unpackhi_epi8(zero, v_hi),
        );

        let r_0 = _mm_packus_epi16(r0, r1);
        let r_1 = _mm_packus_epi16(r2, r3);
        let g_0 = _mm_packus_epi16(g0, g1);
        let g_1 = _mm_packus_epi16(g2, g3);
        let b_0 = _mm_packus_epi16(b0, b1);
        let b_1 = _mm_packus_epi16(b2, b3);

        let (o0, o1, o2, o3, o4, o5) = planar_to_24b(r_0, r_1, g_0, g_1, b_0, b_1);

        let (s0, rest) = rgb.split_at_mut(16);
        let (s1, rest) = rest.split_at_mut(16);
        let (s2, rest) = rest.split_at_mut(16);
        let (s3, rest) = rest.split_at_mut(16);
        let (s4, s5) = rest.split_at_mut(16);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s0).unwrap(), o0);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s1).unwrap(), o1);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s2).unwrap(), o2);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s3).unwrap(), o3);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s4).unwrap(), o4);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s5).unwrap(), o5);
    }

    /// Upsample + convert + store 16 Y pixels (8 chroma pairs) → 48 bytes RGB.
    #[rite(v3, import_intrinsics)]
    fn process_16_rgb(
        y: &[u8; 16],
        u_near: &[u8; 9],
        u_far: &[u8; 9],
        v_near: &[u8; 9],
        v_far: &[u8; 9],
        rgb: &mut [u8; 48],
    ) {
        macro_rules! load_8 {
            ($arr:expr, $off:expr) => {{
                let bytes: [u8; 8] = [
                    $arr[$off],
                    $arr[$off + 1],
                    $arr[$off + 2],
                    $arr[$off + 3],
                    $arr[$off + 4],
                    $arr[$off + 5],
                    $arr[$off + 6],
                    $arr[$off + 7],
                ];
                _mm_cvtsi64_si128(i64::from_le_bytes(bytes))
            }};
        }

        let (u_d1, u_d2) = fancy_upsample_16(
            load_8!(u_near, 0),
            load_8!(u_near, 1),
            load_8!(u_far, 0),
            load_8!(u_far, 1),
        );
        let (v_d1, v_d2) = fancy_upsample_16(
            load_8!(v_near, 0),
            load_8!(v_near, 1),
            load_8!(v_far, 0),
            load_8!(v_far, 1),
        );

        let u_interleaved = _mm_unpacklo_epi8(u_d1, u_d2);
        let v_interleaved = _mm_unpacklo_epi8(v_d1, v_d2);

        let y_vec = simd_mem::_mm_loadu_si128(y);
        let zero = _mm_setzero_si128();

        let (r0, g0, b0) = yuv_to_rgb_8(
            _mm_unpacklo_epi8(zero, y_vec),
            _mm_unpacklo_epi8(zero, u_interleaved),
            _mm_unpacklo_epi8(zero, v_interleaved),
        );
        let (r1, g1, b1) = yuv_to_rgb_8(
            _mm_unpackhi_epi8(zero, y_vec),
            _mm_unpackhi_epi8(zero, u_interleaved),
            _mm_unpackhi_epi8(zero, v_interleaved),
        );

        let r8 = _mm_packus_epi16(r0, r1);
        let g8 = _mm_packus_epi16(g0, g1);
        let b8 = _mm_packus_epi16(b0, b1);

        let (o0, o1, o2, _, _, _) = planar_to_24b(
            r8,
            _mm_setzero_si128(),
            g8,
            _mm_setzero_si128(),
            b8,
            _mm_setzero_si128(),
        );

        let (s0, rest) = rgb.split_at_mut(16);
        let (s1, s2) = rest.split_at_mut(16);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s0).unwrap(), o0);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s1).unwrap(), o1);
        simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(s2).unwrap(), o2);
    }

    /// Process 32 Y pixels → 128 bytes RGBA.
    #[rite(v3, import_intrinsics)]
    fn process_32_rgba(
        y: &[u8; 32],
        u_near: &[u8; 17],
        u_far: &[u8; 17],
        v_near: &[u8; 17],
        v_far: &[u8; 17],
        rgba: &mut [u8; 128],
    ) {
        let u_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_near[0..16]).unwrap());
        let u_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_near[1..17]).unwrap());
        let u_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_far[0..16]).unwrap());
        let u_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_far[1..17]).unwrap());
        let v_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_near[0..16]).unwrap());
        let v_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_near[1..17]).unwrap());
        let v_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_far[0..16]).unwrap());
        let v_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_far[1..17]).unwrap());

        let (u_d1, u_d2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
        let (v_d1, v_d2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

        let u_lo = _mm_unpacklo_epi8(u_d1, u_d2);
        let u_hi = _mm_unpackhi_epi8(u_d1, u_d2);
        let v_lo = _mm_unpacklo_epi8(v_d1, v_d2);
        let v_hi = _mm_unpackhi_epi8(v_d1, v_d2);

        let y0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y[0..16]).unwrap());
        let y1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y[16..32]).unwrap());
        let zero = _mm_setzero_si128();
        let alpha = _mm_set1_epi8(-1i8); // 0xFF

        let (r0, g0, b0) = yuv_to_rgb_8(
            _mm_unpacklo_epi8(zero, y0),
            _mm_unpacklo_epi8(zero, u_lo),
            _mm_unpacklo_epi8(zero, v_lo),
        );
        let (r1, g1, b1) = yuv_to_rgb_8(
            _mm_unpackhi_epi8(zero, y0),
            _mm_unpackhi_epi8(zero, u_lo),
            _mm_unpackhi_epi8(zero, v_lo),
        );
        let (r2, g2, b2) = yuv_to_rgb_8(
            _mm_unpacklo_epi8(zero, y1),
            _mm_unpacklo_epi8(zero, u_hi),
            _mm_unpacklo_epi8(zero, v_hi),
        );
        let (r3, g3, b3) = yuv_to_rgb_8(
            _mm_unpackhi_epi8(zero, y1),
            _mm_unpackhi_epi8(zero, u_hi),
            _mm_unpackhi_epi8(zero, v_hi),
        );

        let r_0 = _mm_packus_epi16(r0, r1);
        let r_1 = _mm_packus_epi16(r2, r3);
        let g_0 = _mm_packus_epi16(g0, g1);
        let g_1 = _mm_packus_epi16(g2, g3);
        let b_0 = _mm_packus_epi16(b0, b1);
        let b_1 = _mm_packus_epi16(b2, b3);

        // Interleave RGBA: for each group of 16 packed u8, interleave R,G,B,A
        macro_rules! store_rgba_16 {
            ($r:expr, $g:expr, $b:expr, $out:expr) => {{
                let rg_lo = _mm_unpacklo_epi8($r, $g);  // R0G0R1G1...R7G7
                let rg_hi = _mm_unpackhi_epi8($r, $g);  // R8G8R9G9...R15G15
                let ba_lo = _mm_unpacklo_epi8($b, alpha); // B0A0B1A1...B7A7
                let ba_hi = _mm_unpackhi_epi8($b, alpha); // B8A8B9A9...B15A15
                let rgba_0 = _mm_unpacklo_epi16(rg_lo, ba_lo); // R0G0B0A0 R1G1B1A1 R2G2B2A2 R3G3B3A3
                let rgba_1 = _mm_unpackhi_epi16(rg_lo, ba_lo); // R4G4B4A4 ... R7G7B7A7
                let rgba_2 = _mm_unpacklo_epi16(rg_hi, ba_hi);
                let rgba_3 = _mm_unpackhi_epi16(rg_hi, ba_hi);
                simd_mem::_mm_storeu_si128(
                    <&mut [u8; 16]>::try_from(&mut $out[0..16]).unwrap(), rgba_0);
                simd_mem::_mm_storeu_si128(
                    <&mut [u8; 16]>::try_from(&mut $out[16..32]).unwrap(), rgba_1);
                simd_mem::_mm_storeu_si128(
                    <&mut [u8; 16]>::try_from(&mut $out[32..48]).unwrap(), rgba_2);
                simd_mem::_mm_storeu_si128(
                    <&mut [u8; 16]>::try_from(&mut $out[48..64]).unwrap(), rgba_3);
            }};
        }

        store_rgba_16!(r_0, g_0, b_0, rgba[0..64]);
        store_rgba_16!(r_1, g_1, b_1, rgba[64..128]);
    }

    /// Single `#[arcane]` entry point for one interior row.
    #[arcane]
    pub(super) fn fill_2uv_row_arcane(
        _token: X64V3Token,
        rgb: &mut [u8],
        y_row: &[u8],
        u_near: &[u8],
        u_far: &[u8],
        v_near: &[u8],
        v_far: &[u8],
        bpp: usize,
    ) {
        let width = y_row.len();
        let chroma_width = u_near.len();

        if width == 0 {
            return;
        }

        // First pixel (mirror left edge)
        {
            let u = get_fancy_chroma_value(u_near[0], u_near[0], u_far[0], u_far[0]);
            let v = get_fancy_chroma_value(v_near[0], v_near[0], v_far[0], v_far[0]);
            write_pixel(&mut rgb[..bpp], y_row[0], u, v);
        }

        let mut yx: usize = 1;
        let mut cx: usize = 0;
        let mut rgb_off: usize = bpp;

        if bpp == 3 {
            // 32 Y pixels (16 chroma) per SIMD iteration
            while yx + 32 <= width && cx + 17 <= chroma_width {
                let ya: &[u8; 32] = y_row[yx..yx + 32].try_into().unwrap();
                let un: &[u8; 17] = u_near[cx..cx + 17].try_into().unwrap();
                let uf: &[u8; 17] = u_far[cx..cx + 17].try_into().unwrap();
                let vn: &[u8; 17] = v_near[cx..cx + 17].try_into().unwrap();
                let vf: &[u8; 17] = v_far[cx..cx + 17].try_into().unwrap();
                let out: &mut [u8; 96] = (&mut rgb[rgb_off..rgb_off + 96]).try_into().unwrap();
                process_32_rgb(ya, un, uf, vn, vf, out);
                yx += 32;
                cx += 16;
                rgb_off += 96;
            }

            // 16 Y pixels (8 chroma) per SIMD iteration
            while yx + 16 <= width && cx + 9 <= chroma_width {
                let ya: &[u8; 16] = y_row[yx..yx + 16].try_into().unwrap();
                let un: &[u8; 9] = u_near[cx..cx + 9].try_into().unwrap();
                let uf: &[u8; 9] = u_far[cx..cx + 9].try_into().unwrap();
                let vn: &[u8; 9] = v_near[cx..cx + 9].try_into().unwrap();
                let vf: &[u8; 9] = v_far[cx..cx + 9].try_into().unwrap();
                let out: &mut [u8; 48] = (&mut rgb[rgb_off..rgb_off + 48]).try_into().unwrap();
                process_16_rgb(ya, un, uf, vn, vf, out);
                yx += 16;
                cx += 8;
                rgb_off += 48;
            }
        } else {
            // 32 Y pixels → 128 bytes RGBA
            while yx + 32 <= width && cx + 17 <= chroma_width {
                let ya: &[u8; 32] = y_row[yx..yx + 32].try_into().unwrap();
                let un: &[u8; 17] = u_near[cx..cx + 17].try_into().unwrap();
                let uf: &[u8; 17] = u_far[cx..cx + 17].try_into().unwrap();
                let vn: &[u8; 17] = v_near[cx..cx + 17].try_into().unwrap();
                let vf: &[u8; 17] = v_far[cx..cx + 17].try_into().unwrap();
                let out: &mut [u8; 128] = (&mut rgb[rgb_off..rgb_off + 128]).try_into().unwrap();
                process_32_rgba(ya, un, uf, vn, vf, out);
                yx += 32;
                cx += 16;
                rgb_off += 128;
            }
        }

        // Scalar remainder: pairs
        while yx + 1 < width && cx + 1 < chroma_width {
            {
                let u =
                    get_fancy_chroma_value(u_near[cx], u_near[cx + 1], u_far[cx], u_far[cx + 1]);
                let v =
                    get_fancy_chroma_value(v_near[cx], v_near[cx + 1], v_far[cx], v_far[cx + 1]);
                write_pixel(&mut rgb[rgb_off..rgb_off + bpp], y_row[yx], u, v);
            }
            {
                let u =
                    get_fancy_chroma_value(u_near[cx + 1], u_near[cx], u_far[cx + 1], u_far[cx]);
                let v =
                    get_fancy_chroma_value(v_near[cx + 1], v_near[cx], v_far[cx + 1], v_far[cx]);
                write_pixel(
                    &mut rgb[rgb_off + bpp..rgb_off + 2 * bpp],
                    y_row[yx + 1],
                    u,
                    v,
                );
            }
            yx += 2;
            cx += 1;
            rgb_off += 2 * bpp;
        }

        // Last pixel (mirror right edge)
        if yx < width {
            let lc = chroma_width - 1;
            let u = get_fancy_chroma_value(u_near[lc], u_near[lc], u_far[lc], u_far[lc]);
            let v = get_fancy_chroma_value(v_near[lc], v_near[lc], v_far[lc], v_far[lc]);
            write_pixel(&mut rgb[rgb_off..rgb_off + bpp], y_row[yx], u, v);
        }
    }
}

// ============================================================================
// aarch64 NEON — fused upsample + YUV→RGB, matching x86 V3 structure
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn fill_2uv_row_neon(
    token: archmage::NeonToken,
    rgb: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    neon_impl::fill_2uv_row_arcane(token, rgb, y_row, u_near, u_far, v_near, v_far, bpp);
}

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use archmage::intrinsics::aarch64 as simd_mem;
    use archmage::prelude::*;

    use super::write_pixel;
    use crate::decoder::yuv::get_fancy_chroma_value;

    // YUV→RGB constants matching libwebp's upsampling_neon.c
    // vqdmulh computes (2*a*b) >> 16, so with a = val<<7:
    //   result = (2 * val * 128 * coeff) >> 16 = (val * coeff) >> 8
    // This matches the x86 mulhi_epu16(val<<8, coeff) = (val*coeff) >> 8.
    const K_COEFFS1: [i16; 4] = [19077, 26149, 6419, 13320];
    const R_ROUNDER: i16 = -14234;
    const G_ROUNDER: i16 = 8708;
    const B_ROUNDER: i16 = -17685;
    // 33050 = 32768 + 282; split because 33050 > i16::MAX
    const B_MULT_EXTRA: i16 = 282;

    /// Fancy upsample 8 chroma pairs → 16 luma-aligned chroma values.
    /// Matches libwebp UPSAMPLE_16PIXELS macro (NEON variant).
    /// Input: 8-element halves (near[cx..cx+8], near[cx+1..cx+9], far[..], far[..+1])
    /// Output: 16-element interleaved result.
    #[rite]
    fn upsample_16pixels_neon(
        _token: NeonToken,
        a: uint8x8_t,
        b: uint8x8_t,
        c: uint8x8_t,
        d: uint8x8_t,
    ) -> uint8x16_t {
        let one = vdup_n_u8(1);

        let s = vrhadd_u8(a, d); // (a+d+1)/2
        let t = vrhadd_u8(b, c); // (b+c+1)/2
        let st = veor_u8(s, t);
        let ad = veor_u8(a, d);
        let bc = veor_u8(b, c);

        let t1 = vorr_u8(ad, bc);
        let t2 = vorr_u8(t1, st);
        let t3 = vand_u8(t2, one);
        let t4 = vrhadd_u8(s, t);
        let k = vsub_u8(t4, t3);

        // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
        let tmp1 = vrhadd_u8(k, t);
        let tmp2 = vand_u8(bc, st);
        let tmp3 = veor_u8(k, t);
        let tmp4 = vorr_u8(tmp2, tmp3);
        let tmp5 = vand_u8(tmp4, one);
        let m1 = vsub_u8(tmp1, tmp5);

        // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
        let tmp1 = vrhadd_u8(k, s);
        let tmp2 = vand_u8(ad, st);
        let tmp3 = veor_u8(k, s);
        let tmp4 = vorr_u8(tmp2, tmp3);
        let tmp5 = vand_u8(tmp4, one);
        let m2 = vsub_u8(tmp1, tmp5);

        let diag1 = vrhadd_u8(a, m1);
        let diag2 = vrhadd_u8(b, m2);

        let zip = vzip_u8(diag1, diag2);
        vcombine_u8(zip.0, zip.1)
    }

    /// Convert 16 YUV444 pixels to interleaved RGB and store 48 bytes.
    /// Uses vst3q_u8 for hardware-accelerated RGB interleaving.
    #[rite]
    fn convert_and_store_rgb16_neon(
        _token: NeonToken,
        y_vals: uint8x16_t,
        u_vals: uint8x16_t,
        v_vals: uint8x16_t,
        rgb: &mut [u8; 48],
    ) {
        let (r, g, b) = yuv_to_rgb_16_neon(_token, y_vals, u_vals, v_vals);

        let rgb_array = uint8x16x3_t(r, g, b);
        simd_mem::vst3q_u8(rgb, rgb_array);
    }

    /// Convert 16 YUV444 pixels to interleaved RGBA and store 64 bytes.
    /// Uses vst4q_u8 for hardware-accelerated RGBA interleaving.
    #[rite]
    fn convert_and_store_rgba16_neon(
        _token: NeonToken,
        y_vals: uint8x16_t,
        u_vals: uint8x16_t,
        v_vals: uint8x16_t,
        rgba: &mut [u8; 64],
    ) {
        let (r, g, b) = yuv_to_rgb_16_neon(_token, y_vals, u_vals, v_vals);
        let a = vdupq_n_u8(0xFF);

        let rgba_array = uint8x16x4_t(r, g, b, a);
        simd_mem::vst4q_u8(rgba, rgba_array);
    }

    /// Core YUV→RGB conversion for 16 pixels. Returns (R, G, B) as uint8x16_t.
    /// Math matches libwebp upsampling_neon.c exactly.
    #[rite]
    fn yuv_to_rgb_16_neon(
        _token: NeonToken,
        y_vals: uint8x16_t,
        u_vals: uint8x16_t,
        v_vals: uint8x16_t,
    ) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
        let coeffs1 = simd_mem::vld1_s16(&K_COEFFS1);

        let y_lo = vget_low_u8(y_vals);
        let y_hi = vget_high_u8(y_vals);
        let u_lo = vget_low_u8(u_vals);
        let u_hi = vget_high_u8(u_vals);
        let v_lo = vget_low_u8(v_vals);
        let v_hi = vget_high_u8(v_vals);

        // Widen to i16 and shift left by 7 (multiply by 128)
        let y_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(y_lo));
        let y_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(y_hi));
        let u_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(u_lo));
        let u_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(u_hi));
        let v_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(v_lo));
        let v_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(v_hi));

        // Y * 19077
        let y1_lo = vqdmulhq_lane_s16::<0>(y_lo16, coeffs1);
        let y1_hi = vqdmulhq_lane_s16::<0>(y_hi16, coeffs1);

        // R = Y1 + V*26149 - 14234
        let r_rounder = vdupq_n_s16(R_ROUNDER);
        let r0_lo = vqdmulhq_lane_s16::<1>(v_lo16, coeffs1);
        let r0_hi = vqdmulhq_lane_s16::<1>(v_hi16, coeffs1);
        let r1_lo = vaddq_s16(y1_lo, r_rounder);
        let r1_hi = vaddq_s16(y1_hi, r_rounder);
        let r2_lo = vaddq_s16(r1_lo, r0_lo);
        let r2_hi = vaddq_s16(r1_hi, r0_hi);

        // G = Y1 - U*6419 - V*13320 + 8708
        let g_rounder = vdupq_n_s16(G_ROUNDER);
        let g0_lo = vqdmulhq_lane_s16::<2>(u_lo16, coeffs1);
        let g0_hi = vqdmulhq_lane_s16::<2>(u_hi16, coeffs1);
        let g1_lo = vqdmulhq_lane_s16::<3>(v_lo16, coeffs1);
        let g1_hi = vqdmulhq_lane_s16::<3>(v_hi16, coeffs1);
        let g2_lo = vaddq_s16(y1_lo, g_rounder);
        let g2_hi = vaddq_s16(y1_hi, g_rounder);
        let g3_lo = vaddq_s16(g0_lo, g1_lo);
        let g3_hi = vaddq_s16(g0_hi, g1_hi);
        let g4_lo = vsubq_s16(g2_lo, g3_lo);
        let g4_hi = vsubq_s16(g2_hi, g3_hi);

        // B = Y1 + U*33050 - 17685
        // 33050 = 32768 + 282, split: vqdmulh(U, 282) + U
        let b_rounder = vdupq_n_s16(B_ROUNDER);
        let b0_lo = vqdmulhq_n_s16(u_lo16, B_MULT_EXTRA);
        let b0_hi = vqdmulhq_n_s16(u_hi16, B_MULT_EXTRA);
        let b1_lo = vaddq_s16(b0_lo, vreinterpretq_s16_u16(vshll_n_u8::<7>(u_lo)));
        let b1_hi = vaddq_s16(b0_hi, vreinterpretq_s16_u16(vshll_n_u8::<7>(u_hi)));
        let b2_lo = vaddq_s16(y1_lo, b_rounder);
        let b2_hi = vaddq_s16(y1_hi, b_rounder);
        let b3_lo = vaddq_s16(b2_lo, b1_lo);
        let b3_hi = vaddq_s16(b2_hi, b1_hi);

        // Shift right by 6, clamp to 0..255
        let r_lo = vqshrun_n_s16::<6>(r2_lo);
        let r_hi = vqshrun_n_s16::<6>(r2_hi);
        let g_lo = vqshrun_n_s16::<6>(g4_lo);
        let g_hi = vqshrun_n_s16::<6>(g4_hi);
        let b_lo = vqshrun_n_s16::<6>(b3_lo);
        let b_hi = vqshrun_n_s16::<6>(b3_hi);

        let r = vcombine_u8(r_lo, r_hi);
        let g = vcombine_u8(g_lo, g_hi);
        let b = vcombine_u8(b_lo, b_hi);

        (r, g, b)
    }

    /// Single `#[arcane]` entry point for one interior row.
    /// Handles both RGB (bpp=3) and RGBA (bpp=4).
    #[arcane]
    pub(super) fn fill_2uv_row_arcane(
        _token: NeonToken,
        rgb: &mut [u8],
        y_row: &[u8],
        u_near: &[u8],
        u_far: &[u8],
        v_near: &[u8],
        v_far: &[u8],
        bpp: usize,
    ) {
        let width = y_row.len();
        let chroma_width = u_near.len();

        if width == 0 {
            return;
        }

        // First pixel (mirror left edge)
        {
            let u = get_fancy_chroma_value(u_near[0], u_near[0], u_far[0], u_far[0]);
            let v = get_fancy_chroma_value(v_near[0], v_near[0], v_far[0], v_far[0]);
            write_pixel(&mut rgb[..bpp], y_row[0], u, v);
        }

        let mut yx: usize = 1;
        let mut cx: usize = 0;
        let mut rgb_off: usize = bpp;

        // 32 Y pixels (16 chroma pairs) per SIMD iteration
        while yx + 32 <= width && cx + 17 <= chroma_width {
            let un: &[u8; 17] = u_near[cx..cx + 17].try_into().unwrap();
            let uf: &[u8; 17] = u_far[cx..cx + 17].try_into().unwrap();
            let vn: &[u8; 17] = v_near[cx..cx + 17].try_into().unwrap();
            let vf: &[u8; 17] = v_far[cx..cx + 17].try_into().unwrap();

            // Load chroma halves (8 samples each) and upsample to 16
            let u_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[0..8]).unwrap());
            let u_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[1..9]).unwrap());
            let u_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[0..8]).unwrap());
            let u_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[1..9]).unwrap());
            let u_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[8..16]).unwrap());
            let u_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[9..17]).unwrap());
            let u_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[8..16]).unwrap());
            let u_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[9..17]).unwrap());

            let v_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[0..8]).unwrap());
            let v_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[1..9]).unwrap());
            let v_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[0..8]).unwrap());
            let v_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[1..9]).unwrap());
            let v_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[8..16]).unwrap());
            let v_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[9..17]).unwrap());
            let v_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[8..16]).unwrap());
            let v_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[9..17]).unwrap());

            let u_up0 = upsample_16pixels_neon(_token, u_a0, u_b0, u_c0, u_d0);
            let u_up1 = upsample_16pixels_neon(_token, u_a1, u_b1, u_c1, u_d1);
            let v_up0 = upsample_16pixels_neon(_token, v_a0, v_b0, v_c0, v_d0);
            let v_up1 = upsample_16pixels_neon(_token, v_a1, v_b1, v_c1, v_d1);

            let ya: &[u8; 32] = y_row[yx..yx + 32].try_into().unwrap();
            let y0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&ya[0..16]).unwrap());
            let y1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&ya[16..32]).unwrap());

            if bpp == 3 {
                let out: &mut [u8; 96] = (&mut rgb[rgb_off..rgb_off + 96]).try_into().unwrap();
                let (rgb_0, rgb_1) = out.split_at_mut(48);
                convert_and_store_rgb16_neon(
                    _token,
                    y0,
                    u_up0,
                    v_up0,
                    <&mut [u8; 48]>::try_from(rgb_0).unwrap(),
                );
                convert_and_store_rgb16_neon(
                    _token,
                    y1,
                    u_up1,
                    v_up1,
                    <&mut [u8; 48]>::try_from(rgb_1).unwrap(),
                );
            } else {
                let out: &mut [u8; 128] = (&mut rgb[rgb_off..rgb_off + 128]).try_into().unwrap();
                let (rgba_0, rgba_1) = out.split_at_mut(64);
                convert_and_store_rgba16_neon(
                    _token,
                    y0,
                    u_up0,
                    v_up0,
                    <&mut [u8; 64]>::try_from(rgba_0).unwrap(),
                );
                convert_and_store_rgba16_neon(
                    _token,
                    y1,
                    u_up1,
                    v_up1,
                    <&mut [u8; 64]>::try_from(rgba_1).unwrap(),
                );
            }

            yx += 32;
            cx += 16;
            rgb_off += 32 * bpp;
        }

        // 16 Y pixels (8 chroma pairs) per iteration
        while yx + 16 <= width && cx + 9 <= chroma_width {
            let un: &[u8; 9] = u_near[cx..cx + 9].try_into().unwrap();
            let uf: &[u8; 9] = u_far[cx..cx + 9].try_into().unwrap();
            let vn: &[u8; 9] = v_near[cx..cx + 9].try_into().unwrap();
            let vf: &[u8; 9] = v_far[cx..cx + 9].try_into().unwrap();

            let u_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[0..8]).unwrap());
            let u_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&un[1..9]).unwrap());
            let u_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[0..8]).unwrap());
            let u_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&uf[1..9]).unwrap());
            let v_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[0..8]).unwrap());
            let v_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vn[1..9]).unwrap());
            let v_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[0..8]).unwrap());
            let v_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&vf[1..9]).unwrap());

            let u_up = upsample_16pixels_neon(_token, u_a, u_b, u_c, u_d);
            let v_up = upsample_16pixels_neon(_token, v_a, v_b, v_c, v_d);

            let ya: &[u8; 16] = y_row[yx..yx + 16].try_into().unwrap();
            let y_vec = simd_mem::vld1q_u8(ya);

            if bpp == 3 {
                convert_and_store_rgb16_neon(
                    _token,
                    y_vec,
                    u_up,
                    v_up,
                    <&mut [u8; 48]>::try_from(&mut rgb[rgb_off..rgb_off + 48]).unwrap(),
                );
            } else {
                convert_and_store_rgba16_neon(
                    _token,
                    y_vec,
                    u_up,
                    v_up,
                    <&mut [u8; 64]>::try_from(&mut rgb[rgb_off..rgb_off + 64]).unwrap(),
                );
            }

            yx += 16;
            cx += 8;
            rgb_off += 16 * bpp;
        }

        // Scalar remainder: pairs
        while yx + 1 < width && cx + 1 < chroma_width {
            {
                let u =
                    get_fancy_chroma_value(u_near[cx], u_near[cx + 1], u_far[cx], u_far[cx + 1]);
                let v =
                    get_fancy_chroma_value(v_near[cx], v_near[cx + 1], v_far[cx], v_far[cx + 1]);
                write_pixel(&mut rgb[rgb_off..rgb_off + bpp], y_row[yx], u, v);
            }
            {
                let u =
                    get_fancy_chroma_value(u_near[cx + 1], u_near[cx], u_far[cx + 1], u_far[cx]);
                let v =
                    get_fancy_chroma_value(v_near[cx + 1], v_near[cx], v_far[cx + 1], v_far[cx]);
                write_pixel(
                    &mut rgb[rgb_off + bpp..rgb_off + 2 * bpp],
                    y_row[yx + 1],
                    u,
                    v,
                );
            }
            yx += 2;
            cx += 1;
            rgb_off += 2 * bpp;
        }

        // Last pixel (mirror right edge)
        if yx < width {
            let lc = chroma_width - 1;
            let u = get_fancy_chroma_value(u_near[lc], u_near[lc], u_far[lc], u_far[lc]);
            let v = get_fancy_chroma_value(v_near[lc], v_near[lc], v_far[lc], v_far[lc]);
            write_pixel(&mut rgb[rgb_off..rgb_off + bpp], y_row[yx], u, v);
        }
    }
}

// ============================================================================
// wasm32 SIMD128 — delegates to scalar for now
// ============================================================================

#[cfg(target_arch = "wasm32")]
fn fill_2uv_row_wasm128(
    _token: archmage::Wasm128Token,
    rgb: &mut [u8],
    y_row: &[u8],
    u_near: &[u8],
    u_far: &[u8],
    v_near: &[u8],
    v_far: &[u8],
    bpp: usize,
) {
    fill_2uv_row_generic(rgb, y_row, u_near, u_far, v_near, v_far, bpp);
}

// ============================================================================
// Helpers
// ============================================================================

/// Write one pixel (RGB or RGBA) using the exact libwebp YUV→RGB formula.
#[inline(always)]
fn write_pixel(dst: &mut [u8], y: u8, u: u8, v: u8) {
    dst[0] = yuv_to_r(y, v);
    dst[1] = yuv_to_g(y, u, v);
    dst[2] = yuv_to_b(y, u);
    if dst.len() >= 4 {
        dst[3] = 255;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout};
    #[allow(unused_imports)]
    use alloc::{vec, vec::Vec};
    use archmage::incant;

    fn encode_lossy(rgb: &[u8], w: usize, h: usize, quality: f32) -> alloc::vec::Vec<u8> {
        let config = EncoderConfig::new_lossy()
            .with_quality(quality)
            .with_method(4);
        EncodeRequest::new(&config, rgb, PixelLayout::Rgb8, w as u32, h as u32)
            .encode()
            .expect("encode failed")
    }

    /// Encode 64x64 Q75, decode with yuv_exact, compare against webpx::decode_rgb().
    /// Asserts zero diffs (bit-exact with libwebp).
    #[test]
    fn bit_exact_vs_libwebp_64x64_q75() {
        let (w, h) = (64, 64);
        let mut pixels = alloc::vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                pixels[idx] = (x * 4) as u8;
                pixels[idx + 1] = (y * 4) as u8;
                pixels[idx + 2] = ((x + y) * 2) as u8;
            }
        }

        let webp = encode_lossy(&pixels, w, h, 75.0);

        // Decode with our yuv_exact kernel
        let config = DecodeConfig::default();
        let (zen_rgb, zen_w, zen_h) = DecodeRequest::new(&config, &webp)
            .decode_rgb_lossy()
            .expect("decode failed");

        // Decode with libwebp (via webpx)
        let (lib_rgb, lib_w, lib_h) = webpx::decode_rgb(&webp).expect("webpx decode failed");

        assert_eq!(zen_w as u32, lib_w, "width mismatch");
        assert_eq!(zen_h as u32, lib_h, "height mismatch");
        assert_eq!(zen_rgb.len(), lib_rgb.len(), "buffer size mismatch");

        let mut max_diff = 0u8;
        let mut diff_count = 0usize;
        for (i, (&a, &b)) in zen_rgb.iter().zip(lib_rgb.iter()).enumerate() {
            let d = a.abs_diff(b);
            if d > 0 {
                diff_count += 1;
                if d > max_diff {
                    max_diff = d;
                    let px = i / 3;
                    let ch = ["R", "G", "B"][i % 3];
                    std::eprintln!("diff at pixel {px} ({ch}): zen={a} libwebp={b} diff={d}");
                }
            }
        }

        assert_eq!(
            diff_count, 0,
            "yuv_exact vs libwebp: {diff_count} byte diffs, max_diff={max_diff}"
        );
    }

    /// Same test but for RGBA output.
    #[test]
    fn bit_exact_vs_libwebp_64x64_q75_rgba() {
        let (w, h) = (64, 64);
        let mut pixels = alloc::vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                pixels[idx] = (x * 4) as u8;
                pixels[idx + 1] = (y * 4) as u8;
                pixels[idx + 2] = ((x + y) * 2) as u8;
            }
        }

        let webp = encode_lossy(&pixels, w, h, 75.0);

        // Decode with RGBA
        let config = DecodeConfig::default();
        let (zen_rgba, zen_w, zen_h) = DecodeRequest::new(&config, &webp)
            .decode_rgba_lossy()
            .expect("decode failed");

        // Decode with libwebp RGBA
        let (lib_rgba, lib_w, lib_h) = webpx::decode_rgba(&webp).expect("webpx decode failed");

        assert_eq!(zen_w as u32, lib_w);
        assert_eq!(zen_h as u32, lib_h);
        assert_eq!(zen_rgba.len(), lib_rgba.len());

        let mut max_diff = 0u8;
        let mut diff_count = 0usize;
        for (&a, &b) in zen_rgba.iter().zip(lib_rgba.iter()) {
            let d = a.abs_diff(b);
            if d > 0 {
                diff_count += 1;
                max_diff = max_diff.max(d);
            }
        }

        assert_eq!(
            diff_count, 0,
            "yuv_exact RGBA vs libwebp: {diff_count} byte diffs, max_diff={max_diff}"
        );
    }

    /// Quick decode speed comparison: zenwebp (yuv_exact) vs libwebp.
    #[test]
    fn bench_vs_libwebp_512() {
        let (w, h) = (512, 512);
        let mut pixels = alloc::vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                pixels[idx] = ((x * 255) / w) as u8;
                pixels[idx + 1] = ((y * 255) / h) as u8;
                pixels[idx + 2] = 128;
            }
        }

        let webp = encode_lossy(&pixels, w, h, 75.0);

        let n = 50;
        let config = DecodeConfig::default();

        // Warm up
        for _ in 0..3 {
            let _ = DecodeRequest::new(&config, &webp)
                .decode_rgb_lossy()
                .unwrap();
            let _ = webpx::decode_rgb(&webp).unwrap();
        }

        let start = std::time::Instant::now();
        for _ in 0..n {
            let _ = DecodeRequest::new(&config, &webp)
                .decode_rgb_lossy()
                .unwrap();
        }
        let zen_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..n {
            let _ = webpx::decode_rgb(&webp).unwrap();
        }
        let lib_time = start.elapsed();

        let zen_us = zen_time.as_micros() as f64 / n as f64;
        let lib_us = lib_time.as_micros() as f64 / n as f64;

        std::eprintln!("512x512 Q75 RGB decode ({n} iters):");
        std::eprintln!("  zenwebp (yuv_exact): {zen_us:.0} us");
        std::eprintln!("  libwebp:             {lib_us:.0} us");
        std::eprintln!("  ratio:               {:.2}x", zen_us / lib_us);
    }

    /// Test odd dimensions to verify edge handling.
    #[test]
    fn bit_exact_vs_libwebp_odd_dimensions() {
        for &(w, h) in &[(1, 1), (3, 3), (17, 9), (63, 63), (65, 33)] {
            let mut pixels = alloc::vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    pixels[idx] = ((x * 255) / w.max(1)) as u8;
                    pixels[idx + 1] = ((y * 255) / h.max(1)) as u8;
                    pixels[idx + 2] = 128;
                }
            }

            let webp = encode_lossy(&pixels, w, h, 75.0);

            let config = DecodeConfig::default();
            let (zen_rgb, zen_w, zen_h) = DecodeRequest::new(&config, &webp)
                .decode_rgb_lossy()
                .expect("decode failed");

            let (lib_rgb, lib_w, lib_h) = webpx::decode_rgb(&webp).expect("webpx decode failed");

            assert_eq!(zen_w as u32, lib_w, "{w}x{h}: width mismatch");
            assert_eq!(zen_h as u32, lib_h, "{w}x{h}: height mismatch");

            let mut max_diff = 0u8;
            let mut diff_count = 0usize;
            for (&a, &b) in zen_rgb.iter().zip(lib_rgb.iter()) {
                let d = a.abs_diff(b);
                if d > 0 {
                    diff_count += 1;
                    max_diff = max_diff.max(d);
                }
            }

            assert_eq!(diff_count, 0, "{w}x{h}: {diff_count} diffs, max={max_diff}");
        }
    }

    /// Compare streaming conversion vs full-frame conversion to isolate
    /// streaming-specific bugs.
    #[test]
    fn streaming_matches_fullframe() {
        for &(w, h, q) in &[
            (64, 64, 75.0),
            (64, 64, 99.0),
            (64, 64, 100.0),
            (15, 15, 75.0),
            (3, 3, 75.0),
            (128, 128, 75.0),
            (256, 256, 75.0),
        ] {
            let mut pixels = alloc::vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    pixels[idx] = ((x * 255) / w.max(1)) as u8;
                    pixels[idx + 1] = ((y * 255) / h.max(1)) as u8;
                    pixels[idx + 2] = 128;
                }
            }

            let webp = encode_lossy(&pixels, w, h, q as f32);

            // Strip RIFF header to get raw VP8 data
            let chunk_size = u32::from_le_bytes([webp[16], webp[17], webp[18], webp[19]]) as usize;
            let vp8_data = &webp[20..20 + chunk_size.min(webp.len() - 20)];

            // Full-frame: decode_to_frame + yuv420_to_rgb_exact
            let mut ctx1 = super::super::DecoderContext::new();
            let frame = ctx1.decode_to_frame(vp8_data).expect("frame decode failed");
            let fw = usize::from(frame.width);
            let fh = usize::from(frame.height);
            let mbwidth = (fw + 15) / 16;
            let y_stride = mbwidth * 16;
            let uv_stride = mbwidth * 8;
            let mut fullframe_rgb = alloc::vec::Vec::new();
            super::yuv420_to_rgb_exact(
                &frame.ybuf,
                &frame.ubuf,
                &frame.vbuf,
                fw,
                fh,
                y_stride,
                uv_stride,
                &mut fullframe_rgb,
                3,
            );

            // Streaming: decode_to_rgb
            let mut ctx2 = super::super::DecoderContext::new();
            let mut streaming_rgb = alloc::vec::Vec::new();
            let (sw, sh) = ctx2
                .decode_to_rgb(vp8_data, &mut streaming_rgb, 3)
                .expect("streaming decode failed");

            assert_eq!(fw, usize::from(sw));
            assert_eq!(fh, usize::from(sh));
            assert_eq!(fullframe_rgb.len(), streaming_rgb.len());

            let mut max_diff = 0u8;
            let mut diff_count = 0usize;
            for (&a, &b) in fullframe_rgb.iter().zip(streaming_rgb.iter()) {
                let d = a.abs_diff(b);
                if d > 0 {
                    diff_count += 1;
                    max_diff = max_diff.max(d);
                }
            }

            assert_eq!(
                diff_count, 0,
                "{w}x{h} Q{q}: streaming differs from fullframe: {diff_count} diffs, max={max_diff}"
            );
        }
    }

    /// Verify all SIMD tiers produce identical output for fill_2uv_row.
    /// On x86: tests V3 vs scalar. On aarch64: tests NEON vs scalar.
    #[test]
    fn simd_tiers_produce_identical_output() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Generate test data: 128 Y pixels, 65 chroma samples
        let width = 128usize;
        let chroma_width = (width + 1) / 2;
        let y_row: Vec<u8> = (0..width).map(|i| (i * 2) as u8).collect();
        let u_near: Vec<u8> = (0..chroma_width).map(|i| (i * 3 + 10) as u8).collect();
        let u_far: Vec<u8> = (0..chroma_width).map(|i| (i * 3 + 20) as u8).collect();
        let v_near: Vec<u8> = (0..chroma_width).map(|i| (i * 3 + 30) as u8).collect();
        let v_far: Vec<u8> = (0..chroma_width).map(|i| (i * 3 + 40) as u8).collect();

        for bpp in [3, 4] {
            // Compute reference output with all tokens enabled (best SIMD)
            let mut reference = vec![0u8; width * bpp];
            super::fill_2uv_row_generic(
                &mut reference,
                &y_row,
                &u_near,
                &u_far,
                &v_near,
                &v_far,
                bpp,
            );

            let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                let mut output = vec![0u8; width * bpp];
                incant!(
                    super::fill_2uv_row(&mut output, &y_row, &u_near, &u_far, &v_near, &v_far, bpp,),
                    [v3, neon, wasm128, scalar]
                );

                assert_eq!(
                    output, reference,
                    "bpp={bpp}, tier '{perm}' differs from scalar reference"
                );
            });

            std::eprintln!("fill_2uv_row bpp={bpp}: {report}");
        }
    }

    /// Same permutation test but with odd widths to exercise scalar tails.
    #[test]
    fn simd_tiers_identical_odd_widths() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        for width in [1, 3, 15, 17, 31, 33, 63, 65] {
            let chroma_width = (width + 1) / 2;
            let y_row: Vec<u8> = (0..width).map(|i| (i * 7 + 50) as u8).collect();
            let u_near: Vec<u8> = (0..chroma_width).map(|i| (i * 5 + 100) as u8).collect();
            let u_far: Vec<u8> = (0..chroma_width).map(|i| (i * 5 + 110) as u8).collect();
            let v_near: Vec<u8> = (0..chroma_width).map(|i| (i * 5 + 120) as u8).collect();
            let v_far: Vec<u8> = (0..chroma_width).map(|i| (i * 5 + 130) as u8).collect();

            for bpp in [3, 4] {
                let mut reference = vec![0u8; width * bpp];
                super::fill_2uv_row_generic(
                    &mut reference,
                    &y_row,
                    &u_near,
                    &u_far,
                    &v_near,
                    &v_far,
                    bpp,
                );

                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let mut output = vec![0u8; width * bpp];
                    incant!(
                        super::fill_2uv_row(
                            &mut output,
                            &y_row,
                            &u_near,
                            &u_far,
                            &v_near,
                            &v_far,
                            bpp,
                        ),
                        [v3, neon, wasm128, scalar]
                    );

                    assert_eq!(
                        output, reference,
                        "w={width} bpp={bpp}, tier '{perm}' differs from scalar"
                    );
                });

                if width == 128 {
                    std::eprintln!("w={width} bpp={bpp}: {report}");
                }
            }
        }
    }
}
