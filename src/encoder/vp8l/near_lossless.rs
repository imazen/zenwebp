//! Near-lossless preprocessing for VP8L encoder.
//!
//! Two levels of near-lossless support (both from libwebp):
//!
//! 1. **Pixel-level preprocessing** (`apply_near_lossless`): Quantizes non-smooth
//!    raw pixels before any transforms. Only active when predictor is NOT used.
//!    Ported from `near_lossless_enc.c`.
//!
//! 2. **Residual-level quantization** (`near_lossless_residual`): Quantizes
//!    prediction residuals inside the predictor transform. Active for ALL images
//!    when near-lossless is enabled and predictor mode is selected.
//!    Ported from `predictor_enc.c`.

use alloc::vec;

use super::types::make_argb;

const MIN_DIM_FOR_NEAR_LOSSLESS: usize = 64;

/// Convert near-lossless quality (0-100) to max_quantization (power of 2).
///    100 -> 1 (no quantization)
///  80..99 -> 2
///  60..79 -> 4
///  40..59 -> 8
///  20..39 -> 16
///   0..19 -> 32
#[inline]
pub fn max_quantization_from_quality(quality: u8) -> u32 {
    1u32 << near_lossless_bits(quality)
}

/// Convert near-lossless quality (0-100) to limit_bits.
///    100 -> 0
///  80..99 -> 1
///  60..79 -> 2
///  40..59 -> 3
///  20..39 -> 4
///   0..19 -> 5
#[inline]
fn near_lossless_bits(quality: u8) -> u8 {
    5 - quality / 20
}

// ---------------------------------------------------------------------------
// Residual-level near-lossless (from predictor_enc.c)
// ---------------------------------------------------------------------------

/// Maximum per-channel difference between two pixels.
#[inline]
fn max_diff_between_pixels(p1: u32, p2: u32) -> i32 {
    let diff_a = ((p1 >> 24) as i32 - (p2 >> 24) as i32).abs();
    let diff_r = (((p1 >> 16) & 0xff) as i32 - ((p2 >> 16) & 0xff) as i32).abs();
    let diff_g = (((p1 >> 8) & 0xff) as i32 - ((p2 >> 8) & 0xff) as i32).abs();
    let diff_b = ((p1 & 0xff) as i32 - (p2 & 0xff) as i32).abs();
    diff_a.max(diff_r).max(diff_g.max(diff_b))
}

/// Maximum per-channel difference across 4-connected neighbors.
#[inline]
fn max_diff_around_pixel(current: u32, up: u32, down: u32, left: u32, right: u32) -> u8 {
    let d = max_diff_between_pixels(current, up)
        .max(max_diff_between_pixels(current, down))
        .max(max_diff_between_pixels(current, left).max(max_diff_between_pixels(current, right)));
    d.min(255) as u8
}

/// Undo subtract-green for a single pixel (for max_diffs computation).
#[inline]
fn add_green_to_blue_and_red(argb: u32) -> u32 {
    let green = (argb >> 8) & 0xff;
    let mut red_blue = argb & 0x00ff00ff;
    red_blue = red_blue.wrapping_add((green << 16) | green);
    red_blue &= 0x00ff00ff;
    (argb & 0xff00ff00) | red_blue
}

/// Compute max_diffs for a row of pixels.
///
/// For each interior pixel x in [1, width-2], computes the maximum per-channel
/// difference to its 4-connected neighbors. If subtract-green was applied,
/// undoes it first for accurate comparison.
///
/// `pixels` is the full pixel array, `y` is the current row index.
/// Only call for rows 1..height-2 (needs valid rows above and below).
pub fn max_diffs_for_row(
    pixels: &[u32],
    width: usize,
    y: usize,
    max_diffs: &mut [u8],
    used_subtract_green: bool,
) {
    if width <= 2 {
        return;
    }
    let row = y * width;
    let above = (y - 1) * width;
    let below = (y + 1) * width;

    let transform = |p: u32| -> u32 {
        if used_subtract_green {
            add_green_to_blue_and_red(p)
        } else {
            p
        }
    };

    let mut current = transform(pixels[row]);
    let mut right = transform(pixels[row + 1]);

    // max_diffs[0] and max_diffs[width-1] are never used (border pixels)
    for x in 1..width - 1 {
        let up = transform(pixels[above + x]);
        let down = transform(pixels[below + x]);
        let left = current;
        current = right;
        right = transform(pixels[row + x + 1]);
        max_diffs[x] = max_diff_around_pixel(current, up, down, left, right);
    }
}

/// Quantize a single component's residual to a multiple of quantization,
/// working modulo 256, taking care not to cross a boundary.
///
/// Ported from libwebp's `NearLosslessComponent` in predictor_enc.c.
#[inline]
fn near_lossless_component(value: u8, predict: u8, boundary: u8, quantization: i32) -> u8 {
    let residual = (value as i32 - predict as i32) & 0xff;
    let boundary_residual = (boundary as i32 - predict as i32) & 0xff;
    let lower = residual & !(quantization - 1);
    let upper = lower + quantization;
    // Resolve ties towards a value closer to prediction (lower if value > predict,
    // upper otherwise).
    let bias = (((boundary as i32 - value as i32) & 0xff) < boundary_residual) as i32;
    if residual - lower < upper - residual + bias {
        // lower is closer
        if residual > boundary_residual && lower <= boundary_residual {
            // Halve step to avoid crossing boundary
            (lower + (quantization >> 1)) as u8
        } else {
            lower as u8
        }
    } else {
        // upper is closer
        if residual <= boundary_residual && upper > boundary_residual {
            // Halve step to avoid crossing boundary
            (lower + (quantization >> 1)) as u8
        } else {
            (upper & 0xff) as u8
        }
    }
}

/// Compute quantized near-lossless residual for a pixel.
///
/// Returns the quantized residual (pixel - prediction, quantized to multiples
/// of a power-of-2 quantization level that depends on local smoothness).
///
/// Also returns the reconstructed pixel value (predict + residual) so the
/// caller can update the source image for subsequent predictions.
///
/// Ported from libwebp's `NearLossless()` in predictor_enc.c.
#[inline]
pub fn near_lossless_residual(
    value: u32,
    predict: u32,
    max_quantization: u32,
    max_diff: u8,
    used_subtract_green: bool,
) -> (u32, u32) {
    // If pixel is very smooth, no quantization
    if max_diff as i32 <= 2 {
        let res = sub_pixels(value, predict);
        return (res, value);
    }

    // Reduce quantization until it's smaller than max_diff
    let mut quantization = max_quantization as i32;
    while quantization >= max_diff as i32 {
        quantization >>= 1;
    }

    // Alpha: preserve fully transparent/opaque
    let a = if (value >> 24) == 0 || (value >> 24) == 0xff {
        ((value >> 24) as u8).wrapping_sub((predict >> 24) as u8)
    } else {
        near_lossless_component(
            (value >> 24) as u8,
            (predict >> 24) as u8,
            0xff,
            quantization,
        )
    };

    // Green
    let g = near_lossless_component(
        ((value >> 8) & 0xff) as u8,
        ((predict >> 8) & 0xff) as u8,
        0xff,
        quantization,
    );

    // If subtract-green was applied, compensate R and B for green quantization error
    let (new_green, green_diff) = if used_subtract_green {
        let new_g = ((predict >> 8) as u8).wrapping_add(g);
        let g_diff = new_g.wrapping_sub(((value >> 8) & 0xff) as u8);
        (new_g, g_diff)
    } else {
        (0, 0)
    };

    // Red: adjust for green compensation if subtract-green was used
    let r_value = ((value >> 16) & 0xff) as u8;
    let r = near_lossless_component(
        r_value.wrapping_sub(green_diff),
        ((predict >> 16) & 0xff) as u8,
        0xffu8.wrapping_sub(new_green),
        quantization,
    );

    // Blue: adjust for green compensation if subtract-green was used
    let b_value = (value & 0xff) as u8;
    let b = near_lossless_component(
        b_value.wrapping_sub(green_diff),
        (predict & 0xff) as u8,
        0xffu8.wrapping_sub(new_green),
        quantization,
    );

    let residual = make_argb(a, r, g, b);
    let reconstructed = add_pixels(predict, residual);
    (residual, reconstructed)
}

/// Subtract pixels: (a - b) per channel, wrapping.
#[inline]
fn sub_pixels(a: u32, b: u32) -> u32 {
    let pa = ((a >> 24) as u8).wrapping_sub((b >> 24) as u8);
    let pr = (((a >> 16) & 0xff) as u8).wrapping_sub(((b >> 16) & 0xff) as u8);
    let pg = (((a >> 8) & 0xff) as u8).wrapping_sub(((b >> 8) & 0xff) as u8);
    let pb = ((a & 0xff) as u8).wrapping_sub((b & 0xff) as u8);
    make_argb(pa, pr, pg, pb)
}

/// Add pixels: (a + b) per channel, wrapping.
#[inline]
fn add_pixels(a: u32, b: u32) -> u32 {
    let pa = ((a >> 24) as u8).wrapping_add((b >> 24) as u8);
    let pr = (((a >> 16) & 0xff) as u8).wrapping_add(((b >> 16) & 0xff) as u8);
    let pg = (((a >> 8) & 0xff) as u8).wrapping_add(((b >> 8) & 0xff) as u8);
    let pb = ((a & 0xff) as u8).wrapping_add((b & 0xff) as u8);
    make_argb(pa, pr, pg, pb)
}

/// Quantize a value to nearest multiple of `1 << bits` using banker's rounding
/// (round-half-to-even). Clamps to 255.
#[inline]
fn find_closest_discretized(a: u8, bits: u8) -> u8 {
    let a = a as u32;
    let mask = (1u32 << bits) - 1;
    let biased = a + (mask >> 1) + ((a >> bits) & 1);
    if biased > 0xff {
        0xff
    } else {
        (biased & !mask) as u8
    }
}

/// Apply `find_closest_discretized` to all four ARGB channels of a pixel.
#[inline]
fn closest_discretized_argb(argb: u32, bits: u8) -> u32 {
    (find_closest_discretized((argb >> 24) as u8, bits) as u32) << 24
        | (find_closest_discretized((argb >> 16) as u8, bits) as u32) << 16
        | (find_closest_discretized((argb >> 8) as u8, bits) as u32) << 8
        | find_closest_discretized(argb as u8, bits) as u32
}

/// Check if all four channels of pixels `a` and `b` are within `limit`.
#[inline]
fn is_near(a: u32, b: u32, limit: i32) -> bool {
    for k in 0..4 {
        let delta = ((a >> (k * 8)) & 0xff) as i32 - ((b >> (k * 8)) & 0xff) as i32;
        if delta >= limit || delta <= -limit {
            return false;
        }
    }
    true
}

/// Check if a pixel is smooth (all 4-connected neighbors within limit).
#[inline]
fn is_smooth(prev_row: &[u32], curr_row: &[u32], next_row: &[u32], x: usize, limit: i32) -> bool {
    is_near(curr_row[x], curr_row[x - 1], limit)
        && is_near(curr_row[x], curr_row[x + 1], limit)
        && is_near(curr_row[x], prev_row[x], limit)
        && is_near(curr_row[x], next_row[x], limit)
}

/// Single pass of near-lossless processing.
///
/// Reads from `src`, writes to `dst`. Border pixels (first/last row, first/last
/// column) are copied unchanged. Interior non-smooth pixels are quantized.
fn near_lossless_pass(src: &[u32], w: usize, h: usize, bits: u8, dst: &mut [u32]) {
    let limit = 1i32 << bits;

    // Working rows: prev, curr, next (matching libwebp's copy_buffer approach)
    let mut prev_row = vec![0u32; w];
    let mut curr_row = vec![0u32; w];
    let mut next_row = vec![0u32; w];

    curr_row.copy_from_slice(&src[..w]);
    if h > 1 {
        next_row.copy_from_slice(&src[w..2 * w]);
    }

    for y in 0..h {
        if y == 0 || y == h - 1 {
            // Border rows: copy unchanged
            dst[y * w..(y + 1) * w].copy_from_slice(&src[y * w..(y + 1) * w]);
        } else {
            // Load next row
            if y + 1 < h {
                next_row.copy_from_slice(&src[(y + 1) * w..(y + 2) * w]);
            }
            // Border columns: copy unchanged
            dst[y * w] = src[y * w];
            dst[y * w + w - 1] = src[y * w + w - 1];
            // Interior pixels
            for x in 1..w - 1 {
                if is_smooth(&prev_row, &curr_row, &next_row, x, limit) {
                    dst[y * w + x] = curr_row[x];
                } else {
                    dst[y * w + x] = closest_discretized_argb(curr_row[x], bits);
                }
            }
        }
        // Three-way rotate: prev <- curr <- next <- prev
        let temp = core::mem::replace(&mut prev_row, curr_row);
        curr_row = core::mem::replace(&mut next_row, temp);
    }
}

/// Apply near-lossless preprocessing to an ARGB pixel array.
///
/// Multi-pass: iterates from `limit_bits` down to 1, each pass refining the
/// previous result. Border pixels are never modified.
///
/// Skips processing if:
/// - quality >= 100 (exact lossless)
/// - both dimensions < 64 (small icon)
/// - height < 3 (too few rows for 4-connected neighborhood)
pub fn apply_near_lossless(argb: &mut [u32], w: usize, h: usize, quality: u8) {
    if quality >= 100 {
        return;
    }
    let limit_bits = near_lossless_bits(quality);
    if limit_bits == 0 {
        return;
    }
    // Skip small images (matching libwebp)
    if (w < MIN_DIM_FOR_NEAR_LOSSLESS && h < MIN_DIM_FOR_NEAR_LOSSLESS) || h < 3 {
        return;
    }

    // First pass: full limit_bits
    let mut copy_buffer = argb.to_vec();
    near_lossless_pass(&copy_buffer, w, h, limit_bits, argb);

    // Refinement passes: limit_bits-1 down to 1
    for bits in (1..limit_bits).rev() {
        copy_buffer.copy_from_slice(argb);
        near_lossless_pass(&copy_buffer, w, h, bits, argb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_max_quantization_from_quality() {
        assert_eq!(max_quantization_from_quality(100), 1);
        assert_eq!(max_quantization_from_quality(99), 2);
        assert_eq!(max_quantization_from_quality(80), 2);
        assert_eq!(max_quantization_from_quality(60), 4);
        assert_eq!(max_quantization_from_quality(40), 8);
        assert_eq!(max_quantization_from_quality(20), 16);
        assert_eq!(max_quantization_from_quality(0), 32);
    }

    #[test]
    fn test_max_diff_between_pixels() {
        let a = 0xFF_80_40_20u32;
        let b = 0xFF_90_40_20u32;
        assert_eq!(max_diff_between_pixels(a, b), 16); // R channel differs by 16
    }

    #[test]
    fn test_near_lossless_component_basic() {
        // No quantization boundary crossing: simple rounding
        // value=130, predict=128, residual=2, quantization=4
        // lower=0, upper=4, residual-lower=2 == upper-residual=2, bias resolves
        let r = near_lossless_component(130, 128, 0xff, 4);
        // residual=2, lower=0, upper=4
        // bias = ((255-130)&0xff < (255-128)&0xff) = (125 < 127) = 1
        // 2-0 < 4-2+1 => 2 < 3, so lower wins => 0
        assert_eq!(r, 0);
    }

    #[test]
    fn test_near_lossless_residual_smooth() {
        // Very smooth pixel (max_diff <= 2): no quantization
        let value = 0xFF_80_40_20u32;
        let predict = 0xFF_80_40_1Eu32;
        let (res, _recon) = near_lossless_residual(value, predict, 4, 2, false);
        // Should be exact subtraction
        assert_eq!(res, sub_pixels(value, predict));
    }

    #[test]
    fn test_near_lossless_residual_reconstruct() {
        // Verify that predict + residual = reconstructed
        let value = 0xFF_80_60_40u32;
        let predict = 0xFF_70_50_30u32;
        let (res, recon) = near_lossless_residual(value, predict, 8, 20, false);
        assert_eq!(add_pixels(predict, res), recon);
    }

    #[test]
    fn test_add_green_to_blue_and_red() {
        // R=0x10, G=0x20, B=0x30, A=0xFF
        let argb = 0xFF_10_20_30u32;
        let result = add_green_to_blue_and_red(argb);
        // R = 0x10 + 0x20 = 0x30, B = 0x30 + 0x20 = 0x50
        assert_eq!(result, 0xFF_30_20_50u32);
    }

    #[test]
    fn test_near_lossless_bits() {
        assert_eq!(near_lossless_bits(100), 0);
        assert_eq!(near_lossless_bits(99), 1);
        assert_eq!(near_lossless_bits(80), 1);
        assert_eq!(near_lossless_bits(79), 2);
        assert_eq!(near_lossless_bits(60), 2);
        assert_eq!(near_lossless_bits(59), 3);
        assert_eq!(near_lossless_bits(40), 3);
        assert_eq!(near_lossless_bits(39), 4);
        assert_eq!(near_lossless_bits(20), 4);
        assert_eq!(near_lossless_bits(19), 5);
        assert_eq!(near_lossless_bits(0), 5);
    }

    #[test]
    fn test_find_closest_discretized() {
        // bits=1: mask=0, mask>>1=0, round to multiples of 2
        assert_eq!(find_closest_discretized(0, 1), 0); // 0+0+0=0
        assert_eq!(find_closest_discretized(1, 1), 0); // 1+0+0=1, &~1=0 (banker's: round to even)
        assert_eq!(find_closest_discretized(2, 1), 2); // 2+0+1=3, &~1=2
        assert_eq!(find_closest_discretized(3, 1), 4); // 3+0+1=4, &~1=4
        assert_eq!(find_closest_discretized(4, 1), 4); // 4+0+0=4, &~1=4
        assert_eq!(find_closest_discretized(254, 1), 254);
        assert_eq!(find_closest_discretized(255, 1), 255); // 255+0+1=256>255, clamp

        // bits=2: mask=3, mask>>1=1, round to multiples of 4
        assert_eq!(find_closest_discretized(0, 2), 0); // 0+1+0=1, &~3=0
        assert_eq!(find_closest_discretized(1, 2), 0); // 1+1+0=2, &~3=0
        assert_eq!(find_closest_discretized(2, 2), 0); // 2+1+0=3, &~3=0
        assert_eq!(find_closest_discretized(3, 2), 4); // 3+1+0=4, &~3=4
        assert_eq!(find_closest_discretized(4, 2), 4); // 4+1+1=6, &~3=4
        assert_eq!(find_closest_discretized(253, 2), 252);
        assert_eq!(find_closest_discretized(255, 2), 255); // clamped
    }

    #[test]
    fn test_find_closest_discretized_matches_libwebp() {
        // Verify the formula matches libwebp's FindClosestDiscretized exactly
        for bits in 1u8..=5 {
            let mask = (1u32 << bits) - 1;
            for a in 0u8..=255 {
                let a32 = a as u32;
                let biased = a32 + (mask >> 1) + ((a32 >> bits) & 1);
                let expected = if biased > 0xff {
                    0xff
                } else {
                    (biased & !mask) as u8
                };
                assert_eq!(
                    find_closest_discretized(a, bits),
                    expected,
                    "mismatch at a={a}, bits={bits}"
                );
            }
        }
    }

    #[test]
    fn test_is_near() {
        let a = 0xFF_80_40_20u32; // A=255, R=128, G=64, B=32
        let b = 0xFF_82_3E_21u32; // A=255, R=130, G=62, B=33

        // limit=4: all deltas (0, 2, 2, 1) are within [-4, 4)
        assert!(is_near(a, b, 4));
        // limit=2: R delta=2 is NOT < 2
        assert!(!is_near(a, b, 2));
    }

    #[test]
    fn test_quality_100_is_noop() {
        let mut argb = vec![0xFF_00_00_00u32; 100 * 100];
        let original = argb.clone();
        apply_near_lossless(&mut argb, 100, 100, 100);
        assert_eq!(argb, original);
    }

    #[test]
    fn test_small_image_skipped() {
        // Both dims < 64: skip
        let mut argb = vec![0xFF_80_80_80u32; 32 * 32];
        let original = argb.clone();
        apply_near_lossless(&mut argb, 32, 32, 50);
        assert_eq!(argb, original);
    }

    #[test]
    fn test_short_image_skipped() {
        // height < 3: skip
        let mut argb = vec![0xFF_80_80_80u32; 100 * 2];
        let original = argb.clone();
        apply_near_lossless(&mut argb, 100, 2, 50);
        assert_eq!(argb, original);
    }

    #[test]
    fn test_near_lossless_modifies_pixels() {
        // Create a 64x64 image with varying pixels (not smooth)
        let w = 64;
        let h = 64;
        let mut argb = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // Create non-smooth pattern (high frequency noise)
                let r = ((x * 37 + y * 13) % 256) as u8;
                let g = ((x * 53 + y * 7) % 256) as u8;
                let b = ((x * 11 + y * 29) % 256) as u8;
                argb.push(0xFF000000 | (r as u32) << 16 | (g as u32) << 8 | b as u32);
            }
        }
        let original = argb.clone();
        apply_near_lossless(&mut argb, w, h, 60); // bits=2

        // Borders should be unchanged
        for x in 0..w {
            assert_eq!(argb[x], original[x], "top row modified at x={x}");
            assert_eq!(
                argb[(h - 1) * w + x],
                original[(h - 1) * w + x],
                "bottom row modified at x={x}"
            );
        }
        for y in 0..h {
            assert_eq!(argb[y * w], original[y * w], "left col modified at y={y}");
            assert_eq!(
                argb[y * w + w - 1],
                original[y * w + w - 1],
                "right col modified at y={y}"
            );
        }

        // Interior should have some modified pixels
        let mut changed = 0;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if argb[y * w + x] != original[y * w + x] {
                    changed += 1;
                }
            }
        }
        assert!(changed > 0, "no interior pixels were modified");
    }

    #[test]
    fn test_near_lossless_bounded_error() {
        // Verify that each channel differs by at most (1 << limit_bits) - 1
        let w = 64;
        let h = 64;
        let mut argb = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 37 + y * 13) % 256) as u8;
                let g = ((x * 53 + y * 7) % 256) as u8;
                let b = ((x * 11 + y * 29) % 256) as u8;
                argb.push(0xFF000000 | (r as u32) << 16 | (g as u32) << 8 | b as u32);
            }
        }
        let original = argb.clone();

        for quality in [0u8, 20, 40, 60, 80, 99] {
            let mut test = original.clone();
            apply_near_lossless(&mut test, w, h, quality);

            let limit_bits = near_lossless_bits(quality);
            // After all passes (limit_bits down to 1), max error per pass is
            // bounded by (1 << bits). The final pass uses bits=1, so the result
            // should be quantized to multiples of 2.
            for i in 0..w * h {
                for ch in 0..4 {
                    let orig_val = ((original[i] >> (ch * 8)) & 0xff) as i32;
                    let new_val = ((test[i] >> (ch * 8)) & 0xff) as i32;
                    let max_error = (1i32 << limit_bits) - 1;
                    assert!(
                        (orig_val - new_val).abs() <= max_error,
                        "quality={quality}, pixel={i}, channel={ch}: orig={orig_val}, new={new_val}, max_error={max_error}"
                    );
                }
            }
        }
    }
}
