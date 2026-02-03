//! Near-lossless preprocessing for VP8L encoder.
//!
//! Quantizes non-smooth pixels to reduce color precision slightly, improving
//! compression with bounded quality loss. Quality 100 = exact lossless (disabled),
//! quality 0 = most aggressive quantization.
//!
//! Ported from libwebp's `near_lossless_enc.c`.

use alloc::vec;

const MIN_DIM_FOR_NEAR_LOSSLESS: usize = 64;

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
