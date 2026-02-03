//! Main VP8L encoder implementation.

use alloc::vec::Vec;

use super::backward_refs::get_backward_references;
use super::bitwriter::BitWriter;
use super::color_cache::ColorCache;
use super::histogram::{
    distance_code_extra_bits, distance_code_to_prefix, length_code_extra_bits, length_to_code,
    Histogram,
};
use super::huffman::{
    build_huffman_codes, build_huffman_lengths, write_huffman_tree, write_single_entry_tree,
};
use super::transforms::{apply_predictor_transform, apply_subtract_green, ColorIndexTransform};
use super::types::{
    argb_alpha, argb_blue, argb_green, argb_red, make_argb, PixOrCopy, Vp8lConfig,
    NUM_LENGTH_CODES, NUM_LITERAL_CODES,
};

use crate::encoder::api::EncodingError;

/// Encode an image using VP8L lossless compression.
pub fn encode_vp8l(
    pixels: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    config: &Vp8lConfig,
) -> Result<Vec<u8>, EncodingError> {
    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodingError::InvalidDimensions);
    }

    let w = width as usize;
    let h = height as usize;
    let expected_len = w * h * if has_alpha { 4 } else { 3 };
    if pixels.len() != expected_len {
        return Err(EncodingError::InvalidBufferSize(alloc::format!(
            "expected {} bytes, got {}",
            expected_len,
            pixels.len()
        )));
    }

    // Convert to ARGB u32 array
    let mut argb: Vec<u32> = if has_alpha {
        pixels
            .chunks_exact(4)
            .map(|p| make_argb(p[3], p[0], p[1], p[2]))
            .collect()
    } else {
        pixels
            .chunks_exact(3)
            .map(|p| make_argb(255, p[0], p[1], p[2]))
            .collect()
    };

    // Encode with the full pipeline
    encode_argb(&mut argb, w, h, has_alpha, config)
}

/// Encode ARGB pixels (internal).
fn encode_argb(
    argb: &mut [u32],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &Vp8lConfig,
) -> Result<Vec<u8>, EncodingError> {
    let mut writer = BitWriter::with_capacity(width * height / 2);

    // Write VP8L signature
    writer.write_bits(0x2f, 8);

    // Write dimensions
    writer.write_bits((width - 1) as u64, 14);
    writer.write_bits((height - 1) as u64, 14);

    // Alpha hint and version
    writer.write_bit(has_alpha);
    writer.write_bits(0, 3); // version 0

    // Determine which transforms to use
    let use_palette = config.use_palette && can_use_palette(argb);
    let use_predictor = config.use_predictor && !use_palette;
    let use_subtract_green = config.use_subtract_green && !use_palette;

    // Try palette encoding if beneficial
    let palette_transform = if use_palette {
        ColorIndexTransform::try_build(argb)
    } else {
        None
    };

    // Apply transforms and signal them in bitstream.
    // Order: Predictor first, then SubtractGreen (matching libwebp).
    // The decoder reverses this order.

    // Predictor transform (applied first, on original image)
    if use_predictor && palette_transform.is_none() {
        // Choose best predictors per block and compute residuals
        let predictor_data = apply_predictor_transform(argb, width, height, config.predictor_bits);

        // Signal predictor transform
        writer.write_bit(true); // transform present
        writer.write_bits(0, 2); // predictor = 0
        writer.write_bits((config.predictor_bits - 2) as u64, 3);

        // Write predictor sub-image
        write_predictor_image(&mut writer, &predictor_data);
    }

    // Subtract green transform (applied second, on residuals)
    if use_subtract_green && palette_transform.is_none() {
        writer.write_bit(true); // transform present
        writer.write_bits(2, 2); // subtract green = 2
        apply_subtract_green(argb);
    }

    // Color indexing transform
    if let Some(ref palette) = palette_transform {
        writer.write_bit(true); // transform present
        writer.write_bits(3, 2); // color indexing = 3
        writer.write_bits((palette.palette.len() - 1) as u64, 8);

        // Write palette
        write_palette(&mut writer, &palette.palette);

        // Apply transform
        palette.apply(argb);

        // Update width for packed pixels
        let bits = palette.bits_per_pixel();
        if bits > 0 {
            // Pack pixels (implementation simplified for now)
        }
    }

    // No more transforms
    writer.write_bit(false);

    // Build backward references (determines optimal cache_bits via entropy estimation)
    let cache_bits_max = if config.cache_bits == 0 {
        10 // Auto-detect: try all sizes up to MAX_COLOR_CACHE_BITS
    } else {
        config.cache_bits.min(10)
    };
    let (refs, cache_bits) = get_backward_references(
        argb, width, height, config.quality.quality, cache_bits_max,
    );

    // Write color cache info
    if cache_bits > 0 {
        writer.write_bit(true);
        writer.write_bits(cache_bits as u64, 4);
    } else {
        writer.write_bit(false);
    }

    // Meta-Huffman (disabled for now - single Huffman group)
    writer.write_bit(false);

    // Build histogram
    let histogram = Histogram::from_refs(&refs, cache_bits);

    // Build Huffman codes
    let literal_lengths = build_huffman_lengths(&histogram.literal, 15);
    let literal_codes = build_huffman_codes(&literal_lengths);

    let red_lengths = build_huffman_lengths(&histogram.red, 15);
    let red_codes = build_huffman_codes(&red_lengths);

    let blue_lengths = build_huffman_lengths(&histogram.blue, 15);
    let blue_codes = build_huffman_codes(&blue_lengths);

    let alpha_lengths = build_huffman_lengths(&histogram.alpha, 15);
    let alpha_codes = build_huffman_codes(&alpha_lengths);

    let dist_lengths = build_huffman_lengths(&histogram.distance, 15);
    let dist_codes = build_huffman_codes(&dist_lengths);

    // Write Huffman trees
    write_huffman_tree(&mut writer, &literal_lengths);
    write_huffman_tree(&mut writer, &red_lengths);
    write_huffman_tree(&mut writer, &blue_lengths);
    write_huffman_tree(&mut writer, &alpha_lengths);
    write_huffman_tree(&mut writer, &dist_lengths);

    // Write image data
    let mut color_cache = if cache_bits > 0 {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    // Check for trivial trees (single symbol = no bits needed)
    let literal_trivial = literal_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let red_trivial = red_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let blue_trivial = blue_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let alpha_trivial = alpha_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let dist_trivial = dist_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let mut argb_idx = 0usize;

    for token in refs.iter() {
        match *token {
            PixOrCopy::Literal(argb_val) => {
                // Write green + length code (skip if trivial)
                if !literal_trivial {
                    let g = argb_green(argb_val) as usize;
                    writer.write_bits(literal_codes[g].code as u64, literal_codes[g].length);
                }

                // Write red (skip if trivial)
                if !red_trivial {
                    let r = argb_red(argb_val) as usize;
                    writer.write_bits(red_codes[r].code as u64, red_codes[r].length);
                }

                // Write blue (skip if trivial)
                if !blue_trivial {
                    let b = argb_blue(argb_val) as usize;
                    writer.write_bits(blue_codes[b].code as u64, blue_codes[b].length);
                }

                // Write alpha (skip if trivial)
                if !alpha_trivial {
                    let a = argb_alpha(argb_val) as usize;
                    writer.write_bits(alpha_codes[a].code as u64, alpha_codes[a].length);
                }

                // Update cache
                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb_val);
                }
                argb_idx += 1;
            }
            PixOrCopy::CacheIdx(idx) => {
                // Cache code = 256 + 24 + idx (in literal tree, so check literal_trivial)
                if !literal_trivial {
                    let code = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
                    writer.write_bits(literal_codes[code].code as u64, literal_codes[code].length);
                }

                // Update cache with actual pixel to maintain state
                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb[argb_idx]);
                }
                argb_idx += 1;
            }
            PixOrCopy::Copy { len, dist } => {
                // Write length prefix code (in literal tree, so check literal_trivial)
                let (len_prefix, len_extra) = length_to_code(len);
                let len_code = NUM_LITERAL_CODES + len_prefix as usize;
                if !literal_trivial {
                    writer.write_bits(
                        literal_codes[len_code].code as u64,
                        literal_codes[len_code].length,
                    );
                }

                // Write length extra bits (always written, not from Huffman tree)
                let len_extra_bits = length_code_extra_bits(len_prefix);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra as u64, len_extra_bits);
                }

                // Write distance prefix code (in distance tree, check dist_trivial)
                let (dist_prefix, dist_extra) = distance_code_to_prefix(dist);
                if !dist_trivial {
                    writer.write_bits(
                        dist_codes[dist_prefix as usize].code as u64,
                        dist_codes[dist_prefix as usize].length,
                    );
                }

                // Write distance extra bits (always written, not from Huffman tree)
                let dist_extra_bits = distance_code_extra_bits(dist_prefix);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra as u64, dist_extra_bits);
                }

                // Update cache with copied pixels
                if let Some(ref mut cache) = color_cache {
                    for k in 0..len as usize {
                        cache.insert(argb[argb_idx + k]);
                    }
                }
                argb_idx += len as usize;
            }
        }
    }

    Ok(writer.finish())
}

/// Check if palette encoding is beneficial.
fn can_use_palette(argb: &[u32]) -> bool {
    if argb.len() < 100 {
        return false; // Too small
    }

    // Quick check: sample pixels to estimate unique color count
    let sample_size = (argb.len() / 16).clamp(64, 1024);
    let step = argb.len() / sample_size;

    let mut seen = alloc::collections::BTreeSet::new();
    for i in (0..argb.len()).step_by(step) {
        seen.insert(argb[i]);
        if seen.len() > 256 {
            return false;
        }
    }

    // If sample has few colors, likely palette-friendly
    seen.len() <= 64
}

/// Write predictor sub-image (variable modes per block).
///
/// The sub-image encodes one pixel per block with the predictor mode in the
/// green channel (0-13). Other channels are zero. Uses Huffman coding for
/// the green channel and trivial single-entry trees for other channels.
fn write_predictor_image(writer: &mut BitWriter, predictor_data: &[u32]) {
    // No color cache for sub-images
    writer.write_bit(false);

    // NOTE: No meta-Huffman flag for sub-images (only for main image)

    // Build histogram for the green channel (predictor modes 0-13)
    let mut hist_green = [0u32; 280]; // 256 literals + 24 length codes
    for &pixel in predictor_data {
        hist_green[argb_green(pixel) as usize] += 1;
    }

    // Check if all blocks use the same mode (trivial case)
    let unique_count = hist_green.iter().filter(|&&c| c > 0).count();

    if unique_count <= 1 {
        // Trivial: single mode for all blocks
        let val = hist_green.iter().position(|&c| c > 0).unwrap_or(2); // default Top
        write_single_entry_tree(writer, val); // green
        write_single_entry_tree(writer, 0); // red
        write_single_entry_tree(writer, 0); // blue
        write_single_entry_tree(writer, 0); // alpha
        write_single_entry_tree(writer, 0); // distance
                                            // No image data needed with all-trivial trees
        return;
    }

    // Non-trivial: build Huffman tree for green channel
    let green_lengths = build_huffman_lengths(&hist_green, 15);
    let green_codes = build_huffman_codes(&green_lengths);

    // Write 5 Huffman trees
    write_huffman_tree(writer, &green_lengths); // green (variable modes)
    write_single_entry_tree(writer, 0); // red (all 0)
    write_single_entry_tree(writer, 0); // blue (all 0)
    write_single_entry_tree(writer, 0); // alpha (all 0)
    write_single_entry_tree(writer, 0); // distance (unused)

    // Write image data: one green value per block
    // Red/blue/alpha are trivial (no bits), only green emits bits
    for &pixel in predictor_data {
        let g = argb_green(pixel) as usize;
        writer.write_bits(green_codes[g].code as u64, green_codes[g].length);
    }
}

/// Write palette data.
fn write_palette(writer: &mut BitWriter, palette: &[u32]) {
    // Palette is written as a 1-row image
    // No color cache
    writer.write_bit(false);

    // No meta-Huffman
    writer.write_bit(false);

    // Build histogram from palette
    let mut hist_g = [0u32; 280];
    let mut hist_r = [0u32; 256];
    let mut hist_b = [0u32; 256];
    let mut hist_a = [0u32; 256];

    // Palette is differentially coded
    let mut prev = 0u32;
    for &color in palette {
        let diff = sub_pixels(color, prev);
        hist_g[argb_green(diff) as usize] += 1;
        hist_r[argb_red(diff) as usize] += 1;
        hist_b[argb_blue(diff) as usize] += 1;
        hist_a[argb_alpha(diff) as usize] += 1;
        prev = color;
    }

    // Build and write Huffman trees
    let g_len = build_huffman_lengths(&hist_g, 15);
    let g_codes = build_huffman_codes(&g_len);
    let r_len = build_huffman_lengths(&hist_r, 15);
    let r_codes = build_huffman_codes(&r_len);
    let b_len = build_huffman_lengths(&hist_b, 15);
    let b_codes = build_huffman_codes(&b_len);
    let a_len = build_huffman_lengths(&hist_a, 15);
    let a_codes = build_huffman_codes(&a_len);

    write_huffman_tree(writer, &g_len);
    write_huffman_tree(writer, &r_len);
    write_huffman_tree(writer, &b_len);
    write_huffman_tree(writer, &a_len);
    write_single_entry_tree(writer, 0); // distance (not used in palette)

    // Write palette pixels (differentially coded)
    let mut prev = 0u32;
    for &color in palette {
        let diff = sub_pixels(color, prev);

        let g = argb_green(diff) as usize;
        writer.write_bits(g_codes[g].code as u64, g_codes[g].length);

        let r = argb_red(diff) as usize;
        writer.write_bits(r_codes[r].code as u64, r_codes[r].length);

        let b = argb_blue(diff) as usize;
        writer.write_bits(b_codes[b].code as u64, b_codes[b].length);

        let a = argb_alpha(diff) as usize;
        writer.write_bits(a_codes[a].code as u64, a_codes[a].length);

        prev = color;
    }
}

/// Subtract two pixels component-wise (wrapping).
#[inline]
fn sub_pixels(a: u32, b: u32) -> u32 {
    let aa = argb_alpha(a).wrapping_sub(argb_alpha(b));
    let ar = argb_red(a).wrapping_sub(argb_red(b));
    let ag = argb_green(a).wrapping_sub(argb_green(b));
    let ab = argb_blue(a).wrapping_sub(argb_blue(b));
    make_argb(aa, ar, ag, ab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple() {
        // 4x4 red image
        let pixels: Vec<u8> = [255, 0, 0].repeat(16);
        let config = Vp8lConfig::default();

        let result = encode_vp8l(&pixels, 4, 4, false, &config);
        assert!(result.is_ok());

        let data = result.unwrap();
        // Check VP8L signature
        assert_eq!(data[0], 0x2f);
    }

    #[test]
    fn test_encode_with_alpha() {
        // 4x4 semi-transparent red
        let pixels: Vec<u8> = [255, 0, 0, 128].repeat(16);
        let config = Vp8lConfig::default();

        let result = encode_vp8l(&pixels, 4, 4, true, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_gradient() {
        // 16x16 gradient
        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for y in 0..16 {
            for x in 0..16 {
                pixels.push((x * 16) as u8);
                pixels.push((y * 16) as u8);
                pixels.push(128);
            }
        }

        let config = Vp8lConfig::default();
        let result = encode_vp8l(&pixels, 16, 16, false, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sub_pixels() {
        let a = make_argb(100, 50, 200, 10);
        let b = make_argb(50, 100, 100, 5);
        let diff = sub_pixels(a, b);

        assert_eq!(argb_alpha(diff), 50); // 100 - 50
        assert_eq!(argb_red(diff), 206); // 50 - 100 = -50 = 206 (wrapping)
        assert_eq!(argb_green(diff), 100); // 200 - 100
        assert_eq!(argb_blue(diff), 5); // 10 - 5
    }
}
