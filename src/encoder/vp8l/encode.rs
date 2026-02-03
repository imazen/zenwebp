//! Main VP8L encoder implementation.

use alloc::vec::Vec;

use super::backward_refs::get_backward_references;
use super::bitwriter::BitWriter;
use super::color_cache::ColorCache;
use super::histogram::{
    distance_code_extra_bits, distance_code_to_prefix, length_code_extra_bits, length_to_code,
};
use super::huffman::{
    build_huffman_codes, build_huffman_lengths, write_huffman_tree, write_single_entry_tree,
    HuffmanCode,
};
use super::meta_huffman::{build_meta_huffman, build_single_histogram, MetaHuffmanInfo};
use super::transforms::{
    apply_cross_color_transform, apply_predictor_transform, apply_subtract_green,
    ColorIndexTransform,
};
use super::types::{
    argb_alpha, argb_blue, argb_green, argb_red, make_argb, subsample_size, PixOrCopy, Vp8lConfig,
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
    let use_cross_color = config.use_cross_color && !use_palette;
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
        let pred_bits = config.predictor_bits.clamp(2, 8);
        let predictor_data = apply_predictor_transform(argb, width, height, pred_bits);

        // Signal predictor transform
        writer.write_bit(true); // transform present
        writer.write_bits(0, 2); // predictor = 0
        writer.write_bits((pred_bits - 2) as u64, 3);

        // Write predictor sub-image
        write_predictor_image(&mut writer, &predictor_data);
    }

    // Cross-color transform (applied second, decorrelates color channels)
    if use_cross_color && palette_transform.is_none() {
        let cc_bits = config.cross_color_bits.clamp(2, 8);
        let cross_color_data =
            apply_cross_color_transform(argb, width, height, cc_bits, config.quality.quality);

        // Signal cross-color transform
        writer.write_bit(true); // transform present
        writer.write_bits(1, 2); // cross-color = 1
        writer.write_bits((cc_bits - 2) as u64, 3);

        // Write cross-color sub-image
        write_cross_color_image(&mut writer, &cross_color_data);
    }

    // Subtract green transform (applied third, on cross-color-adjusted residuals)
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
    let (refs, cache_bits) =
        get_backward_references(argb, width, height, config.quality.quality, cache_bits_max);

    // Write color cache info
    if cache_bits > 0 {
        writer.write_bit(true);
        writer.write_bits(cache_bits as u64, 4);
    } else {
        writer.write_bit(false);
    }

    // Build histogram(s) — either single or meta-Huffman with clustering
    let use_meta = config.use_meta_huffman && width > 16 && height > 16;
    let histo_bits = if use_meta {
        get_histo_bits(width, height, config.quality.method)
    } else {
        0u8
    };

    let meta_info = if use_meta {
        build_meta_huffman(
            &refs,
            width,
            height,
            histo_bits,
            cache_bits,
            config.quality.quality,
        )
    } else {
        build_single_histogram(&refs, cache_bits)
    };

    // Write meta-Huffman flag and prefix image
    if meta_info.num_histograms > 1 {
        writer.write_bit(true); // meta-Huffman present
        writer.write_bits((meta_info.histo_bits - 2) as u64, 3);

        // Write histogram prefix image (encoded as sub-image)
        write_histogram_image(&mut writer, &meta_info);
    } else {
        writer.write_bit(false); // no meta-Huffman
    }

    // Build Huffman codes for each histogram group (5 trees per group)
    let mut all_codes: Vec<HuffmanGroupCodes> = Vec::with_capacity(meta_info.num_histograms);
    for h in &meta_info.histograms {
        all_codes.push(build_huffman_group(h));
    }

    // Write all Huffman trees (5 per histogram group)
    for group in &all_codes {
        write_huffman_tree(&mut writer, &group.literal_lengths);
        write_huffman_tree(&mut writer, &group.red_lengths);
        write_huffman_tree(&mut writer, &group.blue_lengths);
        write_huffman_tree(&mut writer, &group.alpha_lengths);
        write_huffman_tree(&mut writer, &group.dist_lengths);
    }

    // Write image data with spatially-varying Huffman selection
    let mut color_cache = if cache_bits > 0 {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    let histo_xsize = if meta_info.num_histograms > 1 {
        subsample_size(width as u32, meta_info.histo_bits) as usize
    } else {
        0
    };

    let mut x = 0usize;
    let mut y = 0usize;
    let mut argb_idx = 0usize;

    for token in refs.iter() {
        // Select Huffman group for current tile
        let group_idx = if meta_info.num_histograms > 1 {
            let tile_idx = (y >> meta_info.histo_bits) * histo_xsize + (x >> meta_info.histo_bits);
            meta_info.histogram_symbols[tile_idx] as usize
        } else {
            0
        };
        let codes = &all_codes[group_idx];

        match *token {
            PixOrCopy::Literal(argb_val) => {
                if !codes.literal_trivial {
                    let g = argb_green(argb_val) as usize;
                    writer.write_bits(
                        codes.literal_codes[g].code as u64,
                        codes.literal_codes[g].length,
                    );
                }
                if !codes.red_trivial {
                    let r = argb_red(argb_val) as usize;
                    writer.write_bits(codes.red_codes[r].code as u64, codes.red_codes[r].length);
                }
                if !codes.blue_trivial {
                    let b = argb_blue(argb_val) as usize;
                    writer.write_bits(codes.blue_codes[b].code as u64, codes.blue_codes[b].length);
                }
                if !codes.alpha_trivial {
                    let a = argb_alpha(argb_val) as usize;
                    writer.write_bits(
                        codes.alpha_codes[a].code as u64,
                        codes.alpha_codes[a].length,
                    );
                }

                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb_val);
                }
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
                argb_idx += 1;
            }
            PixOrCopy::CacheIdx(idx) => {
                if !codes.literal_trivial {
                    let code = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
                    writer.write_bits(
                        codes.literal_codes[code].code as u64,
                        codes.literal_codes[code].length,
                    );
                }

                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb[argb_idx]);
                }
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
                argb_idx += 1;
            }
            PixOrCopy::Copy { len, dist } => {
                let (len_prefix, len_extra) = length_to_code(len);
                let len_code = NUM_LITERAL_CODES + len_prefix as usize;
                if !codes.literal_trivial {
                    writer.write_bits(
                        codes.literal_codes[len_code].code as u64,
                        codes.literal_codes[len_code].length,
                    );
                }

                let len_extra_bits = length_code_extra_bits(len_prefix);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra as u64, len_extra_bits);
                }

                let (dist_prefix, dist_extra) = distance_code_to_prefix(dist);
                if !codes.dist_trivial {
                    writer.write_bits(
                        codes.dist_codes[dist_prefix as usize].code as u64,
                        codes.dist_codes[dist_prefix as usize].length,
                    );
                }

                let dist_extra_bits = distance_code_extra_bits(dist_prefix);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra as u64, dist_extra_bits);
                }

                if let Some(ref mut cache) = color_cache {
                    for k in 0..len as usize {
                        cache.insert(argb[argb_idx + k]);
                    }
                }
                for _ in 0..len {
                    x += 1;
                    if x >= width {
                        x = 0;
                        y += 1;
                    }
                }
                argb_idx += len as usize;
            }
        }
    }

    Ok(writer.finish())
}

/// Maximum number of histogram tiles (matching libwebp's MAX_HUFF_IMAGE_SIZE).
const MAX_HUFF_IMAGE_SIZE: usize = 2600;
/// Min/max Huffman bits range (VP8L spec: 3 bits, range [2, 9]).
const MIN_HUFFMAN_BITS: u8 = 2;
const MAX_HUFFMAN_BITS: u8 = 9; // 2 + (1 << 3) - 1

/// Calculate optimal histogram bits based on method and image size.
/// Matches libwebp's GetHistoBits + ClampBits.
fn get_histo_bits(width: usize, height: usize, method: u8) -> u8 {
    // Make tile size a function of encoding method
    let histo_bits = 7i32 - method as i32;
    let mut bits = histo_bits.clamp(MIN_HUFFMAN_BITS as i32, MAX_HUFFMAN_BITS as i32) as u8;

    // Clamp to keep number of tiles under MAX_HUFF_IMAGE_SIZE
    let mut image_size =
        subsample_size(width as u32, bits) as usize * subsample_size(height as u32, bits) as usize;
    while bits < MAX_HUFFMAN_BITS && image_size > MAX_HUFF_IMAGE_SIZE {
        bits += 1;
        image_size = subsample_size(width as u32, bits) as usize
            * subsample_size(height as u32, bits) as usize;
    }

    bits
}

/// Pre-built Huffman codes for one histogram group (5 trees).
struct HuffmanGroupCodes {
    literal_lengths: Vec<u8>,
    literal_codes: Vec<HuffmanCode>,
    literal_trivial: bool,
    red_lengths: Vec<u8>,
    red_codes: Vec<HuffmanCode>,
    red_trivial: bool,
    blue_lengths: Vec<u8>,
    blue_codes: Vec<HuffmanCode>,
    blue_trivial: bool,
    alpha_lengths: Vec<u8>,
    alpha_codes: Vec<HuffmanCode>,
    alpha_trivial: bool,
    dist_lengths: Vec<u8>,
    dist_codes: Vec<HuffmanCode>,
    dist_trivial: bool,
}

/// Build Huffman codes for one histogram.
fn build_huffman_group(h: &super::histogram::Histogram) -> HuffmanGroupCodes {
    let literal_lengths = build_huffman_lengths(&h.literal, 15);
    let literal_codes = build_huffman_codes(&literal_lengths);
    let literal_trivial = literal_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let red_lengths = build_huffman_lengths(&h.red, 15);
    let red_codes = build_huffman_codes(&red_lengths);
    let red_trivial = red_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let blue_lengths = build_huffman_lengths(&h.blue, 15);
    let blue_codes = build_huffman_codes(&blue_lengths);
    let blue_trivial = blue_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let alpha_lengths = build_huffman_lengths(&h.alpha, 15);
    let alpha_codes = build_huffman_codes(&alpha_lengths);
    let alpha_trivial = alpha_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let dist_lengths = build_huffman_lengths(&h.distance, 15);
    let dist_codes = build_huffman_codes(&dist_lengths);
    let dist_trivial = dist_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    HuffmanGroupCodes {
        literal_lengths,
        literal_codes,
        literal_trivial,
        red_lengths,
        red_codes,
        red_trivial,
        blue_lengths,
        blue_codes,
        blue_trivial,
        alpha_lengths,
        alpha_codes,
        alpha_trivial,
        dist_lengths,
        dist_codes,
        dist_trivial,
    }
}

/// Write the histogram prefix image (meta-Huffman sub-image).
///
/// Each pixel encodes a histogram group index. The index is stored
/// as: green = (index & 0xFF), red = (index >> 8), blue = 0, alpha = 0.
fn write_histogram_image(writer: &mut BitWriter, meta_info: &MetaHuffmanInfo) {
    // No color cache for sub-images
    writer.write_bit(false);

    // Build histogram of group indices
    let mut hist_green = [0u32; 280]; // green channel = low byte of index
    let mut hist_red = [0u32; 256]; // red channel = high byte of index
    for &sym in &meta_info.histogram_symbols {
        hist_green[(sym & 0xFF) as usize] += 1;
        hist_red[(sym >> 8) as usize] += 1;
    }

    // Build Huffman codes
    let g_lengths = build_huffman_lengths(&hist_green, 15);
    let g_codes = build_huffman_codes(&g_lengths);
    let r_lengths = build_huffman_lengths(&hist_red, 15);
    let r_codes = build_huffman_codes(&r_lengths);

    let g_trivial = g_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let r_trivial = r_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    // Write 5 Huffman trees for the sub-image
    write_huffman_tree(writer, &g_lengths); // green (group index low byte)
    write_huffman_tree(writer, &r_lengths); // red (group index high byte)
    write_single_entry_tree(writer, 0); // blue (always 0)
    write_single_entry_tree(writer, 0); // alpha (always 0)
    write_single_entry_tree(writer, 0); // distance (unused)

    // Write image data: one pixel per tile
    for &sym in &meta_info.histogram_symbols {
        let g = (sym & 0xFF) as usize;
        let r = (sym >> 8) as usize;

        if !g_trivial {
            writer.write_bits(g_codes[g].code as u64, g_codes[g].length);
        }
        if !r_trivial {
            writer.write_bits(r_codes[r].code as u64, r_codes[r].length);
        }
        // blue and alpha are trivial (0), no bits emitted
    }
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

/// Write cross-color sub-image (multiplier data).
///
/// Each pixel encodes three multipliers:
///   blue channel  = green_to_red
///   green channel = green_to_blue
///   red channel   = red_to_blue
///   alpha         = 0xFF
fn write_cross_color_image(writer: &mut BitWriter, cross_color_data: &[u32]) {
    // No color cache for sub-images
    writer.write_bit(false);

    // Build histogram for all channels
    let mut hist_green = [0u32; 280]; // green_to_blue values
    let mut hist_red = [0u32; 256]; // red_to_blue values
    let mut hist_blue = [0u32; 256]; // green_to_red values
    let mut hist_alpha = [0u32; 256]; // always 0xFF

    for &pixel in cross_color_data {
        hist_green[argb_green(pixel) as usize] += 1;
        hist_red[argb_red(pixel) as usize] += 1;
        hist_blue[argb_blue(pixel) as usize] += 1;
        hist_alpha[argb_alpha(pixel) as usize] += 1;
    }

    // Build Huffman codes
    let green_lengths = build_huffman_lengths(&hist_green, 15);
    let green_codes = build_huffman_codes(&green_lengths);
    let red_lengths = build_huffman_lengths(&hist_red, 15);
    let red_codes = build_huffman_codes(&red_lengths);
    let blue_lengths = build_huffman_lengths(&hist_blue, 15);
    let blue_codes = build_huffman_codes(&blue_lengths);
    let alpha_lengths = build_huffman_lengths(&hist_alpha, 15);
    let alpha_codes = build_huffman_codes(&alpha_lengths);

    // Check for trivial trees
    let green_trivial = green_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let red_trivial = red_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let blue_trivial = blue_lengths.iter().filter(|&&l| l > 0).count() <= 1;
    let alpha_trivial = alpha_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    // Write 5 Huffman trees
    write_huffman_tree(writer, &green_lengths);
    write_huffman_tree(writer, &red_lengths);
    write_huffman_tree(writer, &blue_lengths);
    write_huffman_tree(writer, &alpha_lengths);
    write_single_entry_tree(writer, 0); // distance (unused)

    // Write image data
    for &pixel in cross_color_data {
        if !green_trivial {
            let g = argb_green(pixel) as usize;
            writer.write_bits(green_codes[g].code as u64, green_codes[g].length);
        }
        if !red_trivial {
            let r = argb_red(pixel) as usize;
            writer.write_bits(red_codes[r].code as u64, red_codes[r].length);
        }
        if !blue_trivial {
            let b = argb_blue(pixel) as usize;
            writer.write_bits(blue_codes[b].code as u64, blue_codes[b].length);
        }
        if !alpha_trivial {
            let a = argb_alpha(pixel) as usize;
            writer.write_bits(alpha_codes[a].code as u64, alpha_codes[a].length);
        }
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
