//! Color quantization for near-lossless palette encoding.
//!
//! Uses imagequant (Kornel Lesiński's libimagequant) to reduce images with >256 unique
//! colors to a 256-color palette. The quantized image can then be encoded losslessly
//! via VP8L's color indexing transform for significant compression gains.
//!
//! This module requires the `quantize` feature flag, which enables the `imagequant` dependency.
//! Note: imagequant is GPL-3.0-or-later, so enabling this feature changes the effective license.

use alloc::vec::Vec;

/// Result of color quantization.
pub struct QuantizedImage {
    /// ARGB pixels (each pixel is palette[index]), suitable for VP8L encoding.
    pub argb: Vec<u32>,
    /// Width of the image.
    pub width: u32,
    /// Height of the image.
    pub height: u32,
    /// Number of unique colors in the palette (≤256).
    pub palette_size: usize,
}

/// Quantize an RGB image to ≤256 colors using imagequant.
///
/// # Arguments
/// * `rgb` - Input RGB pixels (3 bytes per pixel, no alpha)
/// * `width` - Image width
/// * `height` - Image height
/// * `quality` - Quantization quality 0-100 (higher = better quality, more colors)
/// * `max_colors` - Maximum palette size (2-256)
///
/// # Returns
/// Quantized image with ARGB pixels using only palette colors, or None if
/// quantization fails (e.g., quality too low to achieve target).
pub fn quantize_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    max_colors: u16,
) -> Option<QuantizedImage> {
    quantize_rgba_impl(rgb, width, height, false, quality, max_colors)
}

/// Quantize an RGBA image to ≤256 colors using imagequant.
///
/// # Arguments
/// * `rgba` - Input RGBA pixels (4 bytes per pixel)
/// * `width` - Image width
/// * `height` - Image height
/// * `quality` - Quantization quality 0-100 (higher = better quality, more colors)
/// * `max_colors` - Maximum palette size (2-256)
///
/// # Returns
/// Quantized image with ARGB pixels using only palette colors, or None if
/// quantization fails (e.g., quality too low to achieve target).
pub fn quantize_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    max_colors: u16,
) -> Option<QuantizedImage> {
    quantize_rgba_impl(rgba, width, height, true, quality, max_colors)
}

fn quantize_rgba_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    quality: u8,
    max_colors: u16,
) -> Option<QuantizedImage> {
    let w = width as usize;
    let h = height as usize;
    let bpp = if has_alpha { 4 } else { 3 };
    let expected_len = w * h * bpp;
    if pixels.len() < expected_len {
        return None;
    }

    // Convert to imagequant RGBA format
    let rgba_pixels: Vec<imagequant::RGBA> = if has_alpha {
        pixels[..expected_len]
            .chunks_exact(4)
            .map(|p| imagequant::RGBA::new(p[0], p[1], p[2], p[3]))
            .collect()
    } else {
        pixels[..expected_len]
            .chunks_exact(3)
            .map(|p| imagequant::RGBA::new(p[0], p[1], p[2], 255))
            .collect()
    };

    // Configure quantizer
    let mut attr = imagequant::Attributes::new();
    let _ = attr.set_quality(0, quality);
    let max_colors = max_colors.clamp(2, 256);
    let _ = attr.set_max_colors(max_colors as u32);

    // Create image and quantize
    let mut image = attr.new_image(rgba_pixels, w, h, 0.0).ok()?;
    let mut result = attr.quantize(&mut image).ok()?;

    // Disable dithering for lossless encoding (dithering adds noise)
    result.set_dithering_level(0.0).ok()?;

    // Get remapped palette and indices
    let (palette, indices) = result.remapped(&mut image).ok()?;
    let palette_size = palette.len();

    // Convert to ARGB format used by VP8L
    let argb: Vec<u32> = indices
        .iter()
        .map(|&idx| {
            let c = &palette[idx as usize];
            ((c.a as u32) << 24) | ((c.r as u32) << 16) | ((c.g as u32) << 8) | (c.b as u32)
        })
        .collect();

    Some(QuantizedImage {
        argb,
        width,
        height,
        palette_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_uniform() {
        // Single color image → should quantize to 1-2 colors
        let rgb: Vec<u8> = vec![128, 64, 32].repeat(100);
        let result = quantize_rgb(&rgb, 10, 10, 100, 256).unwrap();
        assert_eq!(result.argb.len(), 100);
        assert!(result.palette_size <= 2);
        // All pixels should be the same (or very close)
        let first = result.argb[0];
        assert!(result.argb.iter().all(|&p| p == first));
    }

    #[test]
    fn test_quantize_max_colors() {
        // Random-ish image with many colors
        let mut rgb = Vec::with_capacity(256 * 256 * 3);
        for y in 0..256u32 {
            for x in 0..256u32 {
                rgb.push((x & 0xFF) as u8);
                rgb.push((y & 0xFF) as u8);
                rgb.push(((x + y) & 0xFF) as u8);
            }
        }
        let result = quantize_rgb(&rgb, 256, 256, 80, 64).unwrap();
        assert_eq!(result.argb.len(), 256 * 256);
        assert!(result.palette_size <= 64);
    }
}
