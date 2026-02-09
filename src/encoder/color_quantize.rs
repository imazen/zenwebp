//! Color quantization for near-lossless palette encoding.
//!
//! Reduces images with >256 unique colors to a 256-color palette. The quantized image
//! can then be encoded losslessly via VP8L's color indexing transform for significant
//! compression gains.
//!
//! # Backends
//!
//! Three backends are available (choose via feature flags):
//!
//! - **`quantize-zenquant`** (default): Best perceptual quality via AQ-informed
//!   palette selection. AGPL-3.0-or-later licensed.
//!
//! - **`quantize-quantizr`**: MIT-licensed backend using the `quantizr` crate.
//!   Decent quality, compatible with MIT/Apache-2.0 licensing.
//!
//! - **`quantize-imagequant`**: GPL-3.0-or-later backend using Kornel Lesiński's
//!   [`imagequant`](https://github.com/ImageOptim/libimagequant) crate.
//!   Good compression, but requires GPL-3.0-or-later compliance.
//!   [Commercial license available from upstream](https://supso.org/projects/pngquant).
//!
//! The `quantize` feature is an alias for `quantize-zenquant`.
//!
//! # Example
//!
//! ```toml
//! # Use default backend (best quality)
//! zenwebp = { version = "0.3", features = ["quantize"] }
//!
//! # Or explicitly choose MIT backend
//! zenwebp = { version = "0.3", features = ["quantize-quantizr"] }
//! ```

use alloc::vec::Vec;

/// Result of color quantization.
pub struct QuantizedImage {
    /// ARGB pixels (each pixel is palette\[index\]), suitable for VP8L encoding.
    pub argb: Vec<u32>,
    /// Width of the image.
    pub width: u32,
    /// Height of the image.
    pub height: u32,
    /// Number of unique colors in the palette (≤256).
    pub palette_size: usize,
}

/// Quantize an RGB image to ≤256 colors.
///
/// Uses the selected backend (quantizr or imagequant) based on feature flags.
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

/// Quantize an RGBA image to ≤256 colors.
///
/// Uses the selected backend (quantizr or imagequant) based on feature flags.
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

#[cfg(feature = "quantize-zenquant")]
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

    // Build zenquant config
    let mut config = zenquant::QuantizeConfig::new(zenquant::OutputFormat::WebpLossless);
    config = if quality >= 75 {
        config.quality(zenquant::Quality::Best)
    } else if quality >= 40 {
        config.quality(zenquant::Quality::Balanced)
    } else {
        config.quality(zenquant::Quality::Fast)
    };
    let max_colors = max_colors.clamp(2, 256);
    config = config.max_colors(max_colors as u32);
    // Disable dithering for lossless encoding (dithering adds noise)
    config = config._no_dither();

    if has_alpha {
        // RGBA path
        let rgba_pixels: Vec<zenquant::RGBA<u8>> = pixels[..expected_len]
            .chunks_exact(4)
            .map(|p| zenquant::RGBA::new(p[0], p[1], p[2], p[3]))
            .collect();

        let result = zenquant::quantize_rgba(&rgba_pixels, w, h, &config).ok()?;
        let palette = result.palette_rgba();
        let palette_size = result.palette_len();

        let argb: Vec<u32> = result
            .indices()
            .iter()
            .map(|&idx| {
                let c = palette[idx as usize];
                ((c[3] as u32) << 24) | ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
            })
            .collect();

        Some(QuantizedImage {
            argb,
            width,
            height,
            palette_size,
        })
    } else {
        // RGB path
        let rgb_pixels: Vec<zenquant::RGB<u8>> = pixels[..expected_len]
            .chunks_exact(3)
            .map(|p| zenquant::RGB::new(p[0], p[1], p[2]))
            .collect();

        let result = zenquant::quantize(&rgb_pixels, w, h, &config).ok()?;
        let palette = result.palette();
        let palette_size = result.palette_len();

        let argb: Vec<u32> = result
            .indices()
            .iter()
            .map(|&idx| {
                let c = palette[idx as usize];
                (0xFF00_0000) | ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
            })
            .collect();

        Some(QuantizedImage {
            argb,
            width,
            height,
            palette_size,
        })
    }
}

#[cfg(all(feature = "quantize-imagequant", not(feature = "quantize-zenquant")))]
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

#[cfg(all(feature = "quantize-quantizr", not(feature = "quantize-zenquant"), not(feature = "quantize-imagequant")))]
fn quantize_rgba_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    _quality: u8, // quantizr doesn't have quality parameter
    max_colors: u16,
) -> Option<QuantizedImage> {
    let w = width as usize;
    let h = height as usize;
    let bpp = if has_alpha { 4 } else { 3 };
    let expected_len = w * h * bpp;
    if pixels.len() < expected_len {
        return None;
    }

    // Convert to RGBA format (quantizr expects contiguous RGBA)
    let rgba_pixels: Vec<u8> = if has_alpha {
        pixels[..expected_len].to_vec()
    } else {
        // Expand RGB to RGBA
        let mut rgba = Vec::with_capacity(w * h * 4);
        for chunk in pixels[..expected_len].chunks_exact(3) {
            rgba.push(chunk[0]); // R
            rgba.push(chunk[1]); // G
            rgba.push(chunk[2]); // B
            rgba.push(255); // A
        }
        rgba
    };

    // Create image
    let image = quantizr::Image::new(&rgba_pixels, w, h).ok()?;

    // Configure options
    let mut opts = quantizr::Options::default();
    let max_colors = max_colors.clamp(2, 256) as i32;
    opts.set_max_colors(max_colors).ok()?;

    // Quantize
    let mut result = quantizr::QuantizeResult::quantize(&image, &opts);

    // Disable dithering for lossless encoding
    result.set_dithering_level(0.0).ok()?;

    // Get palette and remap
    let palette = result.get_palette();
    let palette_size = palette.count as usize;

    let mut indices = vec![0u8; w * h];
    result.remap_image(&image, &mut indices).ok()?;

    // Convert to ARGB format used by VP8L
    let argb: Vec<u32> = indices
        .iter()
        .map(|&idx| {
            let c = &palette.entries[idx as usize];
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
        let rgb: Vec<u8> = [128, 64, 32].repeat(100);
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
