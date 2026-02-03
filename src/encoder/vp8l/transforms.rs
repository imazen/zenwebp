//! Image transforms for VP8L encoding.
//!
//! Transforms are applied to decorrelate image data before entropy coding.

use alloc::vec;
use alloc::vec::Vec;

use super::types::{argb_alpha, argb_blue, argb_green, argb_red, make_argb, subsample_size};

/// Transform type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransformType {
    Predictor = 0,
    CrossColor = 1,
    SubtractGreen = 2,
    ColorIndexing = 3,
}

/// Predictor modes (14 total, per VP8L spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PredictorMode {
    Black = 0,                 // 0xff000000
    Left = 1,                  // L
    Top = 2,                   // T
    TopRight = 3,              // TR
    TopLeft = 4,               // TL
    AvgAvgLtrT = 5,            // Avg(Avg(L,TR), T)
    AvgLTl = 6,                // Avg(L, TL)
    AvgLT = 7,                 // Avg(L, T)
    AvgTlT = 8,                // Avg(TL, T)
    AvgTTr = 9,                // Avg(T, TR)
    AvgAvgLTlAvgTTr = 10,      // Avg(Avg(L,TL), Avg(T,TR))
    Select = 11,               // Select(L, T, TL)
    ClampAddSubtractFull = 12, // Clamp(L + T - TL)
    ClampAddSubtractHalf = 13, // Clamp(Avg(L,T) + (Avg(L,T)-TL)/2)
}

impl PredictorMode {
    /// Get all predictor modes.
    pub const fn all() -> [PredictorMode; 14] {
        [
            PredictorMode::Black,
            PredictorMode::Left,
            PredictorMode::Top,
            PredictorMode::TopRight,
            PredictorMode::TopLeft,
            PredictorMode::AvgAvgLtrT,
            PredictorMode::AvgLTl,
            PredictorMode::AvgLT,
            PredictorMode::AvgTlT,
            PredictorMode::AvgTTr,
            PredictorMode::AvgAvgLTlAvgTTr,
            PredictorMode::Select,
            PredictorMode::ClampAddSubtractFull,
            PredictorMode::ClampAddSubtractHalf,
        ]
    }

    /// Convert from u8 mode value.
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => PredictorMode::Black,
            1 => PredictorMode::Left,
            2 => PredictorMode::Top,
            3 => PredictorMode::TopRight,
            4 => PredictorMode::TopLeft,
            5 => PredictorMode::AvgAvgLtrT,
            6 => PredictorMode::AvgLTl,
            7 => PredictorMode::AvgLT,
            8 => PredictorMode::AvgTlT,
            9 => PredictorMode::AvgTTr,
            10 => PredictorMode::AvgAvgLTlAvgTTr,
            11 => PredictorMode::Select,
            12 => PredictorMode::ClampAddSubtractFull,
            _ => PredictorMode::ClampAddSubtractHalf,
        }
    }
}

/// Apply subtract green transform in place.
/// R -= G, B -= G
pub fn apply_subtract_green(pixels: &mut [u32]) {
    for pixel in pixels.iter_mut() {
        let a = argb_alpha(*pixel);
        let r = argb_red(*pixel);
        let g = argb_green(*pixel);
        let b = argb_blue(*pixel);

        let new_r = r.wrapping_sub(g);
        let new_b = b.wrapping_sub(g);

        *pixel = make_argb(a, new_r, g, new_b);
    }
}

/// Predict a pixel using the given mode and neighbors.
/// Must match the decoder's predictor functions exactly for lossless round-tripping.
#[inline]
fn predict(mode: PredictorMode, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        PredictorMode::Black => 0xff000000,
        PredictorMode::Left => left,
        PredictorMode::Top => top,
        PredictorMode::TopRight => top_right,
        PredictorMode::TopLeft => top_left,
        PredictorMode::AvgAvgLtrT => average2(average2(left, top_right), top),
        PredictorMode::AvgLTl => average2(left, top_left),
        PredictorMode::AvgLT => average2(left, top),
        PredictorMode::AvgTlT => average2(top_left, top),
        PredictorMode::AvgTTr => average2(top, top_right),
        PredictorMode::AvgAvgLTlAvgTTr => {
            average2(average2(left, top_left), average2(top, top_right))
        }
        PredictorMode::Select => select(left, top, top_left),
        PredictorMode::ClampAddSubtractFull => clamp_add_subtract_full(left, top, top_left),
        PredictorMode::ClampAddSubtractHalf => clamp_add_subtract_half(left, top, top_left),
    }
}

/// Average two pixels component-wise.
#[inline]
fn average2(a: u32, b: u32) -> u32 {
    let aa = argb_alpha(a) as u16 + argb_alpha(b) as u16;
    let ar = argb_red(a) as u16 + argb_red(b) as u16;
    let ag = argb_green(a) as u16 + argb_green(b) as u16;
    let ab = argb_blue(a) as u16 + argb_blue(b) as u16;
    make_argb(
        (aa / 2) as u8,
        (ar / 2) as u8,
        (ag / 2) as u8,
        (ab / 2) as u8,
    )
}

/// Select predictor: choose left or top based on gradient.
/// Matches decoder's tie-breaking: when distances are equal, returns top.
#[inline]
fn select(left: u32, top: u32, top_left: u32) -> u32 {
    // predict_left = |T - TL|: distance from prediction (L+T-TL) to L
    let pa = (argb_alpha(top) as i16 - argb_alpha(top_left) as i16).unsigned_abs();
    let pr = (argb_red(top) as i16 - argb_red(top_left) as i16).unsigned_abs();
    let pg = (argb_green(top) as i16 - argb_green(top_left) as i16).unsigned_abs();
    let pb = (argb_blue(top) as i16 - argb_blue(top_left) as i16).unsigned_abs();
    let predict_left = pa + pr + pg + pb;

    // predict_top = |L - TL|: distance from prediction (L+T-TL) to T
    let pa = (argb_alpha(left) as i16 - argb_alpha(top_left) as i16).unsigned_abs();
    let pr = (argb_red(left) as i16 - argb_red(top_left) as i16).unsigned_abs();
    let pg = (argb_green(left) as i16 - argb_green(top_left) as i16).unsigned_abs();
    let pb = (argb_blue(left) as i16 - argb_blue(top_left) as i16).unsigned_abs();
    let predict_top = pa + pr + pg + pb;

    // If prediction is closer to left, choose left; otherwise choose top.
    // On ties (predict_left == predict_top), choose top (matching decoder).
    if predict_left < predict_top {
        left
    } else {
        top
    }
}

/// Clamp a value to [0, 255].
#[inline]
fn clamp(val: i16) -> u8 {
    val.clamp(0, 255) as u8
}

/// ClampAddSubtractFull: left + top - top_left, clamped.
#[inline]
fn clamp_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let a = clamp(argb_alpha(left) as i16 + argb_alpha(top) as i16 - argb_alpha(top_left) as i16);
    let r = clamp(argb_red(left) as i16 + argb_red(top) as i16 - argb_red(top_left) as i16);
    let g = clamp(argb_green(left) as i16 + argb_green(top) as i16 - argb_green(top_left) as i16);
    let b = clamp(argb_blue(left) as i16 + argb_blue(top) as i16 - argb_blue(top_left) as i16);
    make_argb(a, r, g, b)
}

/// ClampAddSubtractHalf: clamp(avg + (avg - top_left) / 2) per component,
/// where avg = (left + top) / 2.
#[inline]
fn clamp_add_subtract_half(left: u32, top: u32, top_left: u32) -> u32 {
    let avg_a = (argb_alpha(left) as i16 + argb_alpha(top) as i16) / 2;
    let avg_r = (argb_red(left) as i16 + argb_red(top) as i16) / 2;
    let avg_g = (argb_green(left) as i16 + argb_green(top) as i16) / 2;
    let avg_b = (argb_blue(left) as i16 + argb_blue(top) as i16) / 2;
    let a = (avg_a + (avg_a - argb_alpha(top_left) as i16) / 2).clamp(0, 255) as u8;
    let r = (avg_r + (avg_r - argb_red(top_left) as i16) / 2).clamp(0, 255) as u8;
    let g = (avg_g + (avg_g - argb_green(top_left) as i16) / 2).clamp(0, 255) as u8;
    let b = (avg_b + (avg_b - argb_blue(top_left) as i16) / 2).clamp(0, 255) as u8;
    make_argb(a, r, g, b)
}

/// Compute residual (pixel - prediction) with wrapping.
#[inline]
fn residual(pixel: u32, pred: u32) -> u32 {
    let a = argb_alpha(pixel).wrapping_sub(argb_alpha(pred));
    let r = argb_red(pixel).wrapping_sub(argb_red(pred));
    let g = argb_green(pixel).wrapping_sub(argb_green(pred));
    let b = argb_blue(pixel).wrapping_sub(argb_blue(pred));
    make_argb(a, r, g, b)
}

/// Apply predictor transform.
/// Returns the predictor mode data (subsampled image of modes).
///
/// Border handling matches the VP8L decoder:
/// - (0,0): Black predictor (0xff000000)
/// - Row 0, x>0: Left predictor (fixed, regardless of mode)
/// - Col 0, y>0: Top predictor (fixed, regardless of mode)
/// - Interior (y>0, x>0): mode from predictor_data
///
/// Pixels are processed in reverse order so neighbors remain as original values.
pub fn apply_predictor_transform(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    size_bits: u8,
) -> Vec<u32> {
    let block_size = 1usize << size_bits;
    let blocks_x = subsample_size(width as u32, size_bits) as usize;
    let blocks_y = subsample_size(height as u32, size_bits) as usize;

    // Choose best predictor for each block (only scoring interior pixels)
    let mut predictor_data = vec![0u32; blocks_x * blocks_y];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let best_mode = choose_best_predictor(pixels, width, height, bx, by, block_size);
            // Store mode in green channel (as per spec), alpha=0
            predictor_data[by * blocks_x + bx] = make_argb(0, 0, best_mode as u8, 0);
        }
    }

    // Apply prediction to pixels in reverse order.
    // Border pixels use fixed predictors matching the decoder.
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let pred = if y == 0 && x == 0 {
                // Top-left corner: Black
                0xff000000
            } else if y == 0 {
                // First row: Left predictor
                pixels[x - 1]
            } else if x == 0 {
                // First column: Top predictor
                pixels[(y - 1) * width]
            } else {
                // Interior: use block's predictor mode
                let bx = x >> size_bits;
                let by = y >> size_bits;
                let mode = PredictorMode::from_u8(argb_green(predictor_data[by * blocks_x + bx]));
                let left = pixels[y * width + x - 1];
                let top = pixels[(y - 1) * width + x];
                let top_left = pixels[(y - 1) * width + x - 1];
                // At right edge (x == width-1), top_right wraps to the first
                // pixel of the current row. This matches the decoder's memory
                // layout where image_data[i - width*4 + 4] wraps to row y's start.
                let top_right = if x + 1 < width {
                    pixels[(y - 1) * width + x + 1]
                } else {
                    pixels[y * width]
                };
                predict(mode, left, top, top_left, top_right)
            };
            pixels[y * width + x] = residual(pixels[y * width + x], pred);
        }
    }

    predictor_data
}

/// Choose the best predictor mode for a block.
/// Only scores interior pixels (y > 0 AND x > 0) since border pixels use
/// fixed predictors regardless of the block's mode.
fn choose_best_predictor(
    pixels: &[u32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
) -> PredictorMode {
    let x_start = bx * block_size;
    let y_start = by * block_size;
    let x_end = (x_start + block_size).min(width);
    let y_end = (y_start + block_size).min(height);

    // Effective start: skip row 0 and col 0 (they use fixed predictors)
    let x_eff = x_start.max(1);
    let y_eff = y_start.max(1);

    // If no interior pixels to score, default to Top
    if x_eff >= x_end || y_eff >= y_end {
        return PredictorMode::Top;
    }

    let mut best_mode = PredictorMode::Top;
    let mut best_score = u64::MAX;

    for mode in PredictorMode::all() {
        let mut score = 0u64;

        for y in y_eff..y_end {
            for x in x_eff..x_end {
                let pixel = pixels[y * width + x];
                let left = pixels[y * width + x - 1];
                let top = pixels[(y - 1) * width + x];
                let top_left = pixels[(y - 1) * width + x - 1];
                // At right edge, top_right wraps to first pixel of current row
                // (matching decoder's memory layout behavior).
                let top_right = if x + 1 < width {
                    pixels[(y - 1) * width + x + 1]
                } else {
                    pixels[y * width]
                };

                let pred = predict(mode, left, top, top_left, top_right);
                let res = residual(pixel, pred);

                // Score: sum of absolute residuals across all channels
                score += argb_alpha(res) as u64;
                score += argb_red(res) as u64;
                score += argb_green(res) as u64;
                score += argb_blue(res) as u64;
            }
        }

        if score < best_score {
            best_score = score;
            best_mode = mode;
        }
    }

    best_mode
}

/// Simple predictor transform using only vertical prediction.
/// Used for quick encoding at lower quality levels.
pub fn apply_simple_predictor(pixels: &mut [u32], width: usize, height: usize) {
    // Process from bottom-right to top-left
    for y in (1..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            let top = pixels[(y - 1) * width + x];
            pixels[idx] = residual(pixels[idx], top);
        }
    }
    // First row: subtract left neighbor
    for x in (1..width).rev() {
        let left = pixels[x - 1];
        pixels[x] = residual(pixels[x], left);
    }
    // Top-left corner: subtract 0xff000000 (black with full alpha)
    pixels[0] = residual(pixels[0], 0xff000000);
}

/// Color indexing (palette) transform data.
pub struct ColorIndexTransform {
    pub palette: Vec<u32>,
}

impl ColorIndexTransform {
    /// Try to build a color index transform.
    /// Returns None if image has more than 256 colors.
    pub fn try_build(pixels: &[u32]) -> Option<Self> {
        let mut palette = Vec::with_capacity(256);
        let mut seen = alloc::collections::BTreeSet::new();

        for &pixel in pixels {
            if seen.insert(pixel) {
                if palette.len() >= 256 {
                    return None;
                }
                palette.push(pixel);
            }
        }

        Some(Self { palette })
    }

    /// Get bits per pixel based on palette size.
    pub fn bits_per_pixel(&self) -> u8 {
        let n = self.palette.len();
        if n <= 2 {
            3 // Pack 8 pixels per pixel (use only 1 bit each)
        } else if n <= 4 {
            2 // Pack 4 pixels per pixel (use only 2 bits each)
        } else if n <= 16 {
            1 // Pack 2 pixels per pixel (use only 4 bits each)
        } else {
            0 // No packing, 8 bits each
        }
    }

    /// Apply the transform: convert ARGB to palette indices.
    pub fn apply(&self, pixels: &mut [u32]) {
        // Build reverse lookup
        let mut lookup = alloc::collections::BTreeMap::new();
        for (i, &color) in self.palette.iter().enumerate() {
            lookup.insert(color, i as u8);
        }

        for pixel in pixels.iter_mut() {
            let idx = lookup[pixel];
            // Store index in green channel
            *pixel = make_argb(255, 0, idx, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtract_green() {
        let mut pixels = vec![make_argb(255, 100, 50, 150)];
        apply_subtract_green(&mut pixels);

        let p = pixels[0];
        assert_eq!(argb_alpha(p), 255);
        assert_eq!(argb_red(p), 50); // 100 - 50
        assert_eq!(argb_green(p), 50); // unchanged
        assert_eq!(argb_blue(p), 100); // 150 - 50
    }

    #[test]
    fn test_average2() {
        let a = make_argb(100, 100, 100, 100);
        let b = make_argb(200, 200, 200, 200);
        let avg = average2(a, b);

        assert_eq!(argb_alpha(avg), 150);
        assert_eq!(argb_red(avg), 150);
        assert_eq!(argb_green(avg), 150);
        assert_eq!(argb_blue(avg), 150);
    }

    #[test]
    fn test_residual() {
        let pixel = make_argb(100, 50, 80, 200);
        let pred = make_argb(90, 60, 70, 150);
        let res = residual(pixel, pred);

        assert_eq!(argb_alpha(res), 10); // 100 - 90
        assert_eq!(argb_red(res), 246); // 50 - 60 = -10 = 246 (wrapping)
        assert_eq!(argb_green(res), 10); // 80 - 70
        assert_eq!(argb_blue(res), 50); // 200 - 150
    }

    #[test]
    fn test_color_index_small_palette() {
        let pixels = vec![
            make_argb(255, 255, 0, 0), // Red
            make_argb(255, 0, 255, 0), // Green
            make_argb(255, 255, 0, 0), // Red again
        ];

        let transform = ColorIndexTransform::try_build(&pixels).unwrap();
        assert_eq!(transform.palette.len(), 2);
        assert_eq!(transform.bits_per_pixel(), 3); // Can pack 8 per pixel
    }

    #[test]
    fn test_color_index_too_many_colors() {
        // Create 257 unique colors - use different channels to ensure uniqueness
        let pixels: Vec<u32> = (0..257)
            .map(|i| make_argb(255, (i % 256) as u8, (i / 256) as u8, 0))
            .collect();
        let transform = ColorIndexTransform::try_build(&pixels);
        assert!(transform.is_none());
    }
}
