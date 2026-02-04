//! Content type classification for auto-preset selection.
//!
//! Analyzes the Y plane and alpha histogram to detect content type
//! (photo, drawing, text, icon) and select appropriate encoding parameters.
//!
//! ## SIMD Optimization Opportunities
//!
//! - `compute_edge_density`: Horizontal abs_diff scan
//! - `compute_color_uniformity`: Block-wise distinct value counting

#![allow(dead_code)]

/// Detected content type for auto-preset selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentType {
    /// Natural photograph or complex texture.
    Photo,
    /// Hand or line drawing, screenshot, UI.
    Drawing,
    /// Text-heavy content.
    Text,
    /// Small icon or sprite.
    Icon,
}

/// Diagnostic info from the classifier.
#[derive(Debug, Clone, Copy)]
pub struct ClassifierDiag {
    /// Detected content type.
    pub content_type: ContentType,
    /// Fraction of alpha histogram in low quarter (0-63).
    pub low_frac: f32,
    /// Fraction of alpha histogram in high quarter (192-255).
    pub high_frac: f32,
    /// Whether the alpha histogram is bimodal.
    pub is_bimodal: bool,
    /// Fraction of sampled pixels with sharp horizontal transitions.
    pub edge_density: f32,
    /// Fraction of sampled blocks with few distinct Y values.
    pub uniformity: f32,
}

/// Classify image content type from Y plane and alpha histogram.
///
/// This runs after `analyze_image()` and uses the alpha histogram (nearly free)
/// plus a lightweight scan of the Y plane to determine content type.
///
/// Heuristics:
/// 1. Small images (≤128x128) → Icon
/// 2. Bimodal alpha histogram + high edge density + uniform blocks → Text
/// 3. Bimodal alpha histogram + uniform blocks → Drawing (screenshots, UI)
/// 4. Otherwise → Photo
pub fn classify_image_type(
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    alpha_histogram: &[u32; 256],
) -> ContentType {
    classify_image_type_diag(y_src, width, height, y_stride, alpha_histogram).content_type
}

/// Classify with full diagnostic output.
pub fn classify_image_type_diag(
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    alpha_histogram: &[u32; 256],
) -> ClassifierDiag {
    // 1. Small images → Icon
    if width <= 128 && height <= 128 {
        return ClassifierDiag {
            content_type: ContentType::Icon,
            low_frac: 0.0,
            high_frac: 0.0,
            is_bimodal: false,
            edge_density: 0.0,
            uniformity: 0.0,
        };
    }

    // Compute alpha histogram shape
    let total: u32 = alpha_histogram.iter().sum();
    if total == 0 {
        return ClassifierDiag {
            content_type: ContentType::Photo,
            low_frac: 0.0,
            high_frac: 0.0,
            is_bimodal: false,
            edge_density: 0.0,
            uniformity: 0.0,
        };
    }

    // Check if histogram is bimodal: significant mass at both ends
    // Low alpha = flat/simple regions, high alpha = textured regions
    let low_quarter: u32 = alpha_histogram[..64].iter().sum();
    let high_quarter: u32 = alpha_histogram[192..].iter().sum();
    let low_frac = low_quarter as f32 / total as f32;
    let high_frac = high_quarter as f32 / total as f32;
    let is_bimodal = low_frac > 0.15 && high_frac > 0.15;

    // 2. Compute edge density from Y plane
    // Sample every 16th row, count sharp horizontal transitions
    let edge_density = compute_edge_density(y_src, width, height, y_stride);

    // 3. Compute color uniformity: count distinct Y values in sampled blocks
    let uniformity = compute_color_uniformity(y_src, width, height, y_stride);

    // Classification logic: uniformity-based approach.
    // High uniformity (many flat blocks) → Photo tuning (SNS=80, lighter filter)
    // Low uniformity (complex textures) → Default tuning (SNS=50, stronger filter)
    //
    // Empirically, Drawing/Text presets produce larger files than Default on all
    // tested corpora (CID22, gb82-sc screenshots). Photo preset benefits images
    // with large uniform regions (screenshots, graphics, and clean photos).
    let content_type = if uniformity >= 0.45 {
        ContentType::Photo
    } else {
        ContentType::Drawing // "complex content" — uses Default tuning values
    };

    ClassifierDiag {
        content_type,
        low_frac,
        high_frac,
        is_bimodal,
        edge_density,
        uniformity,
    }
}

/// Compute edge density by scanning the Y plane for sharp horizontal transitions.
/// Returns fraction of sampled pixels that are sharp edges (0.0 to 1.0).
fn compute_edge_density(y_src: &[u8], width: usize, height: usize, y_stride: usize) -> f32 {
    if width < 2 || height < 16 {
        return 0.0;
    }

    let mut edge_count = 0u32;
    let mut sample_count = 0u32;
    let threshold = 32u8; // Sharp edge threshold

    // Sample every 16th row
    let mut y = 0;
    while y < height {
        let row = &y_src[y * y_stride..][..width];
        for x in 1..width {
            let diff = row[x].abs_diff(row[x - 1]);
            if diff > threshold {
                edge_count += 1;
            }
            sample_count += 1;
        }
        y += 16;
    }

    if sample_count == 0 {
        return 0.0;
    }
    edge_count as f32 / sample_count as f32
}

/// Compute color uniformity by sampling 16x16 blocks and measuring Y value spread.
/// Returns fraction of blocks that are "uniform" (low Y variance), 0.0 to 1.0.
fn compute_color_uniformity(y_src: &[u8], width: usize, height: usize, y_stride: usize) -> f32 {
    let mb_w = width / 16;
    let mb_h = height / 16;
    if mb_w == 0 || mb_h == 0 {
        return 0.0;
    }

    let mut uniform_count = 0u32;
    let mut total_blocks = 0u32;

    // Sample every 4th macroblock in both dimensions
    let mut mby = 0;
    while mby < mb_h {
        let mut mbx = 0;
        while mbx < mb_w {
            // Count distinct Y values in this 16x16 block
            let mut seen = [false; 256];
            let mut distinct = 0u32;
            for dy in 0..16 {
                let row_y = mby * 16 + dy;
                if row_y >= height {
                    break;
                }
                let row = &y_src[row_y * y_stride..];
                for dx in 0..16 {
                    let col_x = mbx * 16 + dx;
                    if col_x >= width {
                        break;
                    }
                    let val = row[col_x] as usize;
                    if !seen[val] {
                        seen[val] = true;
                        distinct += 1;
                    }
                }
            }

            // A block with few distinct values is "uniform"
            // Screenshots/drawings typically have <32 distinct values per block
            if distinct <= 32 {
                uniform_count += 1;
            }
            total_blocks += 1;

            mbx += 4;
        }
        mby += 4;
    }

    if total_blocks == 0 {
        return 0.0;
    }
    uniform_count as f32 / total_blocks as f32
}

/// Get tuning parameters for a detected content type.
/// Returns (sns_strength, filter_strength, filter_sharpness, num_segments).
pub fn content_type_to_tuning(content_type: ContentType) -> (u8, u8, u8, u8) {
    match content_type {
        ContentType::Photo => (80, 30, 3, 4), // Photo preset: high SNS for uniform regions
        ContentType::Drawing => (50, 60, 0, 4), // Default tuning: moderate SNS, strong filter
        ContentType::Text => (50, 60, 0, 4),  // Default tuning (Text preset was counterproductive)
        ContentType::Icon => (0, 0, 0, 4),    // Icon preset: no SNS, no filter
    }
}
