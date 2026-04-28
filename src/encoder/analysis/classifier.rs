//! Content type classification for auto-preset selection.
//!
//! Analyzes the Y plane and alpha histogram to detect content type
//! (photo, drawing, text, icon) and select appropriate encoding parameters.
//!
//! ## SIMD Optimizations
//!
//! - `compute_edge_density`: SIMD horizontal abs_diff scan

#![allow(dead_code)]

use archmage::prelude::*;

#[cfg(target_arch = "aarch64")]
use archmage::intrinsics::aarch64 as simd_mem;
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem;

/// Detected content type for auto-preset selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageContentType {
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
    pub content_type: ImageContentType,
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
) -> ImageContentType {
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
            content_type: ImageContentType::Icon,
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
            content_type: ImageContentType::Photo,
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
        ImageContentType::Photo
    } else {
        ImageContentType::Drawing // "complex content" — uses Default tuning values
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
    incant!(
        compute_edge_density_impl(y_src, width, height, y_stride),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn compute_edge_density_impl_v3(
    token: X64V3Token,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    compute_edge_density_sse2(token, y_src, width, height, y_stride)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn compute_edge_density_impl_neon(
    token: NeonToken,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    compute_edge_density_neon(token, y_src, width, height, y_stride)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn compute_edge_density_impl_wasm128(
    _token: Wasm128Token,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    compute_edge_density_scalar(y_src, width, height, y_stride)
}

#[inline(always)]
fn compute_edge_density_impl_scalar(
    _token: ScalarToken,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    compute_edge_density_scalar(y_src, width, height, y_stride)
}

/// Scalar implementation of edge density computation.
fn compute_edge_density_scalar(y_src: &[u8], width: usize, height: usize, y_stride: usize) -> f32 {
    if width < 2 || height < 16 {
        return 0.0;
    }

    let mut edge_count = 0u32;
    let mut sample_count = 0u32;
    let threshold = 32u8;

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

// compute_edge_density_dispatch removed — replaced by incant! in compute_edge_density

/// SSE2 edge density: Process 16 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn compute_edge_density_sse2(
    _token: X64V3Token,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    if width < 2 || height < 16 {
        return 0.0;
    }

    let mut edge_count = 0u32;
    let mut sample_count = 0u32;
    let threshold_vec = _mm_set1_epi8(32i8);

    let mut y = 0;
    while y < height {
        let row = &y_src[y * y_stride..];

        // Process 16 pixels at a time (comparing pixels x and x-1)
        let mut x = 1usize;
        while x + 15 < width {
            // Load pixels at positions [x, x+1, ..., x+15] and [x-1, x, ..., x+14]
            let curr_arr = <&[u8; 16]>::try_from(&row[x..x + 16]).unwrap();
            let prev_arr = <&[u8; 16]>::try_from(&row[x - 1..x + 15]).unwrap();
            let curr = simd_mem::_mm_loadu_si128(curr_arr);
            let prev = simd_mem::_mm_loadu_si128(prev_arr);

            // Compute |curr - prev| using saturating sub both ways
            let diff1 = _mm_subs_epu8(curr, prev);
            let diff2 = _mm_subs_epu8(prev, curr);
            let abs_diff = _mm_or_si128(diff1, diff2);

            // Compare: abs_diff > threshold
            // Subtract (threshold+1) and check for non-zero (if >= 33, result is non-zero)
            let above_thresh = _mm_subs_epu8(abs_diff, threshold_vec);
            // Convert to 0xFF where above threshold
            let zero = _mm_setzero_si128();
            let mask = _mm_cmpeq_epi8(above_thresh, zero);
            // Invert: we want 0xFF where above threshold (mask is 0xFF where NOT above)
            let edges = _mm_andnot_si128(mask, _mm_set1_epi8(-1i8));

            // Count set bytes (each edge pixel has 0xFF)
            let mask_bits = _mm_movemask_epi8(edges) as u32;
            edge_count += mask_bits.count_ones();
            sample_count += 16;

            x += 16;
        }

        // Handle remaining pixels with scalar
        while x < width {
            let diff = row[x].abs_diff(row[x - 1]);
            if diff > 32 {
                edge_count += 1;
            }
            sample_count += 1;
            x += 1;
        }

        y += 16;
    }

    if sample_count == 0 {
        return 0.0;
    }
    edge_count as f32 / sample_count as f32
}

// =============================================================================
// NEON (aarch64) edge density
// =============================================================================

// compute_edge_density_neon_dispatch removed — replaced by incant! in compute_edge_density

/// NEON edge density: Process 16 pixels at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn compute_edge_density_neon(
    _token: NeonToken,
    y_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
) -> f32 {
    if width < 2 || height < 16 {
        return 0.0;
    }

    let mut edge_count = 0u32;
    let mut sample_count = 0u32;
    let threshold_vec = vdupq_n_u8(32);

    let mut y = 0;
    while y < height {
        let row = &y_src[y * y_stride..];

        // Process 16 pixels at a time (comparing pixels x and x-1)
        let mut x = 1usize;
        while x + 15 < width {
            // Load pixels at positions [x, x+15] and [x-1, x+14]
            let curr = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&row[x..x + 16]).unwrap());
            let prev = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&row[x - 1..x + 15]).unwrap());

            // Compute |curr - prev| using absolute difference
            let abs_diff = vabdq_u8(curr, prev);

            // Compare: abs_diff > threshold
            // vcgtq_u8 returns 0xFF for lanes where abs_diff > threshold
            let above_thresh = vcgtq_u8(abs_diff, threshold_vec);

            // Count set bytes: each edge pixel has 0xFF, AND with 1 gives 0 or 1
            // Use horizontal add after masking with 1
            let ones = vandq_u8(above_thresh, vdupq_n_u8(1));

            // Sum all the 1s: vaddlvq_u8 sums all u8 lanes into a u16
            edge_count += vaddlvq_u8(ones) as u32;
            sample_count += 16;

            x += 16;
        }

        // Handle remaining pixels with scalar
        while x < width {
            let diff = row[x].abs_diff(row[x - 1]);
            if diff > 32 {
                edge_count += 1;
            }
            sample_count += 1;
            x += 1;
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

/// Classify image content type via the `zenanalyze` shared scanner.
///
/// One streaming pass over the RGB(A)8 source extracts the soft
/// content-class likelihoods (`ScreenContentLikelihood`,
/// `TextLikelihood`, `NaturalLikelihood`) plus the cheap palette /
/// flat-colour signals that distinguish "screenshot or UI graphic"
/// from "natural photograph". This replaces the homegrown
/// `classify_image_type` heuristic (alpha histogram + Y-plane edge /
/// uniformity scan) with a single shared signal source, so the same
/// thresholds drive zenwebp / zenjpeg / zenavif preset selection.
///
/// Threshold rationale (ScreenContent ≥ 0.6, Text ≥ 0.5,
/// FlatColorBlockRatio ≥ 0.20): these are starting points distilled
/// from zenanalyze's documented behaviour (photos cluster
/// `ScreenContentLikelihood` below 0.05, screen content above 0.7;
/// ROC-AUC 0.978 at the default budget). Tune against the
/// `auto_detection_tuning` corpus; do not relax thresholds without
/// confirming the test floors still hold.
///
/// `width` and `height` ≤ 128 still routes to `Icon` (preserves
/// the existing small-image carve-out).
#[cfg(feature = "analyzer")]
pub fn classify_image_type_rgb8(rgb: &[u8], width: u32, height: u32) -> ImageContentType {
    classify_image_type_rgb8_diag(rgb, width, height).0
}

/// Diagnostic variant of [`classify_image_type_rgb8`] returning the
/// raw zenanalyze signals alongside the bucket decision. Used by the
/// classifier-comparison harness in `dev/`.
#[cfg(feature = "analyzer")]
pub fn classify_image_type_rgb8_diag(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> (ImageContentType, ZenanalyzeDiag) {
    if width <= 128 && height <= 128 {
        return (ImageContentType::Icon, ZenanalyzeDiag::default());
    }
    if rgb.len() != (width as usize) * (height as usize) * 3 {
        return (ImageContentType::Photo, ZenanalyzeDiag::default());
    }
    use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
    const FEATURES: FeatureSet = FeatureSet::new()
        .with(AnalysisFeature::ScreenContentLikelihood)
        .with(AnalysisFeature::TextLikelihood)
        .with(AnalysisFeature::NaturalLikelihood)
        .with(AnalysisFeature::FlatColorBlockRatio)
        .with(AnalysisFeature::DistinctColorBins)
        .with(AnalysisFeature::Variance)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::Uniformity)
        .with(AnalysisFeature::HighFreqEnergyRatio)
        // Experimental signals (gated on zenanalyze's `experimental`
        // feature). PaletteFitsIn256 / IndexedPaletteWidth catch
        // graphics with a small palette; LineArtScore catches line
        // drawings / engineering diagrams that don't trigger
        // ScreenContentLikelihood (which is palette+HF driven).
        .with(AnalysisFeature::PaletteFitsIn256)
        .with(AnalysisFeature::IndexedPaletteWidth)
        .with(AnalysisFeature::LineArtScore)
        // Physics-based photo-vs-artwork discriminators shipped in
        // zenanalyze 0.1.0 per zenjpeg#123. SkinToneFraction is a
        // "presence of human content" cue (LAB-space skin-region
        // pixel fraction); EdgeSlopeStdev measures the spread of
        // luma gradient magnitudes across the edge subset and
        // separates photographic anti-aliased edges (tight stddev
        // around the lens MTF cutoff, ~15–32) from screen / chart
        // content (>35) and from smooth illustrations / line art
        // (<15).
        .with(AnalysisFeature::SkinToneFraction)
        .with(AnalysisFeature::EdgeSlopeStdev);
    let q = AnalysisQuery::new(FEATURES);
    let r = match zenanalyze::try_analyze_features_rgb8(rgb, width, height, &q) {
        Ok(r) => r,
        Err(_) => return (ImageContentType::Photo, ZenanalyzeDiag::default()),
    };
    let diag = ZenanalyzeDiag {
        screen_content: r
            .get_f32(AnalysisFeature::ScreenContentLikelihood)
            .unwrap_or(0.0),
        text_likelihood: r.get_f32(AnalysisFeature::TextLikelihood).unwrap_or(0.0),
        natural_likelihood: r.get_f32(AnalysisFeature::NaturalLikelihood).unwrap_or(0.0),
        flat_color_block_ratio: r
            .get_f32(AnalysisFeature::FlatColorBlockRatio)
            .unwrap_or(0.0),
        distinct_color_bins: r
            .get(AnalysisFeature::DistinctColorBins)
            .and_then(|v| v.as_u32())
            .unwrap_or(0),
        variance: r.get_f32(AnalysisFeature::Variance).unwrap_or(0.0),
        edge_density: r.get_f32(AnalysisFeature::EdgeDensity).unwrap_or(0.0),
        uniformity: r.get_f32(AnalysisFeature::Uniformity).unwrap_or(0.0),
        high_freq_energy_ratio: r
            .get_f32(AnalysisFeature::HighFreqEnergyRatio)
            .unwrap_or(0.0),
        palette_fits_in_256: r
            .get(AnalysisFeature::PaletteFitsIn256)
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        indexed_palette_width: r
            .get(AnalysisFeature::IndexedPaletteWidth)
            .and_then(|v| v.as_u32())
            .unwrap_or(0),
        line_art_score: r.get_f32(AnalysisFeature::LineArtScore).unwrap_or(0.0),
        skin_tone_fraction: r.get_f32(AnalysisFeature::SkinToneFraction).unwrap_or(0.0),
        edge_slope_stdev: r.get_f32(AnalysisFeature::EdgeSlopeStdev).unwrap_or(0.0),
    };
    let bucket = decide_bucket_from_diag(&diag);
    (bucket, diag)
}

/// Threshold-only decision over the zenanalyze signals using only
/// the *stable* (non-experimental) features. Used as the "default
/// signals" tier in the validation harness so we can isolate the
/// improvement from `palette_fits_in_256` / `line_art_score`.
#[cfg(feature = "analyzer")]
pub fn decide_bucket_stable(diag: &ZenanalyzeDiag) -> ImageContentType {
    if diag.screen_content > 0.6 || diag.text_likelihood > 0.5 {
        return ImageContentType::Drawing;
    }
    if diag.flat_color_block_ratio > 0.20 && diag.distinct_color_bins < 4096 {
        return ImageContentType::Drawing;
    }
    ImageContentType::Photo
}

/// Threshold-only decision over the zenanalyze signals. Pulled out
/// so the validation harness in `dev/zenanalyze_validate_vs_gpt.rs`
/// can replay the decision against pre-recorded signals when tuning.
///
/// Tuned against 219 GPT-5.4-mini-labelled images from the
/// classifier-eval corpus (cid22-train/val, clic2025-1024, gb82,
/// gb82-sc, kadid10k, qoi-benchmark). With `SkinToneFraction` /
/// `EdgeSlopeStdev` (zenanalyze 0.1.0) wired in as a portrait-
/// rescue rule: **93.4%** overall, photo recall **96.9%**, drawing
/// recall **78.4%** (n=198, 21 rows skipped — JPGs and missing
/// files). Up from 92.9% / 96.3% / 78.4% pre-rescue.
///
/// Order of tests matters:
///
/// 0. **Photo rescue (new):** `skin_tone_fraction >= 0.15` AND
///    `edge_slope_stdev < 35.0` → Photo. Catches portraits whose
///    smooth backgrounds confused `screen_content_likelihood` /
///    `flat_color_block_ratio`. Photographic edge stddev (lens-MTF
///    cluster ~15–32) plus visible skin is a strong "natural
///    photo" pair. Rescues `kadid10k/I29.png` (photo_portrait at
///    `skin=0.239, slpSD=16.97, screen=0.61`); does not rescue any
///    actual drawings in the corpus.
/// 1. `line_art_score > 0.5` → Drawing (engineering / line art)
/// 2. `screen_content >= 0.60` or `text_likelihood >= 0.55` →
///    Drawing (qoi-benchmark websites clamp at exactly 0.6000)
/// 3. `screen >= 0.40` AND `flat >= 0.40` AND `uniformity >= 0.85`
///    AND `distinct < 4096` → Drawing (anti-aliased UI fallback)
/// 4. `flat >= 0.50` AND `distinct < 4096` → Drawing (charts / UI
///    overflow)
/// 5. `palette_fits_in_256` AND `natural < 0.10` AND
///    `screen >= 0.50` → Drawing (tiny-palette photo edge case)
///
/// **Why the new features alone don't rescue more drawing FNs:**
/// the 8 remaining drawing→photo errors are paintings and
/// illustrations whose `skin_tone_fraction` and `edge_slope_stdev`
/// fall inside the photographic ranges (skin ≤ 0.42, slpSD 4–28).
/// With only these two physics-based signals, the corpus-wide
/// AUC for "artwork vs natural" stays around 0.80; the noise-
/// spectrum / JPEG-roundtrip signals proposed in zenjpeg#123 are
/// the next discriminator and aren't in 0.1.0.
#[cfg(feature = "analyzer")]
pub fn decide_bucket_from_diag(diag: &ZenanalyzeDiag) -> ImageContentType {
    // Photo rescue: meaningful skin-tone fraction and a
    // photographic edge-stddev cluster. Runs before any drawing
    // rule so portraits with smooth studio backgrounds aren't
    // dragged into Drawing by `screen_content` / `flat`.
    if diag.skin_tone_fraction >= 0.15 && diag.edge_slope_stdev < 35.0 {
        return ImageContentType::Photo;
    }
    // Strong drawing signal: line-art / engineering-drawing score.
    if diag.line_art_score > 0.5 {
        return ImageContentType::Drawing;
    }
    // Screen-content / text — `>=` so qoi-benchmark websites at
    // exactly 0.6000 are caught.
    if diag.screen_content >= 0.60 || diag.text_likelihood >= 0.55 {
        return ImageContentType::Drawing;
    }
    // Combined screen+flat+uniform signal: catches anti-aliased UI
    // pages where the screen-content score sits at 0.4-0.6 but the
    // page is dominated by uniform flat blocks.
    if diag.screen_content >= 0.40
        && diag.flat_color_block_ratio >= 0.40
        && diag.uniformity >= 0.85
        && diag.distinct_color_bins < 4096
    {
        return ImageContentType::Drawing;
    }
    // Flat-block fallback (tightened from the original 0.20 bound):
    // real UI / chart content sits at flat >= 0.50; smooth photos
    // cap below that.
    if diag.flat_color_block_ratio >= 0.50 && diag.distinct_color_bins < 4096 {
        return ImageContentType::Drawing;
    }
    // Fits-in-256-colours is a strong indicator only when paired
    // with low natural likelihood AND a meaningful screen score
    // (rules out flat photos / night scenes with tiny palettes that
    // GPT still labels as "photo").
    if diag.palette_fits_in_256 && diag.natural_likelihood < 0.10 && diag.screen_content >= 0.50 {
        return ImageContentType::Drawing;
    }
    ImageContentType::Photo
}

/// Streaming-analyzer signals for diagnostic and calibration use.
///
/// All fields are zenanalyze stable (non-experimental) features so the
/// numeric scale is governed by the crate's threshold contract.
#[cfg(feature = "analyzer")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ZenanalyzeDiag {
    /// `[0, 1]` soft score: UI / chart / synthetic content.
    pub screen_content: f32,
    /// `[0, 1]` soft score: rendered text / document content.
    pub text_likelihood: f32,
    /// `[0, 1]` soft score: natural photographic content.
    pub natural_likelihood: f32,
    /// Fraction of 8×8 blocks with R/G/B ranges all ≤ 4.
    pub flat_color_block_ratio: f32,
    /// Distinct 5-bit-per-channel RGB bins observed.
    pub distinct_color_bins: u32,
    /// Luma variance on BT.601 [0, 255] scale.
    pub variance: f32,
    /// Fraction of sampled interior pixels with `|∇L| > 20`.
    pub edge_density: f32,
    /// Fraction of 8×8 blocks with luma variance < 25.
    pub uniformity: f32,
    /// `Σ AC[k≥16] / Σ AC[k∈1..16]` over sampled luma blocks.
    pub high_freq_energy_ratio: f32,
    /// `true` iff the source RGB fits in a 256-colour palette (no
    /// quantization required). Experimental signal — strong "graphics
    /// with limited palette" indicator.
    pub palette_fits_in_256: bool,
    /// Indexed palette width estimate. `0` if more than 256 colours.
    /// Experimental.
    pub indexed_palette_width: u32,
    /// `[0, 1]` line-art / engineering-drawing score from Otsu
    /// bimodality + low-entropy gate. Experimental.
    pub line_art_score: f32,
    /// Fraction of pixels whose RGB falls inside a canonical LAB
    /// skin-tone region (Chai & Ngan / Vezhnevets). Tier 1 streaming.
    /// One-direction signal: non-zero → likely natural photo, zero →
    /// ambiguous (could be landscape / artwork / nature). Experimental.
    ///
    /// Empirical p50s (per `AnalysisFeature::SkinToneFraction` docs):
    /// `photo_portrait` 0.21, `photo_natural` 0.04, `illustration`
    /// 0.08, `screen_*` ≤ 0.03.
    pub skin_tone_fraction: f32,
    /// Standard deviation of luma gradient magnitudes across pixels
    /// crossing the `EdgeDensity` threshold (`|∇L| > 20` on 0–255).
    /// Tier 1 — accumulated piggyback on the same SIMD edge sweep.
    /// Experimental.
    ///
    /// Empirical p50s: `photo_*` 20–24, `illustration` ~21,
    /// `screen_document` ~55, `screen_ui` ~42. So **high** (> ~32)
    /// reads as screen content; **low–mid** (15–32) reads as
    /// photographic; very low (<15) reads as smooth content
    /// (illustrations or low-detail photos overlap here).
    pub edge_slope_stdev: f32,
}

/// Convert RGBA8 to RGB8 (drops the alpha channel) for the classifier.
/// `analyze_features` could ingest RGBA8 directly via PixelSlice; this
/// helper exists because the classifier entry deliberately stays
/// rgb8-only to keep the API surface small.
#[cfg(feature = "analyzer")]
pub fn rgba8_to_rgb8(rgba: &[u8]) -> alloc::vec::Vec<u8> {
    use alloc::vec::Vec;
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        out.push(px[0]);
        out.push(px[1]);
        out.push(px[2]);
    }
    out
}

/// Get tuning parameters for a detected content type.
/// Returns (sns_strength, filter_strength, filter_sharpness, num_segments).
pub fn content_type_to_tuning(content_type: ImageContentType) -> (u8, u8, u8, u8) {
    match content_type {
        ImageContentType::Photo => (80, 30, 3, 4), // Photo preset: high SNS for uniform regions
        ImageContentType::Drawing => (50, 60, 0, 4), // Default tuning: moderate SNS, strong filter
        ImageContentType::Text => (50, 60, 0, 4), // Default tuning (Text preset was counterproductive)
        ImageContentType::Icon => (0, 0, 0, 4),   // Icon preset: no SNS, no filter
    }
}
