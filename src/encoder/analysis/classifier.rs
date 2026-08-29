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

/// The bare feature names this classifier reads, in no particular order.
///
/// Named rather than version-pinned on purpose: the decision below is a set of
/// hand-tuned thresholds, and a re-definition of, say, `edge_slope_stdev` moves
/// the numbers a little but does not invalidate the rule — so
/// [`Select::Names`](zenanalyze_api::Select::Names) is the right selector, and it
/// is what lets this file name features **without naming a `zenanalyze`
/// version**. (A compiled model would need
/// [`Select::Features`](zenanalyze_api::Select::Features) instead, so a code
/// drift misses rather than silently feeding it drifted inputs.)
#[cfg(feature = "analyzer")]
pub const CLASSIFIER_FEATURES: [&str; 10] = [
    "flat_color_block_ratio",
    "distinct_color_bins",
    "variance",
    "edge_density",
    "uniformity",
    "high_freq_energy_ratio",
    "palette_fits_in_256",
    "palette_log2_size",
    "skin_tone_fraction",
    "edge_slope_stdev",
];

/// This classifier's ask, for an orchestrator unionizing several codecs'
/// requests before running one shared analysis pass.
#[cfg(feature = "analyzer")]
#[must_use]
pub fn classifier_request() -> zenanalyze_api::Request<'static> {
    zenanalyze_api::Request::new(zenanalyze_api::Select::Names(&CLASSIFIER_FEATURES))
}

/// Read the classifier's signals out of anything that can look a feature up by
/// bare name. Absent features keep their [`Default`] value, which is what the
/// pre-contract code did for the four likelihoods culled from zenanalyze.
#[cfg(feature = "analyzer")]
fn diag_from_lookup(get: impl Fn(&str) -> Option<zenanalyze_api::Value>) -> ZenanalyzeDiag {
    use zenanalyze_api::Value;
    let f32_of = |name: &str| get(name).map_or(0.0, Value::to_f32);
    let u32_of = |name: &str| match get(name) {
        Some(Value::U32(x)) => x,
        Some(v) => v.to_f32() as u32,
        None => 0,
    };
    ZenanalyzeDiag {
        // Culled from zenanalyze's post-cull schema (ids 27/28/29/45 stay
        // reserved); zero here, as before, so the thresholds that read them are
        // inert rather than wrong.
        screen_content: 0.0,
        text_likelihood: 0.0,
        natural_likelihood: 0.0,
        line_art_score: 0.0,
        flat_color_block_ratio: f32_of("flat_color_block_ratio"),
        distinct_color_bins: u32_of("distinct_color_bins"),
        variance: f32_of("variance"),
        edge_density: f32_of("edge_density"),
        uniformity: f32_of("uniformity"),
        high_freq_energy_ratio: f32_of("high_freq_energy_ratio"),
        palette_fits_in_256: matches!(get("palette_fits_in_256"), Some(Value::Bool(true))),
        // `ZenanalyzeDiag.indexed_palette_width` keeps the legacy field name for
        // dev tooling that reads its printouts; the value comes from the
        // wider-codomain `palette_log2_size` (codomain `{1..15, 24}`), which
        // replaced `indexed_palette_width` upstream.
        indexed_palette_width: u32_of("palette_log2_size"),
        skin_tone_fraction: f32_of("skin_tone_fraction"),
        edge_slope_stdev: f32_of("edge_slope_stdev"),
    }
}

/// Read the classifier's signals out of a borrowed [`Offer`](zenanalyze_api::Offer)
/// — the shared-pass path: an orchestrator runs ONE analysis for every codec and
/// lends the result here, so this classification costs no extra pixels.
#[cfg(feature = "analyzer")]
#[must_use]
pub fn diag_from_offer(offer: &zenanalyze_api::Offer<'_>) -> ZenanalyzeDiag {
    diag_from_lookup(|name| offer.get(name).map(zenanalyze_api::FeatureResult::value))
}

/// [`diag_from_offer`] for the owned twin (a deserialized offer — a parquet row,
/// a stored stamp).
#[cfg(feature = "analyzer")]
#[must_use]
pub fn diag_from_owned_offer(offer: &zenanalyze_api::OwnedOffer) -> ZenanalyzeDiag {
    diag_from_lookup(|name| {
        offer
            .get(name)
            .map(zenanalyze_api::OwnedFeatureResult::value)
    })
}

/// Classify content type from a shared [`Offer`](zenanalyze_api::Offer).
///
/// Thresholds are unchanged from the pre-contract classifier; only the source of
/// the values moved. `width`/`height` ≤ 128 still routes to `Icon` (preserving the
/// small-image carve-out) without consulting the offer.
///
/// Threshold rationale (ScreenContent ≥ 0.6, Text ≥ 0.5, FlatColorBlockRatio ≥
/// 0.20): starting points distilled from zenanalyze's documented behaviour (photos
/// cluster `ScreenContentLikelihood` below 0.05, screen content above 0.7; ROC-AUC
/// 0.978 at the default budget). Tune against the `auto_detection_tuning` corpus;
/// do not relax thresholds without confirming the test floors still hold.
#[cfg(feature = "analyzer")]
#[must_use]
pub fn classify_image_type_from_offer(
    offer: &zenanalyze_api::Offer<'_>,
    width: u32,
    height: u32,
) -> (ImageContentType, ZenanalyzeDiag) {
    if width <= 128 && height <= 128 {
        return (ImageContentType::Icon, ZenanalyzeDiag::default());
    }
    let diag = diag_from_offer(offer);
    (decide_bucket_from_diag(&diag), diag)
}

/// Classify content type by extracting through a
/// [`FeatureProvider`](zenanalyze_api::FeatureProvider) the caller supplies — the
/// no-shared-offer path, still without naming a `zenanalyze` type.
///
/// The host chooses the analyzer version by choosing which provider it passes;
/// `zenanalyze::Analyzer` (behind zenanalyze's `api` feature) is the usual one,
/// and the `analyzer-bundled` feature wires it up for you via
/// [`classify_image_type_rgb8`].
///
/// Falls back to `(Photo, default)` — never a panic — if the buffer is malformed
/// or the provider cannot produce the signals.
#[cfg(feature = "analyzer")]
#[must_use]
pub fn classify_image_type_with_provider(
    provider: &dyn zenanalyze_api::FeatureProvider,
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
    let Ok(offer) = provider.extract_rgb8(rgb, width, height, &classifier_request()) else {
        return (ImageContentType::Photo, ZenanalyzeDiag::default());
    };
    let diag = diag_from_owned_offer(&offer);
    (decide_bucket_from_diag(&diag), diag)
}

/// The bundled default provider: `zenanalyze::Analyzer` for the `zenanalyze`
/// version this build pinned.
///
/// **This function is the only place in zenwebp that names a `zenanalyze` type**,
/// and it exists to play the host role (choosing a version) for callers that
/// don't want to supply one — see `docs/sole-contract.md` in imazen/zenanalyze.
/// Everything above it works against `zenanalyze-api` alone.
#[cfg(feature = "analyzer-bundled")]
#[must_use]
pub fn bundled_provider() -> impl zenanalyze_api::FeatureProvider {
    zenanalyze::Analyzer::new()
}

/// Classify image content type using the [`bundled_provider`].
///
/// One analysis pass over the RGB8 source extracts the palette / flat-colour /
/// edge signals that distinguish "screenshot or UI graphic" from "natural
/// photograph", so the same thresholds drive zenwebp / zenjpeg / zenavif preset
/// selection from a single shared signal source.
///
/// Prefer [`classify_image_type_from_offer`] when an orchestrator already ran a
/// pass — this entry point runs its own.
#[cfg(feature = "analyzer-bundled")]
#[must_use]
pub fn classify_image_type_rgb8(rgb: &[u8], width: u32, height: u32) -> ImageContentType {
    classify_image_type_rgb8_diag(rgb, width, height).0
}

/// Diagnostic variant of [`classify_image_type_rgb8`] returning the raw signals
/// alongside the bucket decision. Used by the classifier-comparison harness in
/// `dev/`.
#[cfg(feature = "analyzer-bundled")]
#[must_use]
pub fn classify_image_type_rgb8_diag(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> (ImageContentType, ZenanalyzeDiag) {
    classify_image_type_with_provider(&bundled_provider(), rgb, width, height)
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
    for px in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&px[..3]);
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

/// Tests for the contract-only path — no `zenanalyze` in the build.
///
/// This is the regression gate for the failure that motivated the migration: the
/// classifier used to name upstream enum variants directly, so a rename
/// (`IndexedPaletteWidth` → `PaletteLog2Size`) stopped the crate compiling. The
/// names now live in `CLASSIFIER_FEATURES` and are read out of an offer, and a
/// hand-built offer is the only way to prove that every one of them still reaches
/// the field it feeds. Building the offer by hand (rather than through a
/// provider) is the point: it needs no analyzer at all, so this runs in exactly
/// the configuration a consumer ships.
#[cfg(all(test, feature = "analyzer"))]
mod contract_tests {
    use super::*;
    use zenanalyze_api::{FeatureResult, NamedFeature, Offer, OwnedFeatureResult, Provenance};

    /// Cells for `(name, value)` pairs, qualified with an arbitrary but valid
    /// code version — the classifier matches by BARE name, so the hex is free.
    fn cells(pairs: &[(&str, zenanalyze_api::Value)]) -> Vec<OwnedFeatureResult> {
        pairs
            .iter()
            .map(|(name, value)| {
                let qualified = NamedFeature::qualified_for(name, 0x1234_5678);
                OwnedFeatureResult::new(&qualified, *value)
            })
            .collect()
    }

    fn offer_of<'a>(cells: &'a [OwnedFeatureResult]) -> Vec<FeatureResult<'a>> {
        cells.iter().map(OwnedFeatureResult::as_ref).collect()
    }

    /// Every name in `CLASSIFIER_FEATURES` lands in the `ZenanalyzeDiag` field it
    /// feeds, with its native type preserved. A typo or a stale name would leave
    /// the corresponding field at its default and this catches it — the check the
    /// pre-contract code got from the compiler, and which by-name lookup gives up.
    #[test]
    fn every_classifier_feature_reaches_its_diag_field() {
        use zenanalyze_api::Value;
        let owned = cells(&[
            ("flat_color_block_ratio", Value::F32(0.11)),
            ("distinct_color_bins", Value::U32(1234)),
            ("variance", Value::F32(22.5)),
            ("edge_density", Value::F32(0.33)),
            ("uniformity", Value::F32(0.44)),
            ("high_freq_energy_ratio", Value::F32(0.55)),
            ("palette_fits_in_256", Value::Bool(true)),
            ("palette_log2_size", Value::U32(7)),
            ("skin_tone_fraction", Value::F32(0.66)),
            ("edge_slope_stdev", Value::F32(77.0)),
        ]);
        let borrowed = offer_of(&owned);
        let diag = diag_from_offer(&Offer::new(&borrowed, Provenance::new("test")));

        assert_eq!(diag.flat_color_block_ratio, 0.11);
        assert_eq!(diag.distinct_color_bins, 1234, "u32 must survive natively");
        assert_eq!(diag.variance, 22.5);
        assert_eq!(diag.edge_density, 0.33);
        assert_eq!(diag.uniformity, 0.44);
        assert_eq!(diag.high_freq_energy_ratio, 0.55);
        assert!(diag.palette_fits_in_256, "bool must survive natively");
        assert_eq!(
            diag.indexed_palette_width, 7,
            "the legacy field reads palette_log2_size, the name that replaced \
             IndexedPaletteWidth upstream"
        );
        assert_eq!(diag.skin_tone_fraction, 0.66);
        assert_eq!(diag.edge_slope_stdev, 77.0);

        // The four likelihoods were culled upstream; they must read as 0.0, not
        // as garbage, because `decide_bucket_from_diag` still tests them.
        assert_eq!(diag.screen_content, 0.0);
        assert_eq!(diag.text_likelihood, 0.0);
        assert_eq!(diag.natural_likelihood, 0.0);
        assert_eq!(diag.line_art_score, 0.0);

        // And the ask names exactly these ten, so an orchestrator's shared pass
        // covers the rule.
        assert_eq!(CLASSIFIER_FEATURES.len(), 10);
        for name in CLASSIFIER_FEATURES {
            assert!(
                owned.iter().any(|c| c.name() == name),
                "{name} is requested but was not exercised above"
            );
        }
    }

    /// A feature the offer doesn't carry keeps its default rather than reading a
    /// neighbour's value — the property that lets the rule survive an upstream
    /// cull instead of failing to build.
    #[test]
    fn a_missing_feature_defaults_instead_of_shifting_the_others() {
        use zenanalyze_api::Value;
        let owned = cells(&[("variance", Value::F32(9.0))]);
        let borrowed = offer_of(&owned);
        let diag = diag_from_offer(&Offer::new(&borrowed, Provenance::new("test")));
        assert_eq!(diag.variance, 9.0);
        assert_eq!(diag.edge_density, 0.0);
        assert_eq!(diag.distinct_color_bins, 0);
        assert!(!diag.palette_fits_in_256);
    }

    /// The offer entry reproduces the documented decisions: the small-image
    /// carve-out short-circuits to `Icon` without consulting the offer, and a
    /// screen-content-shaped diag routes to `Drawing` through the flat-block rule.
    #[test]
    fn offer_entry_preserves_the_icon_carve_out_and_the_drawing_rule() {
        use zenanalyze_api::Value;
        let owned = cells(&[
            ("flat_color_block_ratio", Value::F32(0.90)),
            ("distinct_color_bins", Value::U32(64)),
        ]);
        let borrowed = offer_of(&owned);
        let offer = Offer::new(&borrowed, Provenance::new("test"));

        // <= 128 in BOTH dims: Icon, and the offer is not read.
        assert_eq!(
            classify_image_type_from_offer(&offer, 128, 128).0,
            ImageContentType::Icon
        );
        // Above the carve-out: flat >= 0.50 and distinct < 4096 => Drawing.
        assert_eq!(
            classify_image_type_from_offer(&offer, 512, 512).0,
            ImageContentType::Drawing
        );
    }
}
