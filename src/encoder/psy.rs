//! Perceptual distortion model for encoder-side optimizations.
//!
//! This module provides psychovisual enhancements for WebP encoding:
//!
//! - **CSF (Contrast Sensitivity Function)** tables that weight distortion by
//!   frequency, de-emphasizing errors in high frequencies where human vision
//!   is less sensitive
//! - **JND (Just Noticeable Difference)** thresholds that identify coefficients
//!   too small to be perceived, allowing more aggressive quantization
//! - **Masking models** that detect texture complexity and adjust quantization —
//!   textured areas can hide more noise than flat/edge regions
//! - **SATD functions** for measuring signal energy in the Hadamard domain
//!
//! All features are gated by `method` level — methods 0-2 produce bit-identical
//! output to the baseline encoder.
//!
//! ## Relationship to other modules
//!
//! - [`super::cost`]: Uses `TDisto` (frequency-weighted Hadamard difference)
//!   for mode selection distortion. This module's CSF tables affect TDisto weights.
//! - [`super::trellis`]: Uses `PsyConfig` for JND-gated coefficient retention
//!   during trellis quantization.
//! - [`super::analysis`]: Calls `compute_masking_alpha` and `blend_masking_alpha`
//!   for adaptive quantization based on local texture complexity.

use super::tables::{VP8_WEIGHT_TRELLIS, VP8_WEIGHT_Y};

/// Enhanced luma CSF weights with steeper HF rolloff than VP8_WEIGHT_Y.
///
/// Human vision is less sensitive to high-frequency detail, so we can
/// de-emphasize HF distortion more aggressively than libwebp's defaults.
/// This lets the encoder spend fewer bits on HF detail that viewers
/// won't notice, improving rate-quality trade-off.
#[rustfmt::skip]
const PSY_WEIGHT_Y: [u16; 16] = [
    48, 36, 18,  6,
    36, 30, 14,  5,
    18, 14,  8,  3,
     6,  5,  3,  1,
];

/// Enhanced chroma CSF weights with faster spatial frequency rolloff.
///
/// Human vision has much lower spatial acuity for color than luminance,
/// so chroma HF components can be de-emphasized more aggressively.
#[rustfmt::skip]
const PSY_WEIGHT_UV: [u16; 16] = [
    32, 20, 10,  3,
    20, 14,  7,  2,
    10,  7,  4,  1,
     3,  2,  1,  1,
];

//------------------------------------------------------------------------------
// JND (Just Noticeable Difference) Visibility Thresholds
//
// Frequency-dependent visibility thresholds based on the CSF (Contrast
// Sensitivity Function). Coefficients below these thresholds are less
// likely to be perceptible and can be more aggressively quantized.
//
// Values are in the same scale as DCT coefficients (8-bit input).
// Based on research from JPEG-AI and modern perceptual codecs.

/// JND thresholds for 4x4 DCT luma coefficients (zig-zag order).
/// DC (position 0) has low threshold; HF positions have higher thresholds.
/// These represent the minimum perceivable coefficient magnitude.
#[rustfmt::skip]
const JND_THRESHOLD_Y: [u16; 16] = [
    // DC   1    2    3   (row 0 - lowest frequency)
     4,   6,   8,  12,
    // 4    5    6    7   (row 1)
     6,   8,  12,  18,
    // 8    9   10   11   (row 2)
     8,  12,  18,  26,
    //12   13   14   15   (row 3 - highest frequency)
    12,  18,  26,  36,
];

/// JND thresholds for 4x4 DCT chroma coefficients.
/// Chroma has higher thresholds (lower sensitivity) than luma.
#[rustfmt::skip]
const JND_THRESHOLD_UV: [u16; 16] = [
     8,  12,  16,  24,
    12,  16,  24,  36,
    16,  24,  36,  52,
    24,  36,  52,  72,
];

#[derive(Clone, Debug)]
/// Perceptual encoding configuration for a segment.
///
/// Controls how distortion is computed during mode selection and trellis
/// quantization. When all strengths are 0, behavior is identical to baseline.
///
/// This is primarily an internal type exposed for debugging/testing purposes.
#[doc(hidden)]
pub struct PsyConfig {
    /// Psy-RD strength (fixed-point, 0 = disabled).
    /// Penalizes energy loss (smoothing) in SATD domain.
    pub(crate) psy_rd_strength: u32,
    /// Psy-trellis strength (fixed-point, 0 = disabled).
    /// Biases trellis to retain perceptually important coefficients.
    pub(crate) psy_trellis_strength: u32,
    /// CSF weights for luma spectral distortion (TDisto).
    pub(crate) luma_csf: [u16; 16],
    /// CSF weights for chroma spectral distortion (TDisto).
    pub(crate) chroma_csf: [u16; 16],
    /// Distortion weights for trellis quantization.
    pub(crate) trellis_weights: [u16; 16],
    /// JND thresholds for luma coefficients.
    /// Coefficients below these values are less perceptible.
    pub(crate) jnd_threshold_y: [u16; 16],
    /// JND thresholds for chroma coefficients.
    pub(crate) jnd_threshold_uv: [u16; 16],
    /// Luminance factor for adaptive JND (0-255, 128=neutral).
    /// Dark (low) and bright (high) areas have different sensitivity.
    pub(crate) luminance_factor: u8,
    /// Contrast masking factor (0-255).
    /// High texture areas can hide more quantization noise.
    pub(crate) contrast_masking: u8,
}

impl Default for PsyConfig {
    fn default() -> Self {
        Self {
            psy_rd_strength: 0,
            psy_trellis_strength: 0,
            luma_csf: VP8_WEIGHT_Y,
            chroma_csf: VP8_WEIGHT_Y, // same as libwebp default
            trellis_weights: VP8_WEIGHT_TRELLIS,
            jnd_threshold_y: [0; 16],   // disabled by default
            jnd_threshold_uv: [0; 16],  // disabled by default
            luminance_factor: 128,      // neutral
            contrast_masking: 0,        // disabled
        }
    }
}

impl PsyConfig {
    /// Create a PsyConfig for the given encoding parameters.
    ///
    /// Method gating:
    /// - method 0-2: all features disabled (bit-identical to baseline)
    /// - method >= 3: enhanced CSF tables for luma and chroma TDisto
    /// - method >= 4: reserved (psy-rd disabled after butteraugli testing)
    /// - method >= 5: psy-trellis + JND thresholds enabled
    pub(crate) fn new(method: u8, quant_index: u8, _sns_strength: u8) -> Self {
        let mut config = Self::default();

        if method >= 3 {
            // Enhanced CSF tables: steeper HF rolloff for better perceptual coding
            config.luma_csf = PSY_WEIGHT_Y;
            config.chroma_csf = PSY_WEIGHT_UV;
        }

        // Psy-RD disabled for method 4 based on butteraugli testing (2026-02-04):
        // Even very low psy-rd strength hurts butteraugli scores because
        // butteraugli prefers smoother reconstructions while psy-rd penalizes
        // smoothing. The enhanced CSF tables alone (method 3+) provide the
        // best perceptual quality without the psy-rd side effects.
        //
        // Note: Psy-trellis at method 5+ still uses JND-gated coefficient
        // retention which works better with butteraugli since it only
        // protects perceptible coefficients rather than penalizing smoothing.
        let _ = method; // Silence unused warning if method >= 4 is removed

        if method >= 5 {
            // Psy-trellis: bias trellis DP to retain perceptually important
            // coefficients. Strength scales with quantizer since coarser
            // quantization zeros more coefficients.
            //
            // Tuned with butteraugli feedback (2026-02-04):
            // Very conservative: (quant_index * 6) >> 7 ≈ 0.047 * q
            // At q=50: strength=2, at q=80: strength=4
            config.psy_trellis_strength = (quant_index as u32 * 6) >> 7;

            // JND thresholds: scale base thresholds by quantizer
            // Higher quantizer = higher thresholds (more aggressive zeroing)
            // Base thresholds are for q=50, scale by q/50
            let scale = quant_index.max(20) as u32; // min q=20 to avoid divide by zero
            for i in 0..16 {
                // Scale threshold: base * (q / 50), clamped
                config.jnd_threshold_y[i] =
                    ((JND_THRESHOLD_Y[i] as u32 * scale) / 50).min(255) as u16;
                config.jnd_threshold_uv[i] =
                    ((JND_THRESHOLD_UV[i] as u32 * scale) / 50).min(255) as u16;
            }
        }

        config
    }

    /// Check if a luma coefficient is below JND threshold (imperceptible).
    ///
    /// Returns true if the coefficient magnitude is small enough that
    /// zeroing it won't cause visible artifacts.
    #[inline]
    pub(crate) fn is_below_jnd_y(&self, pos: usize, coeff: i32) -> bool {
        let threshold = self.jnd_threshold_y[pos] as i32;
        if threshold == 0 {
            return false; // JND disabled
        }

        let abs_coeff = coeff.abs();

        // Apply luminance adaptation (Weber's law)
        // Dark areas are more sensitive, bright areas less so
        let adapted_threshold = if self.luminance_factor < 100 {
            // Dark: reduce threshold (more sensitive)
            (threshold * self.luminance_factor as i32) / 128
        } else if self.luminance_factor > 160 {
            // Bright: increase threshold (less sensitive)
            (threshold * self.luminance_factor as i32) / 128
        } else {
            threshold
        };

        // Apply contrast masking
        // High texture areas can hide more noise
        let final_threshold = if self.contrast_masking > 0 {
            adapted_threshold + (self.contrast_masking as i32 / 4)
        } else {
            adapted_threshold
        };

        abs_coeff < final_threshold
    }

    /// Check if a chroma coefficient is below JND threshold.
    #[inline]
    pub(crate) fn is_below_jnd_uv(&self, pos: usize, coeff: i32) -> bool {
        let threshold = self.jnd_threshold_uv[pos] as i32;
        if threshold == 0 {
            return false;
        }
        coeff.abs() < threshold
    }

    /// Set luminance factor for adaptive JND.
    /// Call with average luminance of the macroblock (0-255).
    ///
    /// Reserved for per-macroblock JND adaptation (not yet integrated).
    #[allow(dead_code)]
    pub(crate) fn set_luminance(&mut self, avg_luma: u8) {
        self.luminance_factor = avg_luma;
    }

    /// Set contrast masking factor.
    /// Call with texture complexity metric (0=flat, 255=highly textured).
    ///
    /// Reserved for per-macroblock contrast adaptation (not yet integrated).
    #[allow(dead_code)]
    pub(crate) fn set_contrast_masking(&mut self, masking: u8) {
        self.contrast_masking = masking;
    }
}

//------------------------------------------------------------------------------
// SATD (Sum of Absolute Transformed Differences)
//
// Measures total signal energy in the Hadamard (Walsh-Hadamard) domain.
// Unlike TDisto which computes frequency-weighted differences between two blocks,
// SATD measures the absolute energy of a single block. Used by psy-rd to detect
// when the encoder is destroying texture energy.
//
// The #[multiversed] attribute enables autovectorization for the scalar code
// when compiled for x86-64-v3 (AVX2) or x86-64-v4 (AVX-512) targets.

/// Compute SATD for a 4x4 block of pixels.
///
/// Applies a 4x4 Hadamard transform and returns sum of |coefficients|.
/// No frequency weighting — this measures total signal energy.
///
/// With the `simd` feature enabled, uses autovectorization via #[multiversed].
///
/// # Arguments
/// * `block` - Pixel data (accessed with given stride)
/// * `stride` - Row stride of input buffer
#[cfg(feature = "simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
#[inline]
pub(crate) fn satd_4x4(block: &[u8], stride: usize) -> u32 {
    satd_4x4_impl(block, stride)
}

/// Non-SIMD version (or primary implementation when simd feature disabled)
#[cfg(not(feature = "simd"))]
#[inline]
pub(crate) fn satd_4x4(block: &[u8], stride: usize) -> u32 {
    satd_4x4_impl(block, stride)
}

/// SATD implementation - scalar code that autovectorizes well with #[multiversed]
#[inline(always)]
fn satd_4x4_impl(block: &[u8], stride: usize) -> u32 {
    let mut tmp = [0i32; 16];

    // Horizontal Hadamard
    for i in 0..4 {
        let row = i * stride;
        let a0 = i32::from(block[row]) + i32::from(block[row + 2]);
        let a1 = i32::from(block[row + 1]) + i32::from(block[row + 3]);
        let a2 = i32::from(block[row + 1]) - i32::from(block[row + 3]);
        let a3 = i32::from(block[row]) - i32::from(block[row + 2]);
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }

    // Vertical Hadamard + absolute sum
    let mut sum = 0u32;
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];

        sum += (a0 + a1).unsigned_abs();
        sum += (a3 + a2).unsigned_abs();
        sum += (a3 - a2).unsigned_abs();
        sum += (a0 - a1).unsigned_abs();
    }
    sum
}

/// Compute SATD for an 8x8 block as sum of four 4x4 SATDs.
#[inline]
pub(crate) fn satd_8x8(block: &[u8], stride: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..2 {
        for x in 0..2 {
            let offset = y * 4 * stride + x * 4;
            sum += satd_4x4(&block[offset..], stride);
        }
    }
    sum
}

/// Compute SATD for a 16x16 block as sum of sixteen 4x4 SATDs.
#[inline]
pub(crate) fn satd_16x16(block: &[u8], stride: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..4 {
        for x in 0..4 {
            let offset = y * 4 * stride + x * 4;
            sum += satd_4x4(&block[offset..], stride);
        }
    }
    sum
}

//------------------------------------------------------------------------------
// Perceptual Adaptive Quantization (2026 algorithms)
//
// Computes a masking-based alpha that measures local texture complexity.
// High-variance regions (grass, fabric, noise) are more tolerant of
// quantization artifacts, so they can use higher QP without visual loss.
//
// This implementation uses multiple perceptual models:
// 1. SATD-based activity - measures AC energy in transform domain
// 2. Luminance masking - Weber's law (brighter areas tolerate more error)
// 3. Edge detection - protect edges from coarse quantization
// 4. Multi-scale analysis - consider texture at 4x4 and 8x8 scales

/// Compute AC energy for a 4x4 block (DC-subtracted SATD).
///
/// This is more perceptually meaningful than variance because it
/// measures energy in the frequency domain where quantization happens.
#[inline]
fn ac_energy_4x4(block: &[u8], stride: usize) -> u32 {
    // Full SATD includes DC; subtract DC contribution for pure AC energy
    let satd = satd_4x4(block, stride);
    // DC coefficient is sum of all pixels, contributes 4*sum to SATD
    let mut dc_sum = 0u32;
    for y in 0..4 {
        for x in 0..4 {
            dc_sum += block[y * stride + x] as u32;
        }
    }
    // SATD of flat block = DC*4, so subtract that
    satd.saturating_sub(dc_sum)
}

/// Compute average luminance for a 4x4 block.
#[inline]
fn block_luminance_4x4(block: &[u8], stride: usize) -> u8 {
    let mut sum = 0u32;
    for y in 0..4 {
        for x in 0..4 {
            sum += block[y * stride + x] as u32;
        }
    }
    (sum / 16) as u8
}

/// Compute horizontal edge strength (Sobel-like).
#[inline]
fn edge_strength_4x4(block: &[u8], stride: usize) -> u32 {
    let mut h_edge = 0u32;
    let mut v_edge = 0u32;

    // Horizontal edges (vertical gradient)
    for y in 0..3 {
        for x in 0..4 {
            let diff = (block[(y + 1) * stride + x] as i32 - block[y * stride + x] as i32).unsigned_abs();
            h_edge += diff;
        }
    }

    // Vertical edges (horizontal gradient)
    for y in 0..4 {
        for x in 0..3 {
            let diff = (block[y * stride + x + 1] as i32 - block[y * stride + x] as i32).unsigned_abs();
            v_edge += diff;
        }
    }

    // Return maximum of both (edges are important in either direction)
    h_edge.max(v_edge)
}

/// Compute masking alpha for a 16x16 macroblock using modern perceptual models.
///
/// Returns a value 0-255 where:
/// - 0 = flat/edge block (needs fine quantization to avoid visible artifacts)
/// - 255 = highly textured (can tolerate coarser quantization)
///
/// Uses SATD-based AC energy, luminance masking, and edge protection.
///
/// # Arguments
/// * `src` - Y source pixels for this macroblock (accessed with given stride)
/// * `stride` - Row stride of source buffer
pub(crate) fn compute_masking_alpha(src: &[u8], stride: usize) -> u8 {
    let mut total_ac_energy = 0u64;
    let mut total_luminance = 0u32;
    let mut max_edge_strength = 0u32;
    let mut min_block_activity = u32::MAX;

    // Analyze each 4x4 sub-block
    for by in 0..4 {
        for bx in 0..4 {
            let base = by * 4 * stride + bx * 4;
            let block = &src[base..];

            // 1. AC energy (texture complexity in transform domain)
            let ac = ac_energy_4x4(block, stride);
            total_ac_energy += ac as u64;
            min_block_activity = min_block_activity.min(ac);

            // 2. Luminance for Weber's law
            total_luminance += block_luminance_4x4(block, stride) as u32;

            // 3. Edge detection
            let edge = edge_strength_4x4(block, stride);
            max_edge_strength = max_edge_strength.max(edge);
        }
    }

    // Average metrics
    let avg_ac_energy = (total_ac_energy / 16) as u32;
    let avg_luminance = total_luminance / 16;

    // Compute activity masking: textured areas can hide quantization noise
    // Using sqrt-like mapping for perceptual uniformity (similar to butteraugli)
    let activity_mask = if avg_ac_energy > 0 {
        // Map AC energy to 0-255 with diminishing returns at high energy
        // sqrt(energy) gives perceptually uniform scaling
        let energy_sqrt = (avg_ac_energy as f32).sqrt();
        (energy_sqrt * 8.0).min(255.0) as u32
    } else {
        0
    };

    // Luminance masking: Weber's law - brighter areas tolerate more absolute error
    // Dark areas (luminance < 50) are very sensitive
    // Mid areas (luminance ~128) have baseline sensitivity
    // Bright areas (luminance > 200) are slightly more tolerant
    let luminance_factor = if avg_luminance < 50 {
        // Dark: reduce masking (more sensitive to artifacts)
        200u32 // scale down by 200/256 ≈ 0.78
    } else if avg_luminance > 200 {
        // Bright: slightly increase masking (less sensitive)
        280u32 // scale up by 280/256 ≈ 1.09
    } else {
        // Mid-range: baseline
        256u32
    };

    // Edge protection: strong edges should NOT be heavily masked
    // because edge distortion is very visible
    let edge_penalty = if max_edge_strength > 300 {
        // Strong edge detected: reduce masking significantly
        64u32 // Penalize by subtracting this from final alpha
    } else if max_edge_strength > 150 {
        // Moderate edge
        32u32
    } else {
        0u32
    };

    // Uniformity check: if all 4x4 blocks have similar low activity,
    // this is a smooth gradient that needs protection (no masking)
    let uniformity_penalty = if min_block_activity < 50 && avg_ac_energy < 100 {
        // Smooth/gradient area: needs fine quantization
        80u32
    } else {
        0u32
    };

    // Combine factors
    let adjusted_activity = (activity_mask * luminance_factor) >> 8;
    let final_alpha = adjusted_activity
        .saturating_sub(edge_penalty)
        .saturating_sub(uniformity_penalty);

    final_alpha.min(255) as u8
}

//------------------------------------------------------------------------------
// Alpha Blending (used by analysis.rs)

/// Adjust DCT-histogram alpha using masking information.
///
/// Applies a masking-based delta to the DCT alpha. Textured regions
/// (high masking_alpha) get their alpha pushed higher (more tolerant of
/// coarse quantization), flat regions (low masking_alpha) get pushed lower
/// (need finer quantization).
///
/// Uses an additive delta rather than linear blending to preserve the
/// relative spread between DCT alphas across the image. Linear blending
/// compresses the alpha range, which can collapse segment diversity on
/// uniform-variance images.
///
/// # Arguments
/// * `dct_alpha` - Alpha from DCT histogram analysis (before finalization)
/// * `masking_alpha` - Alpha from `compute_masking_alpha` (0=flat, 255=textured)
/// * `method` - Encoding method level (0-6)
pub(crate) fn blend_masking_alpha(dct_alpha: i32, masking_alpha: u8, method: u8) -> i32 {
    if method < 4 {
        // No blending, pure DCT alpha (bit-identical to baseline)
        return dct_alpha;
    }

    // Center masking at 128 so flat regions push alpha down and textured
    // regions push alpha up. This is additive so it preserves the relative
    // spread between DCT alphas.
    let masking_delta = masking_alpha as i32 - 128;

    // Scale the delta by method level:
    // method 4: ±32 max adjustment (masking_delta * 64 / 256)
    // method 5-6: ±48 max adjustment (masking_delta * 96 / 256)
    let scaled_delta = if method >= 5 {
        (masking_delta * 96) >> 8
    } else {
        (masking_delta * 64) >> 8
    };

    dct_alpha + scaled_delta
}

//------------------------------------------------------------------------------
// Psy-RD Cost (currently disabled but kept for future experimentation)

/// Compute the psy-rd penalty for energy loss between source and reconstruction.
///
/// Returns a one-sided penalty: only penalizes when the reconstruction has
/// LESS energy than the source (smoothing/detail loss). Energy gain (ringing)
/// is not penalized because ringing is already captured by SSE.
///
/// # Arguments
/// * `src_satd` - SATD energy of source block
/// * `rec_satd` - SATD energy of reconstructed block
/// * `psy_rd_strength` - Scaling factor (0 = disabled)
#[inline]
pub(crate) fn psy_rd_cost(src_satd: u32, rec_satd: u32, psy_rd_strength: u32) -> i32 {
    if psy_rd_strength == 0 {
        return 0;
    }
    // One-sided: only penalize energy loss (src > rec means smoothing)
    let energy_loss = src_satd.saturating_sub(rec_satd);
    // Scale and shift to match distortion units
    ((psy_rd_strength as i64 * energy_loss as i64) >> 8) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Legacy variance-based masking (kept for comparison with SATD-based approach).
    ///
    /// This was the original masking method before switching to SATD-based AC energy.
    /// Kept here to verify the new method produces similar results on typical content.
    fn compute_masking_alpha_variance(src: &[u8], stride: usize) -> u8 {
        let mut total_variance = 0u64;

        for by in 0..4 {
            for bx in 0..4 {
                let base = by * 4 * stride + bx * 4;

                // Compute mean of 4x4 block
                let mut sum = 0u32;
                for y in 0..4 {
                    for x in 0..4 {
                        sum += src[base + y * stride + x] as u32;
                    }
                }
                let mean = sum / 16;

                // Compute variance
                let mut var = 0u64;
                for y in 0..4 {
                    for x in 0..4 {
                        let diff = src[base + y * stride + x] as i32 - mean as i32;
                        var += (diff * diff) as u64;
                    }
                }
                total_variance += var / 16;
            }
        }

        let avg_variance = total_variance / 16;
        ((avg_variance * 255) / (avg_variance + 100)).min(255) as u8
    }

    #[test]
    fn test_masking_methods_correlate() {
        // Both masking methods should rank blocks similarly
        // (flat < moderate < textured)
        let flat = [128u8; 256];
        let mut textured = [0u8; 256];
        for (i, p) in textured.iter_mut().enumerate() {
            *p = ((i * 17 + i / 16 * 31) % 256) as u8;
        }

        let flat_satd = compute_masking_alpha(&flat, 16);
        let flat_var = compute_masking_alpha_variance(&flat, 16);
        let text_satd = compute_masking_alpha(&textured, 16);
        let text_var = compute_masking_alpha_variance(&textured, 16);

        // Both methods should agree: flat has lower alpha than textured
        assert!(flat_satd < text_satd, "SATD: flat={} should < textured={}", flat_satd, text_satd);
        assert!(flat_var < text_var, "Variance: flat={} should < textured={}", flat_var, text_var);
    }

    #[test]
    fn test_satd_4x4_flat_block() {
        // Flat block (all same value) should have energy only in DC
        let flat = [128u8; 64]; // 4x4 at stride 16
        let s = satd_4x4(&flat, 16);
        // DC = 4*4*128 = 2048, all AC = 0
        // Hadamard of flat input: DC coefficient = sum of all pixels = 2048
        assert_eq!(s, 2048);
    }

    #[test]
    fn test_satd_4x4_noise_has_energy() {
        // Block with varying values should have significant energy
        let noise: [u8; 16] = [
            10, 200, 50, 180, 90, 30, 220, 70, 150, 110, 40, 190, 60, 250, 20, 100,
        ];
        let s = satd_4x4(&noise, 4);
        // Should be much larger than flat block
        assert!(s > 2048, "Noisy block SATD={} should exceed flat DC", s);
    }

    #[test]
    fn test_satd_16x16_is_sum_of_4x4s() {
        // Fill a 16x16 block with a pattern
        let mut block = [0u8; 16 * 16];
        for (i, pixel) in block.iter_mut().enumerate() {
            *pixel = (i * 7 + 13) as u8; // arbitrary pattern
        }
        let total = satd_16x16(&block, 16);
        let mut manual_sum = 0u32;
        for y in 0..4 {
            for x in 0..4 {
                manual_sum += satd_4x4(&block[y * 4 * 16 + x * 4..], 16);
            }
        }
        assert_eq!(total, manual_sum);
    }

    #[test]
    fn test_psy_rd_cost_disabled() {
        assert_eq!(psy_rd_cost(1000, 500, 0), 0);
    }

    #[test]
    fn test_psy_rd_cost_energy_loss() {
        // Source has more energy than recon => penalty
        let cost = psy_rd_cost(1000, 500, 256);
        assert!(cost > 0, "Energy loss should produce positive penalty");
    }

    #[test]
    fn test_psy_rd_cost_energy_gain_no_penalty() {
        // Recon has more energy than source => no penalty (one-sided)
        let cost = psy_rd_cost(500, 1000, 256);
        assert_eq!(cost, 0, "Energy gain should not be penalized");
    }

    #[test]
    fn test_psy_rd_cost_scales_with_strength() {
        let cost_low = psy_rd_cost(1000, 500, 100);
        let cost_high = psy_rd_cost(1000, 500, 200);
        assert!(
            cost_high > cost_low,
            "Higher strength should give higher penalty"
        );
    }
}
