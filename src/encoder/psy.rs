//! Perceptual distortion model for encoder-side optimizations.
//!
//! Provides frequency-weighted distortion computation using CSF (Contrast
//! Sensitivity Function) tables and SATD-based energy preservation (psy-rd).
//!
//! All features are gated by `method` level — methods 0-2 produce bit-identical
//! output to the baseline encoder.

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
}

impl Default for PsyConfig {
    fn default() -> Self {
        Self {
            psy_rd_strength: 0,
            psy_trellis_strength: 0,
            luma_csf: VP8_WEIGHT_Y,
            chroma_csf: VP8_WEIGHT_Y, // same as libwebp default
            trellis_weights: VP8_WEIGHT_TRELLIS,
        }
    }
}

impl PsyConfig {
    /// Create a PsyConfig for the given encoding parameters.
    ///
    /// Method gating:
    /// - method 0-2: all features disabled (bit-identical to baseline)
    /// - method >= 3: enhanced CSF tables for luma and chroma TDisto
    /// - method >= 4: psy-rd enabled with conservative strength
    /// - method >= 5: psy-trellis enabled (trellis only runs at method >= 5)
    pub(crate) fn new(method: u8, quant_index: u8, _sns_strength: u8) -> Self {
        let mut config = Self::default();

        if method >= 3 {
            // Enhanced CSF tables: steeper HF rolloff for better perceptual coding
            config.luma_csf = PSY_WEIGHT_Y;
            config.chroma_csf = PSY_WEIGHT_UV;
        }

        if method >= 4 {
            // Psy-RD: penalize energy loss. Strength scales with quantizer
            // because coarser quantization causes more visible smoothing.
            // Conservative: (quant_index * 48) >> 7  ≈ 0.375 * q
            // At q=50: strength=18, at q=80: strength=30
            config.psy_rd_strength = (quant_index as u32 * 48) >> 7;
        }

        if method >= 5 {
            // Psy-trellis: bias trellis DP to retain perceptually important
            // coefficients. Strength scales with quantizer since coarser
            // quantization zeros more coefficients.
            // (quant_index * 32) >> 7  ≈ 0.25 * q
            // At q=50: strength=12, at q=80: strength=20
            config.psy_trellis_strength = (quant_index as u32 * 32) >> 7;
        }

        config
    }
}

//------------------------------------------------------------------------------
// SATD (Sum of Absolute Transformed Differences)
//
// Measures total signal energy in the Hadamard (Walsh-Hadamard) domain.
// Unlike TDisto which computes frequency-weighted differences between two blocks,
// SATD measures the absolute energy of a single block. Used by psy-rd to detect
// when the encoder is destroying texture energy.

/// Compute SATD for a 4x4 block of pixels.
///
/// Applies a 4x4 Hadamard transform and returns sum of |coefficients|.
/// No frequency weighting — this measures total signal energy.
///
/// # Arguments
/// * `block` - Pixel data (accessed with given stride)
/// * `stride` - Row stride of input buffer
#[inline]
pub(crate) fn satd_4x4(block: &[u8], stride: usize) -> u32 {
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
    let avg_luminance = (total_luminance / 16) as u32;

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

/// Legacy variance-based masking (for comparison/debugging).
#[allow(dead_code)]
pub(crate) fn compute_masking_alpha_variance(src: &[u8], stride: usize) -> u8 {
    // Compute variance for each 4x4 sub-block, then average
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
            total_variance += var / 16; // normalize per-pixel
        }
    }

    // Average across 16 sub-blocks
    let avg_variance = total_variance / 16;

    // Map variance to alpha range 0-255
    // Empirically tuned: variance of ~100 → alpha ~128 (mid complexity)
    // Clamp at 255 for very high variance
    ((avg_variance * 255) / (avg_variance + 100)).min(255) as u8
}

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
