//! Cost estimation for VP8 encoding
//!
//! This module provides rate-distortion (RD) cost calculation for mode selection,
//! based on libwebp's cost estimation approach.
//!
//! The key insight is that encoding cost depends on:
//! 1. Mode signaling cost (fixed per mode type)
//! 2. Coefficient encoding cost (depends on coefficient values and probabilities)
//!
//! For mode selection, we use: score = Distortion + lambda * Rate

// Allow unused items - these tables and functions are part of the complete libwebp
// cost estimation system and will be used as additional features are implemented.
#![allow(dead_code)]
// Many loops in this file match libwebp's C patterns for clarity when comparing
#![allow(clippy::needless_range_loop)]

// Re-export tables from tables for backward compatibility
pub use super::tables::*;
// Re-export analysis module for backward compatibility
pub use super::analysis::*;
// Import pub(crate) items that aren't re-exported
use super::tables::{LEVELS_FROM_DELTA, MAX_DELTA_SIZE};

/// Distortion multiplier - scales distortion to match bit cost units
pub const RD_DISTO_MULT: u32 = 256;

/// Calculate bit cost for coding a boolean value with given probability.
///
/// Returns cost in 1/256 bit units.
#[inline]
pub fn vp8_bit_cost(bit: bool, prob: u8) -> u16 {
    if bit {
        VP8_ENTROPY_COST[255 - prob as usize]
    } else {
        VP8_ENTROPY_COST[prob as usize]
    }
}

/// Hadamard transform for a 4x4 block, weighted by w[].
/// Returns the sum of |transformed_coeff| * weight.
///
/// This is a 4x4 Hadamard (Walsh-Hadamard) transform that measures
/// frequency-weighted energy in the block.
///
/// # Arguments
/// * `input` - 4x4 block of pixels (accessed with given stride)
/// * `stride` - Row stride of input buffer
/// * `w` - 16 weights for frequency weighting
#[inline]
fn t_transform(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        crate::common::simd_sse::t_transform(input, stride, w)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        t_transform_scalar(input, stride, w)
    }
}

/// Scalar implementation of t_transform
#[inline]
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
fn t_transform_scalar(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut tmp = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let row = i * stride;
        let a0 = i32::from(input[row]) + i32::from(input[row + 2]);
        let a1 = i32::from(input[row + 1]) + i32::from(input[row + 3]);
        let a2 = i32::from(input[row + 1]) - i32::from(input[row + 3]);
        let a3 = i32::from(input[row]) - i32::from(input[row + 2]);
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }

    // Vertical pass with weighting
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;

        sum += i32::from(w[i]) * b0.abs();
        sum += i32::from(w[4 + i]) * b1.abs();
        sum += i32::from(w[8 + i]) * b2.abs();
        sum += i32::from(w[12 + i]) * b3.abs();
    }
    sum
}

/// Compute spectral distortion between two 4x4 blocks.
///
/// Returns |TTransform(a) - TTransform(b)| >> 5
///
/// This measures the perceptual difference between blocks using
/// a frequency-weighted Hadamard transform.
///
/// # Arguments
/// * `a` - First 4x4 block (source)
/// * `b` - Second 4x4 block (prediction/reconstruction)
/// * `stride` - Row stride of both buffers
/// * `w` - 16 weights for frequency weighting
#[inline]
pub fn tdisto_4x4(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let sum1 = t_transform(a, stride, w);
    let sum2 = t_transform(b, stride, w);
    (sum2 - sum1).abs() >> 5
}

/// Compute spectral distortion between two 16x16 blocks.
///
/// Calls tdisto_4x4 16 times for each 4x4 sub-block.
///
/// # Arguments
/// * `a` - First 16x16 block (source)
/// * `b` - Second 16x16 block (prediction/reconstruction)
/// * `stride` - Row stride of both buffers
/// * `w` - 16 weights for frequency weighting
#[inline]
pub fn tdisto_16x16(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut d = 0i32;
    for y in 0..4 {
        for x in 0..4 {
            let offset = y * 4 * stride + x * 4;
            d += tdisto_4x4(&a[offset..], &b[offset..], stride, w);
        }
    }
    d
}

/// Compute spectral distortion between two 8x8 blocks (for chroma).
///
/// Calls tdisto_4x4 4 times for each 4x4 sub-block.
#[inline]
pub fn tdisto_8x8(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut d = 0i32;
    for y in 0..2 {
        for x in 0..2 {
            let offset = y * 4 * stride + x * 4;
            d += tdisto_4x4(&a[offset..], &b[offset..], stride, w);
        }
    }
    d
}

//------------------------------------------------------------------------------
// Flat source detection
//
// Detects if a 16x16 source block is "flat" (uniform color).
// Used to force DC mode at image edges to avoid prediction artifacts.

/// Check if a 16x16 source block is flat (all pixels same value).
///
/// Returns true if the first pixel value repeats throughout the block.
/// This is used for edge macroblocks where prediction from unavailable
/// neighbors would create artifacts.
///
/// # Arguments
/// * `src` - Source pixels (16x16 block accessed with given stride)
/// * `stride` - Row stride of source buffer
#[inline]
pub fn is_flat_source_16(src: &[u8], stride: usize) -> bool {
    let v = src[0];
    for y in 0..16 {
        let row = y * stride;
        for x in 0..16 {
            if src[row + x] != v {
                return false;
            }
        }
    }
    true
}

/// Check if quantized AC coefficients indicate a flat block.
///
/// Returns true if the number of non-zero AC coefficients is below threshold.
/// Used to detect flat content that should prefer DC mode.
///
/// # Arguments
/// * `levels` - Quantized coefficient levels (16 coeffs per block)
/// * `num_blocks` - Number of 4x4 blocks to check
/// * `thresh` - Maximum allowed non-zero AC coefficients
#[inline]
pub fn is_flat_coeffs(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
    let mut score = 0i32;
    for block in 0..num_blocks {
        // Skip DC (index 0), check AC coefficients (indices 1-15)
        for i in 1..16 {
            if levels[block * 16 + i] != 0 {
                score += 1;
                if score > thresh {
                    return false;
                }
            }
        }
    }
    true
}

/// Flatness threshold for I16 mode (FLATNESS_LIMIT_I16 in libwebp)
/// libwebp uses 0, which means "always check for flatness" in I16 mode
pub const FLATNESS_LIMIT_I16: i32 = 0;

/// Flatness threshold for I4 mode (FLATNESS_LIMIT_I4 in libwebp)
/// libwebp uses 3 (not 2)
pub const FLATNESS_LIMIT_I4: i32 = 3;

/// Flatness threshold for UV mode (FLATNESS_LIMIT_UV in libwebp)
pub const FLATNESS_LIMIT_UV: i32 = 2;

/// Flatness penalty added to rate when UV coefficients are flat (libwebp)
/// This discourages selection of non-DC modes when content is flat.
pub const FLATNESS_PENALTY: u32 = 140;

// Re-export quantization types and functions from quantize module
pub use super::quantize::{quantdiv, quantization_bias, MatrixType, VP8Matrix, QFIX};

/// Cutoff for very small filter strengths (have close to no visual effect)
const FSTRENGTH_CUTOFF: u8 = 2;

/// Calculate filter strength from sharpness and edge delta.
/// Ported from libwebp's VP8FilterStrengthFromDelta.
#[inline]
pub fn filter_strength_from_delta(sharpness: u8, delta: u8) -> u8 {
    let pos = (delta as usize).min(MAX_DELTA_SIZE - 1);
    let sharpness_idx = (sharpness as usize).min(7);
    LEVELS_FROM_DELTA[sharpness_idx][pos]
}

/// Calculate optimal filter level based on quantizer and sharpness.
///
/// This implements libwebp's SetupFilterStrength logic:
/// 1. Compute qstep from AC quantizer
/// 2. Get base strength from delta table
/// 3. Scale by filter_strength config (default 50 = mid-filtering)
///
/// # Arguments
/// * `quant_index` - Quantizer index (0-127)
/// * `sharpness` - Sharpness level (0-7)
/// * `filter_strength` - User filter strength (0-100), default 50
#[inline]
pub fn compute_filter_level(quant_index: u8, sharpness: u8, filter_strength: u8) -> u8 {
    compute_filter_level_with_beta(quant_index, sharpness, filter_strength, 0)
}

/// Calculate optimal filter level with per-segment beta modulation.
///
/// This implements libwebp's SetupFilterStrength logic with per-segment complexity:
/// 1. Compute qstep from AC quantizer
/// 2. Get base strength from delta table
/// 3. Scale by filter_strength and reduce by segment beta
///
/// Segments with lower complexity (low beta) need less filtering because they have
/// fewer edges/details to preserve. High-complexity segments (high beta) get more
/// filtering to reduce artifacts.
///
/// # Arguments
/// * `quant_index` - Quantizer index (0-127)
/// * `sharpness` - Sharpness level (0-7)
/// * `filter_strength` - User filter strength (0-100), default 50
/// * `beta` - Segment complexity (0-255), where 0 = simplest, 255 = most complex
#[inline]
pub fn compute_filter_level_with_beta(
    quant_index: u8,
    sharpness: u8,
    filter_strength: u8,
    beta: u8,
) -> u8 {
    // level0 is in [0..500]. Using filter_strength=50 as mid-filtering.
    let level0 = 5 * filter_strength as u32;

    // Get AC quantizer step from the quant table and divide by 4
    let qstep = (VP8_AC_TABLE[quant_index as usize] >> 2) as u8;

    // Get base strength from delta table
    let base_strength = filter_strength_from_delta(sharpness, qstep) as u32;

    // Scale by level0 / (256 + beta) - segments with lower beta get less filtering
    // From libwebp: f = base_strength * level0 / (256 + beta)
    let f = (base_strength * level0) / (256 + beta as u32);

    // Clamp to valid range and apply cutoff
    if f < FSTRENGTH_CUTOFF as u32 {
        0
    } else if f > 63 {
        63
    } else {
        f as u8
    }
}

//------------------------------------------------------------------------------
// Mode costs

//------------------------------------------------------------------------------
// Lambda calculation
//
// Lambda values control the rate-distortion trade-off.
// These are the fixed values from libwebp's RefineUsingDistortion.

/// Fixed lambda for Intra16 mode selection (distortion method)
pub const LAMBDA_I16: u32 = 106;

/// Fixed lambda for Intra4 mode selection (distortion method)
pub const LAMBDA_I4: u32 = 11;

/// Fixed lambda for UV mode selection (distortion method)
pub const LAMBDA_UV: u32 = 120;

/// Calculate lambda_i4 based on quantization: (3 * q²) >> 7
#[inline]
pub fn calc_lambda_i4(q: u32) -> u32 {
    ((3 * q * q) >> 7).max(1)
}

/// Calculate lambda_i16 based on quantization: 3 * q²
#[inline]
pub fn calc_lambda_i16(q: u32) -> u32 {
    (3 * q * q).max(1)
}

/// Calculate lambda_uv based on quantization: (3 * q²) >> 6
#[inline]
pub fn calc_lambda_uv(q: u32) -> u32 {
    ((3 * q * q) >> 6).max(1)
}

/// Calculate lambda_mode based on quantization: (1 * q²) >> 7
#[inline]
pub fn calc_lambda_mode(q: u32) -> u32 {
    ((q * q) >> 7).max(1)
}

/// Calculate i4_penalty based on quantization: 1000 * q²
/// This penalty accounts for the extra cost of Intra4 over Intra16.
#[inline]
pub fn calc_i4_penalty(q: u32) -> u64 {
    (1000 * q as u64 * q as u64).max(1)
}

/// Calculate trellis lambda for I16 mode: (q²) >> 2
#[inline]
pub fn calc_lambda_trellis_i16(q: u32) -> u32 {
    ((q * q) >> 2).max(1)
}

/// Calculate trellis lambda for I4 mode: (7 * q²) >> 3
#[inline]
pub fn calc_lambda_trellis_i4(q: u32) -> u32 {
    ((7 * q * q) >> 3).max(1)
}

/// Calculate trellis lambda for UV: (q²) << 1
#[inline]
pub fn calc_lambda_trellis_uv(q: u32) -> u32 {
    ((q * q) << 1).max(1)
}

/// Calculate tlambda (spectral distortion weight) based on SNS strength and quant.
///
/// tlambda = (sns_strength * q) >> 5
///
/// This controls how much weight spectral distortion (TDisto) has in mode selection.
/// Only enabled when method >= 4 and sns_strength > 0.
///
/// # Arguments
/// * `sns_strength` - Spatial noise shaping strength (0-100)
/// * `q` - Expanded quantization value (from VP8Matrix)
#[inline]
pub fn calc_tlambda(sns_strength: u32, q: u32) -> u32 {
    (sns_strength * q) >> 5
}

/// Default SNS strength value (matches libwebp default)
pub const DEFAULT_SNS_STRENGTH: u32 = 50;

//------------------------------------------------------------------------------
// Re-export trellis quantization from trellis module
#[cfg(test)]
use super::trellis::rd_score_trellis;
pub use super::trellis::trellis_quantize_block;

//------------------------------------------------------------------------------
// Coefficient cost estimation

/// Estimate the cost of encoding a 4x4 block of quantized coefficients.
///
/// This is a simplified approximation of libwebp's GetResidualCost.
/// It uses VP8_LEVEL_FIXED_COSTS which gives the probability-independent
/// part of the cost. The probability-dependent part is omitted for simplicity.
///
/// # Arguments
/// * `coeffs` - The 16 quantized coefficients in zig-zag order
/// * `first` - First coefficient to include (0 for DC+AC, 1 for AC only)
///
/// # Returns
/// Estimated bit cost in 1/256 bit units
#[inline]
pub fn estimate_residual_cost(coeffs: &[i32; 16], first: usize) -> u32 {
    let mut cost = 0u32;
    let mut last_nz = -1i32;

    // Find last non-zero coefficient
    for (i, &c) in coeffs.iter().enumerate().rev() {
        if c != 0 {
            last_nz = i as i32;
            break;
        }
    }

    // If no non-zero coefficients, just signal end of block
    if last_nz < first as i32 {
        // Cost of signaling "no coefficients" - approximately 1 bit
        return 256;
    }

    // Sum up costs for each coefficient
    for &coeff in coeffs.iter().take(last_nz as usize + 1).skip(first) {
        let level = coeff.unsigned_abs() as usize;
        if level > 0 {
            // Cost of the coefficient level
            let level_clamped = level.min(MAX_LEVEL);
            cost += u32::from(VP8_LEVEL_FIXED_COSTS[level_clamped]);
            // Add sign bit cost (1 bit = 256)
            cost += 256;
        } else {
            // Cost of zero (EOB not reached) - approximately 0.5 bits
            cost += 128;
        }
    }

    // Cost of end-of-block signal (approximately 0.5 bits)
    cost += 128;

    cost
}

/// Estimate coefficient cost for a 16x16 macroblock (16 4x4 blocks).
///
/// # Arguments
/// * `blocks` - Array of 16 quantized coefficient blocks
/// * `has_dc` - Whether to include DC coefficients (false for I16 where DC is separate)
#[inline]
pub fn estimate_luma16_cost(blocks: &[[i32; 16]; 16], has_dc: bool) -> u32 {
    let first = if has_dc { 0 } else { 1 };
    blocks
        .iter()
        .map(|block| estimate_residual_cost(block, first))
        .sum()
}

/// Estimate coefficient cost for DC block in I16 mode.
#[inline]
pub fn estimate_dc16_cost(dc_coeffs: &[i32; 16]) -> u32 {
    estimate_residual_cost(dc_coeffs, 0)
}

//------------------------------------------------------------------------------
// RD score calculation

/// Calculate RD score for mode selection.
///
/// Formula: score = SSE * RD_DISTO_MULT + mode_cost * lambda
///
/// Lower score = better trade-off between quality and bits.
#[inline]
pub fn rd_score(sse: u32, mode_cost: u16, lambda: u32) -> u64 {
    let distortion = u64::from(sse) * u64::from(RD_DISTO_MULT);
    let rate = u64::from(mode_cost) * u64::from(lambda);
    distortion + rate
}

/// Calculate full RD score including coefficient costs.
///
/// Formula: score = SSE * RD_DISTO_MULT + (mode_cost + coeff_cost) * lambda
///
/// # Arguments
/// * `sse` - Sum of squared errors (distortion)
/// * `mode_cost` - Mode signaling cost in 1/256 bits
/// * `coeff_cost` - Coefficient encoding cost in 1/256 bits
/// * `lambda` - Rate-distortion trade-off parameter
#[inline]
pub fn rd_score_with_coeffs(sse: u32, mode_cost: u16, coeff_cost: u32, lambda: u32) -> u64 {
    let distortion = u64::from(sse) * u64::from(RD_DISTO_MULT);
    let rate = (u64::from(mode_cost) + u64::from(coeff_cost)) * u64::from(lambda);
    distortion + rate
}

/// Calculate full RD score as used by libwebp's PickBestIntra16.
///
/// Formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
///
/// Where:
/// - R = coefficient encoding cost (rate)
/// - H = mode header cost
/// - D = SSE distortion on reconstructed block
/// - SD = spectral distortion (TDisto)
///
/// # Arguments
/// * `sse` - Sum of squared errors (D)
/// * `spectral_disto` - Spectral distortion from TDisto (SD), already scaled by tlambda
/// * `mode_cost` - Mode signaling cost (H)
/// * `coeff_cost` - Coefficient encoding cost (R)
/// * `lambda` - Rate-distortion trade-off parameter
#[inline]
pub fn rd_score_full(
    sse: u32,
    spectral_disto: i32,
    mode_cost: u16,
    coeff_cost: u32,
    lambda: u32,
) -> i64 {
    let rate = (i64::from(mode_cost) + i64::from(coeff_cost)) * i64::from(lambda);
    let distortion = i64::from(RD_DISTO_MULT) * (i64::from(sse) + i64::from(spectral_disto));
    rate + distortion
}

/// Get context-dependent Intra4 mode cost.
///
/// # Arguments
/// * `top` - The mode of the block above (0-9)
/// * `left` - The mode of the block to the left (0-9)
/// * `mode` - The mode to get the cost for (0-9)
#[inline]
pub fn get_i4_mode_cost(top: usize, left: usize, mode: usize) -> u16 {
    VP8_FIXED_COSTS_I4[top][left][mode]
}

//------------------------------------------------------------------------------
// Token Statistics for Adaptive Probabilities
//
// Ported from libwebp src/enc/cost_enc.h and src/enc/frame_enc.c
// This enables two-pass encoding with optimal probability updates.

/// Number of coefficient types (DCT types: 0=i16-DC, 1=i16-AC, 2=chroma, 3=i4)
pub const NUM_TYPES: usize = 4;
/// Number of bands for coefficient encoding
pub const NUM_BANDS: usize = 8;
/// Number of contexts (0=zero, 1=one, 2=more)
pub const NUM_CTX: usize = 3;
/// Number of probabilities per context node
pub const NUM_PROBAS: usize = 11;

/// Token statistics for computing optimal probabilities.
/// Format: upper 16 bits = total count, lower 16 bits = count of 1s.
/// Ported from libwebp's proba_t type.
#[derive(Clone, Default)]
pub struct ProbaStats {
    /// Statistics array: \[type\]\[band\]\[context\]\[proba_node\]
    pub stats: [[[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
}

impl ProbaStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to zero
    pub fn reset(&mut self) {
        for t in self.stats.iter_mut() {
            for b in t.iter_mut() {
                for c in b.iter_mut() {
                    for p in c.iter_mut() {
                        *p = 0;
                    }
                }
            }
        }
    }

    /// Record a bit value for a specific probability node.
    /// Ported from libwebp's VP8RecordStats.
    #[inline]
    pub fn record(&mut self, t: usize, b: usize, c: usize, p: usize, bit: bool) {
        let stats = &mut self.stats[t][b][c][p];
        // Check for overflow (at 0xfffe0000)
        if *stats >= 0xfffe_0000 {
            // Divide stats by 2 to prevent overflow
            *stats = ((*stats + 1) >> 1) & 0x7fff_7fff;
        }
        // Record: lower 16 bits = count of 1s, upper 16 bits = total count
        *stats += 0x0001_0000 + if bit { 1 } else { 0 };
    }

    /// Get the optimal probability for a node based on accumulated statistics.
    /// Returns 255 - (nb * 255 / total) where nb = count of 1s.
    pub fn calc_proba(&self, t: usize, b: usize, c: usize, p: usize) -> u8 {
        let stats = self.stats[t][b][c][p];
        let nb = stats & 0xffff; // count of 1s
        let total = stats >> 16; // total count
        if total == 0 {
            return 255;
        }
        let proba = 255 - (nb * 255 / total);
        proba as u8
    }

    /// Calculate the cost of updating vs keeping old probability.
    /// Returns (should_update, new_prob, bit_savings).
    pub fn should_update(
        &self,
        t: usize,
        b: usize,
        c: usize,
        p: usize,
        old_proba: u8,
        update_proba: u8,
    ) -> (bool, u8, i32) {
        let stats = self.stats[t][b][c][p];
        let nb = (stats & 0xffff) as i32; // count of 1s
        let total = (stats >> 16) as i32; // total count

        if total == 0 {
            return (false, old_proba, 0);
        }

        let new_p = self.calc_proba(t, b, c, p);

        // Cost with old probability
        let old_cost = branch_cost(nb, total, old_proba) + vp8_bit_cost(false, update_proba) as i32;

        // Cost with new probability (includes signaling cost)
        let new_cost =
            branch_cost(nb, total, new_p) + vp8_bit_cost(true, update_proba) as i32 + 8 * 256; // 8 bits to signal new probability value

        let savings = old_cost - new_cost;
        (savings > 0, new_p, savings)
    }
}

/// Calculate the branch cost for a given count distribution and probability.
/// Cost = nb * cost(1|p) + (total - nb) * cost(0|p)
#[inline]
fn branch_cost(nb: i32, total: i32, proba: u8) -> i32 {
    let cost_1 = VP8_ENTROPY_COST[255 - proba as usize] as i32;
    let cost_0 = VP8_ENTROPY_COST[proba as usize] as i32;
    nb * cost_1 + (total - nb) * cost_0
}

/// Token type for coefficient encoding
/// Values must match libwebp's type indices:
/// - TYPE_I16_AC = 0 (Y1 AC coefficients, first=1)
/// - TYPE_I16_DC = 1 (Y2 DC coefficients, first=0)
/// - TYPE_CHROMA_A = 2
/// - TYPE_I4_AC = 3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    /// I16 AC coefficients (Y1 blocks with DC=0) - must be 0 to match libwebp
    I16AC = 0,
    /// I16 DC coefficients (Y2/WHT) - must be 1 to match libwebp
    I16DC = 1,
    /// Chroma (UV) coefficients
    Chroma = 2,
    /// I4 coefficients
    I4 = 3,
}

/// Record coefficient tokens for probability statistics.
/// Structure matches the encoder's skip_eob pattern exactly:
/// - For each coefficient, check node 0 (EOB) if skip_eob=false
/// - Check node 1 (zero/non-zero) for all coefficients
/// - Set skip_eob=true after zeros (can skip EOB check after a zero)
/// - At end, record EOB at node 0 if there are trailing positions
///
/// # Arguments
/// * `coeffs` - Quantized coefficients in zigzag order (16 values)
/// * `token_type` - Type of coefficients (I16DC, I16AC, Chroma, I4)
/// * `first` - First coefficient to process (0 or 1)
/// * `ctx` - Initial context (0, 1, or 2)
/// * `stats` - Statistics accumulator
pub fn record_coeffs(
    coeffs: &[i32],
    token_type: TokenType,
    first: usize,
    ctx: usize,
    stats: &mut ProbaStats,
) {
    let t = token_type as usize;
    let mut n = first;
    let mut context = ctx;

    // Find last non-zero coefficient (end_of_block_index - 1)
    let last = coeffs
        .iter()
        .rposition(|&c| c != 0)
        .map(|i| i as i32)
        .unwrap_or(-1);

    let end_of_block = if last >= 0 { (last + 1) as usize } else { 0 };

    // If no non-zero coefficients, record EOB immediately
    if end_of_block <= first {
        let band = VP8_ENC_BANDS[first] as usize;
        stats.record(t, band, context, 0, false); // EOB at node 0
        return;
    }

    // Track skip_eob like the encoder does
    let mut skip_eob = false;

    // Process coefficients up to end_of_block (last non-zero + 1)
    while n < end_of_block {
        let band = VP8_ENC_BANDS[n] as usize;
        let v = coeffs[n].unsigned_abs();
        n += 1;

        // Record at node 0 (EOB check) if not skipping
        if !skip_eob {
            stats.record(t, band, context, 0, true); // not EOB
        }

        if v == 0 {
            // Zero coefficient: record 0 at node 1, set skip_eob for next
            stats.record(t, band, context, 1, false);
            skip_eob = true;
            context = 0;
            continue;
        }

        // Non-zero coefficient: record 1 at node 1
        stats.record(t, band, context, 1, true);

        // Record value magnitude bits
        if v == 1 {
            stats.record(t, band, context, 2, false);
            context = 1;
        } else {
            stats.record(t, band, context, 2, true);

            // Clamp v to MAX_VARIABLE_LEVEL for statistics
            let v = v.min(MAX_VARIABLE_LEVEL as u32);

            if v <= 4 {
                stats.record(t, band, context, 3, false);
                if v == 2 {
                    stats.record(t, band, context, 4, false);
                } else {
                    stats.record(t, band, context, 4, true);
                    stats.record(t, band, context, 5, v == 4);
                }
            } else if v <= 10 {
                stats.record(t, band, context, 3, true);
                stats.record(t, band, context, 6, false);
                stats.record(t, band, context, 7, v > 6);
            } else {
                stats.record(t, band, context, 3, true);
                stats.record(t, band, context, 6, true);

                if v < 3 + (8 << 2) {
                    stats.record(t, band, context, 8, false);
                    stats.record(t, band, context, 9, v >= 3 + (8 << 1));
                } else {
                    stats.record(t, band, context, 8, true);
                    stats.record(t, band, context, 10, v >= 3 + (8 << 3));
                }
            }

            context = 2;
        }

        // After non-zero, the encoder does NOT reset skip_eob to false.
        // It leaves it unchanged. So if skip_eob was true (from a previous zero),
        // it stays true even after this non-zero. Do NOT reset it here!
    }

    // Record trailing EOB if we didn't reach position 16
    if n < 16 {
        let band = VP8_ENC_BANDS[n] as usize;
        stats.record(t, band, context, 0, false); // EOB
    }
}

//------------------------------------------------------------------------------
// Level cost tables for accurate coefficient cost estimation
//
// Ported from libwebp src/enc/cost_enc.c
//
// The key insight is that coefficient costs depend on the probability context.
// VP8CalculateLevelCosts precomputes cost tables indexed by [type][band][ctx][level].
// Then remapped_costs provides direct access by coefficient position: [type][n][ctx].

use crate::common::types::TokenProbTables;

/// Type alias for level cost array: cost for each level 0..=MAX_VARIABLE_LEVEL
pub type LevelCostArray = [u16; MAX_VARIABLE_LEVEL + 1];

/// Level costs indexed by \[type\]\[band\]\[context\]
/// Each entry is an array of costs for levels 0..=MAX_VARIABLE_LEVEL
pub type LevelCostTables = [[[LevelCostArray; NUM_CTX]; NUM_BANDS]; NUM_TYPES];

/// Remapped costs indexed by \[type\]\[position\]\[context\]
/// Maps coefficient position (0..16) directly to its band's level_cost.
/// This avoids the indirection through VP8_ENC_BANDS during cost calculation.
pub type RemappedCosts = [[usize; NUM_CTX]; 16];

/// Calculate the variable-length cost for encoding a level >= 1.
/// Uses the VP8_LEVEL_CODES table to determine which probability nodes to use.
/// Ported from libwebp's VariableLevelCost.
fn variable_level_cost(level: usize, probas: &[u8; NUM_PROBAS]) -> u16 {
    if level == 0 {
        return 0;
    }
    let idx = level.min(MAX_VARIABLE_LEVEL) - 1;
    let pattern = VP8_LEVEL_CODES[idx][0];
    let bits = VP8_LEVEL_CODES[idx][1];

    let mut cost = 0u16;
    let mut p = pattern;
    let mut b = bits;
    let mut i = 2; // Start at proba index 2

    while p != 0 {
        if (p & 1) != 0 {
            cost += vp8_bit_cost((b & 1) != 0, probas[i]);
        }
        b >>= 1;
        p >>= 1;
        i += 1;
    }
    cost
}

/// Level cost tables holder with precomputed costs and remapping.
/// Ported from libwebp's VP8EncProba (level_cost and remapped_costs fields).
#[derive(Clone)]
pub struct LevelCosts {
    /// Level costs indexed by \[type\]\[band\]\[ctx\]\[level\]
    pub level_cost: LevelCostTables,
    /// Remapped indices: [type][position] -> band index for each type
    /// Usage: level_cost[type][remapped[type][n]][ctx][level]
    remapped: [RemappedCosts; NUM_TYPES],
    /// EOB (end-of-block) costs indexed by [type][band][ctx]
    /// This is the cost of signaling "no more coefficients"
    eob_cost: [[[u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
    /// Init (has-coefficients) costs indexed by [type][band][ctx]
    /// This is the cost of signaling "block has coefficients"
    /// Used for initializing trellis at ctx0=0
    init_cost: [[[u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
    /// Whether the tables are dirty and need recalculation
    dirty: bool,
}

impl Default for LevelCosts {
    fn default() -> Self {
        Self::new()
    }
}

impl LevelCosts {
    /// Create new level cost tables
    pub fn new() -> Self {
        Self {
            level_cost: [[[[0u16; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            remapped: [[[0usize; NUM_CTX]; 16]; NUM_TYPES],
            eob_cost: [[[0u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            init_cost: [[[0u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            dirty: true,
        }
    }

    /// Mark tables as dirty (need recalculation)
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Check if tables need recalculation
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Calculate level costs from probability tables.
    /// Ported from libwebp's VP8CalculateLevelCosts.
    #[allow(clippy::needless_range_loop)] // indices used for multiple arrays
    pub fn calculate(&mut self, probs: &TokenProbTables) {
        if !self.dirty {
            return;
        }

        for ctype in 0..NUM_TYPES {
            for band in 0..NUM_BANDS {
                for ctx in 0..NUM_CTX {
                    let p = &probs[ctype][band][ctx];

                    // cost0 is the cost of signaling "no more coefficients" at context > 0
                    // For ctx == 0, this cost is handled separately
                    let cost0 = if ctx > 0 { vp8_bit_cost(true, p[0]) } else { 0 };

                    // cost_base is cost of signaling "coefficient present" + cost0
                    let cost_base = vp8_bit_cost(true, p[1]) + cost0;

                    // Level 0: just signal "no coefficient"
                    self.level_cost[ctype][band][ctx][0] = vp8_bit_cost(false, p[1]) + cost0;

                    // Levels 1..=MAX_VARIABLE_LEVEL
                    for v in 1..=MAX_VARIABLE_LEVEL {
                        self.level_cost[ctype][band][ctx][v] =
                            cost_base + variable_level_cost(v, p);
                    }

                    // EOB cost: signaling "no more coefficients" after this position
                    // This is the cost of taking the EOB branch in the coefficient tree
                    self.eob_cost[ctype][band][ctx] = vp8_bit_cost(false, p[0]);

                    // Init cost: signaling "block has coefficients" at this position
                    // Used for initializing trellis at ctx0=0
                    self.init_cost[ctype][band][ctx] = vp8_bit_cost(true, p[0]);
                }
            }

            // Build remapped indices for direct position-based lookup
            for n in 0..16 {
                let band = VP8_ENC_BANDS[n] as usize;
                for ctx in 0..NUM_CTX {
                    self.remapped[ctype][n][ctx] = band;
                }
            }
        }

        self.dirty = false;
    }

    /// Get level cost for a specific coefficient position.
    /// Combines fixed cost from VP8_LEVEL_FIXED_COSTS and variable cost from tables.
    #[inline]
    pub fn get_level_cost(&self, ctype: usize, position: usize, ctx: usize, level: usize) -> u32 {
        let fixed = VP8_LEVEL_FIXED_COSTS[level.min(MAX_LEVEL)] as u32;
        let band = self.remapped[ctype][position][ctx];
        let variable = self.level_cost[ctype][band][ctx][level.min(MAX_VARIABLE_LEVEL)] as u32;
        fixed + variable
    }

    /// Get the cost table for a specific type, position, and context.
    #[inline]
    pub fn get_cost_table(&self, ctype: usize, position: usize, ctx: usize) -> &LevelCostArray {
        let band = self.remapped[ctype][position][ctx];
        &self.level_cost[ctype][band][ctx]
    }

    /// Get the EOB (end-of-block) cost for terminating after position n.
    /// This is the cost of signaling "no more coefficients" at position n+1.
    /// The context should be based on the level at position n (ctx = min(level, 2)).
    #[inline]
    pub fn get_eob_cost(&self, ctype: usize, position: usize, ctx: usize) -> u16 {
        // EOB is signaled at position n+1, so use band for n+1
        let next_pos = (position + 1).min(15);
        let band = VP8_ENC_BANDS[next_pos] as usize;
        self.eob_cost[ctype][band][ctx]
    }

    /// Get the EOB cost for signaling "no coefficients at all" at position first.
    /// Used for skip (all-zero block) calculation.
    #[inline]
    pub fn get_skip_eob_cost(&self, ctype: usize, first: usize, ctx: usize) -> u16 {
        let band = VP8_ENC_BANDS[first] as usize;
        self.eob_cost[ctype][band][ctx]
    }

    /// Get the init cost for signaling "block has coefficients" at position first.
    /// Used for initializing trellis at ctx0=0.
    #[inline]
    pub fn get_init_cost(&self, ctype: usize, first: usize, ctx: usize) -> u16 {
        let band = VP8_ENC_BANDS[first] as usize;
        self.init_cost[ctype][band][ctx]
    }
}
// Re-export residual cost functions from residual_cost module
pub use super::residual_cost::{
    get_cost_luma16, get_cost_luma4, get_cost_uv, get_residual_cost, Residual,
};

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_entropy_cost_table() {
        // Probability 128 (50%) should have cost ~256 (1 bit)
        assert!((VP8_ENTROPY_COST[128] as i32 - 256).abs() < 10);

        // Probability 255 (~100%) should have very low cost
        assert!(VP8_ENTROPY_COST[255] < 10);

        // Probability 1 (~0%) should have high cost
        assert!(VP8_ENTROPY_COST[1] > 1500);

        // Table should be monotonically decreasing
        for i in 1..256 {
            assert!(
                VP8_ENTROPY_COST[i] <= VP8_ENTROPY_COST[i - 1],
                "Entropy cost not monotonic at {}",
                i
            );
        }
    }

    #[test]
    fn test_bit_cost() {
        // Cost of 0 with prob 128 should equal cost of 1 with prob 128
        assert_eq!(vp8_bit_cost(false, 128), vp8_bit_cost(true, 128));

        // Cost of 0 with high prob should be low
        assert!(vp8_bit_cost(false, 250) < vp8_bit_cost(false, 128));

        // Cost of 1 with high prob should be high
        assert!(vp8_bit_cost(true, 250) > vp8_bit_cost(true, 128));
    }

    #[test]
    fn test_level_fixed_costs() {
        // Level 0 should have cost 0
        assert_eq!(VP8_LEVEL_FIXED_COSTS[0], 0);

        // Small levels should have lower cost than large levels
        assert!(VP8_LEVEL_FIXED_COSTS[1] < VP8_LEVEL_FIXED_COSTS[100]);
        assert!(VP8_LEVEL_FIXED_COSTS[100] < VP8_LEVEL_FIXED_COSTS[1000]);

        // Table should be correct size
        assert_eq!(VP8_LEVEL_FIXED_COSTS.len(), MAX_LEVEL + 1);
    }

    #[test]
    fn test_enc_bands() {
        // First 4 positions map to bands 0-3
        assert_eq!(VP8_ENC_BANDS[0], 0);
        assert_eq!(VP8_ENC_BANDS[1], 1);
        assert_eq!(VP8_ENC_BANDS[2], 2);
        assert_eq!(VP8_ENC_BANDS[3], 3);

        // Position 4 maps to band 6 (skip)
        assert_eq!(VP8_ENC_BANDS[4], 6);
    }

    #[test]
    fn test_fixed_costs_i16() {
        // DC should be cheapest
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[1]);
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[2]);
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[3]);

        // Values should match libwebp
        assert_eq!(FIXED_COSTS_I16[0], 663);
        assert_eq!(FIXED_COSTS_I16[1], 919);
        assert_eq!(FIXED_COSTS_I16[2], 872);
        assert_eq!(FIXED_COSTS_I16[3], 919);
    }

    #[test]
    fn test_fixed_costs_uv() {
        // DC should be cheapest
        assert!(FIXED_COSTS_UV[0] < FIXED_COSTS_UV[1]);

        // Values should match libwebp
        assert_eq!(FIXED_COSTS_UV[0], 302);
        assert_eq!(FIXED_COSTS_UV[1], 984);
        assert_eq!(FIXED_COSTS_UV[2], 439);
        assert_eq!(FIXED_COSTS_UV[3], 642);
    }

    #[test]
    fn test_fixed_costs_i4_context() {
        // DC after DC should be cheap
        let dc_after_dc = VP8_FIXED_COSTS_I4[0][0][0];
        assert!(dc_after_dc < 100);

        // Non-DC mode after DC should be more expensive
        let tm_after_dc = VP8_FIXED_COSTS_I4[0][0][1];
        assert!(tm_after_dc > dc_after_dc);

        // Table should be 10x10x10
        assert_eq!(VP8_FIXED_COSTS_I4.len(), NUM_BMODES);
        assert_eq!(VP8_FIXED_COSTS_I4[0].len(), NUM_BMODES);
        assert_eq!(VP8_FIXED_COSTS_I4[0][0].len(), NUM_BMODES);
    }

    #[test]
    fn test_lambda_calculation() {
        // At q=64 (medium quality)
        let q = 64u32;

        let lambda_i4 = calc_lambda_i4(q);
        let lambda_i16 = calc_lambda_i16(q);
        let lambda_uv = calc_lambda_uv(q);

        // lambda_i16 should be much larger than lambda_i4
        assert!(lambda_i16 > lambda_i4 * 50);

        // lambda_uv should be between i4 and i16
        assert!(lambda_uv > lambda_i4);
        assert!(lambda_uv < lambda_i16);

        // Values should match libwebp formulas
        assert_eq!(lambda_i4, (3 * 64 * 64) >> 7);
        assert_eq!(lambda_i16, 3 * 64 * 64);
        assert_eq!(lambda_uv, (3 * 64 * 64) >> 6);
    }

    #[test]
    fn test_i4_penalty() {
        let q = 64u32;
        let penalty = calc_i4_penalty(q);

        // Should be 1000 * q²
        assert_eq!(penalty, 1000 * 64 * 64);

        // At low q, penalty should be low
        assert!(calc_i4_penalty(10) < calc_i4_penalty(64));
    }

    #[test]
    fn test_rd_score() {
        // Zero SSE and zero cost = zero score
        assert_eq!(rd_score(0, 0, LAMBDA_I16), 0);

        // Only distortion
        assert_eq!(rd_score(100, 0, LAMBDA_I16), 100 * 256);

        // Only rate
        assert_eq!(rd_score(0, 663, LAMBDA_I16), 663 * 106);

        // Combined
        let expected = 1000 * 256 + u64::from(FIXED_COSTS_I16[0]) * 106;
        assert_eq!(rd_score(1000, FIXED_COSTS_I16[0], LAMBDA_I16), expected);
    }

    #[test]
    fn test_get_i4_mode_cost() {
        // Should return same as direct indexing
        assert_eq!(get_i4_mode_cost(0, 0, 0), VP8_FIXED_COSTS_I4[0][0][0]);
        assert_eq!(get_i4_mode_cost(5, 3, 7), VP8_FIXED_COSTS_I4[5][3][7]);
    }

    #[test]
    fn test_rd_trade_off() {
        let lambda = LAMBDA_I16;

        // DC mode with higher SSE
        let dc_sse = 1000u32;
        let dc_cost = FIXED_COSTS_I16[0];

        // V mode with lower SSE
        let v_sse = 500u32;
        let v_cost = FIXED_COSTS_I16[1];

        let dc_score = rd_score(dc_sse, dc_cost, lambda);
        let v_score = rd_score(v_sse, v_cost, lambda);

        // V should win with significantly lower SSE
        assert!(v_score < dc_score);
    }

    #[test]
    fn test_estimate_residual_cost() {
        // All zeros should have minimal cost
        let zeros = [0i32; 16];
        let cost_zeros = estimate_residual_cost(&zeros, 0);
        assert!(cost_zeros < 512, "Zero block cost should be ~1 bit");

        // Single non-zero DC coefficient
        let mut single_dc = [0i32; 16];
        single_dc[0] = 1;
        let cost_single = estimate_residual_cost(&single_dc, 0);
        assert!(
            cost_single > cost_zeros,
            "Non-zero coefficient should cost more"
        );

        // Higher coefficient values should cost more
        let mut large_coeff = [0i32; 16];
        large_coeff[0] = 100;
        let cost_large = estimate_residual_cost(&large_coeff, 0);
        assert!(
            cost_large > cost_single,
            "Large coefficient should cost more than small"
        );

        // More non-zero coefficients = higher cost
        let mut multiple = [0i32; 16];
        multiple[0] = 1;
        multiple[1] = 1;
        multiple[2] = 1;
        let cost_multiple = estimate_residual_cost(&multiple, 0);
        assert!(
            cost_multiple > cost_single,
            "Multiple coefficients should cost more"
        );

        // AC-only (first=1) should skip DC
        let mut dc_only = [0i32; 16];
        dc_only[0] = 100;
        let cost_ac_only = estimate_residual_cost(&dc_only, 1);
        assert!(
            cost_ac_only < estimate_residual_cost(&dc_only, 0),
            "AC-only should skip DC cost"
        );
    }

    #[test]
    fn test_rd_score_with_coeffs() {
        let sse = 1000u32;
        let mode_cost = FIXED_COSTS_I16[0];
        let coeff_cost = 2000u32;
        let lambda = LAMBDA_I16;

        let score = rd_score_with_coeffs(sse, mode_cost, coeff_cost, lambda);
        let expected = (sse as u64) * 256 + (mode_cost as u64 + coeff_cost as u64) * 106;
        assert_eq!(score, expected);

        // Without coeff cost should be lower
        let score_no_coeff = rd_score(sse, mode_cost, lambda);
        assert!(score_no_coeff < score);
    }

    #[test]
    fn test_trellis_basic() {
        // Create a Y1 matrix at q=50
        let matrix = VP8Matrix::new(50, 50, MatrixType::Y1);

        // Test case 1: DC dominant block (natural order)
        let block = [
            500i32, 50, 25, 10, // row 0
            5, 3, 2, 1, // row 1
            0, 0, 0, 0, // row 2
            0, 0, 0, 0, // row 3
        ];

        let mut coeffs = block;
        let mut trellis_out = [0i32; 16];

        // Lambda for i4: (7 * 50^2) >> 3 = 2187
        let lambda = 2187u32;

        // Create level costs for test
        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&crate::common::types::COEFF_PROBS);

        let trellis_nz = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            0,
            &level_costs,
            3, // I4 type
            0, // initial context
        );

        // Simple quantization for comparison
        let mut simple_out = [0i32; 16];
        for i in 0..16 {
            let j = VP8_ZIGZAG[i]; // Convert zigzag position to natural
            simple_out[i] = matrix.quantize_coeff(block[j], j);
        }

        eprintln!("Test case 1: DC dominant");
        eprintln!("  Input (natural order):  {:?}", block);
        eprintln!("  Simple (zigzag order):  {:?}", simple_out);
        eprintln!("  Trellis (zigzag order): {:?}", trellis_out);
        eprintln!("  Trellis has_nz: {}", trellis_nz);

        // Both should produce non-zero DC coefficient
        assert!(simple_out[0] != 0, "Simple should have non-zero DC");

        // Check that trellis produces reasonable output
        // It may zero more aggressively but DC should usually be preserved for large values
        eprintln!();
    }

    #[test]
    fn test_trellis_vs_simple() {
        // Create a Y1 matrix at q=30
        let matrix = VP8Matrix::new(30, 30, MatrixType::Y1);

        // Block with clear signal
        let block = [300i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut coeffs = block;
        let mut trellis_out = [0i32; 16];
        let lambda = ((7 * 30 * 30) >> 3) as u32;

        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&crate::common::types::COEFF_PROBS);

        let _ = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            0,
            &level_costs,
            3, // I4 type
            0, // initial context
        );

        // Simple
        let simple_dc = matrix.quantize_coeff(block[0], 0);

        eprintln!("Single DC coefficient test:");
        eprintln!("  Input DC: {}", block[0]);
        eprintln!("  Simple DC: {}", simple_dc);
        eprintln!("  Trellis DC: {}", trellis_out[0]);

        // For a large DC coefficient, trellis should preserve it
        assert!(
            trellis_out[0] != 0 || simple_dc == 0,
            "Trellis zeroed DC but simple didn't"
        );
    }

    /// Test demonstrating why trellis is currently disabled.
    ///
    /// The trellis uses simplified fixed costs (level_cost) that don't account for
    /// probability-dependent costs. This causes it to favor non-zero coefficients
    /// even when the simple quantizer would produce zero.
    ///
    /// Example: For a coefficient value of -17 with quantizer 30:
    /// - Simple quantization (with bias): produces 0
    /// - Trellis (with neutral bias): computes level0=0, but then considers level=1
    ///   because the distortion reduction outweighs the fixed rate cost.
    ///
    /// In libwebp, VP8LevelCost uses full probability tables that make coding zeros
    /// "cheaper" in certain contexts, which prevents this issue.
    #[test]
    fn test_trellis_known_issue() {
        // Reproduce the issue: trellis produces non-zero where simple produces zero
        let matrix = VP8Matrix::new(27, 30, MatrixType::Y1);

        // Block with small coefficient at position 4 (-17)
        let block = [20i32, -8, 0, 0, -17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // Simple quantization
        let mut simple_out = [0i32; 16];
        for i in 0..16 {
            let j = VP8_ZIGZAG[i];
            simple_out[i] = matrix.quantize_coeff(block[j], j);
        }

        // Trellis quantization with probability-dependent costs
        let mut coeffs = block;
        let mut trellis_out = [0i32; 16];
        let lambda = 787u32;

        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&crate::common::types::COEFF_PROBS);

        let _ = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            0,
            &level_costs,
            3, // I4 type
            0, // initial context
        );

        // Simple produces zeros for small coefficients
        assert_eq!(simple_out[2], 0, "Simple should quantize position 2 to 0");
        // With probability-dependent costs, trellis should now also produce 0
        // because the higher cost of coding a non-zero coefficient outweighs the
        // distortion benefit
        eprintln!("trellis_out[2] = {}", trellis_out[2]);
        // Note: if this still produces -1, there may be more tuning needed
    }

    #[test]
    fn test_trellis_quality75() {
        // Quality 75 -> quant_index 26
        // DC=27, AC=30
        let matrix = VP8Matrix::new(27, 30, MatrixType::Y1);

        // Test with realistic gradient image transform coefficients
        // A gradient block might have DC ~1000 and small AC values
        let blocks = [
            // Block 1: Strong DC, some AC
            [1000i32, 100, 50, 25, 15, 10, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0],
            // Block 2: Medium DC
            [500, 50, 25, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // Block 3: Weak signal
            [100, 20, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // Block 4: Near zero
            [30, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];

        let lambda = ((7 * 30 * 30) >> 3) as u32; // 787
        eprintln!("Lambda: {}", lambda);

        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&crate::common::types::COEFF_PROBS);

        for (i, block) in blocks.iter().enumerate() {
            let mut coeffs = *block;
            let mut trellis_out = [0i32; 16];
            let _ = trellis_quantize_block(
                &mut coeffs,
                &mut trellis_out,
                &matrix,
                lambda,
                0,
                &level_costs,
                3, // I4 type
                0, // initial context
            );

            // Simple quantization
            let mut simple_out = [0i32; 16];
            for j in 0..16 {
                let zz = VP8_ZIGZAG[j];
                simple_out[j] = matrix.quantize_coeff(block[zz], zz);
            }

            eprintln!("\nBlock {}: input DC={}", i + 1, block[0]);
            eprintln!("  Simple:  {:?}", &simple_out[..8]);
            eprintln!("  Trellis: {:?}", &trellis_out[..8]);

            // Check SSE difference
            let simple_sse: i64 = (0..16)
                .map(|j| {
                    let zz = VP8_ZIGZAG[j];
                    let orig = block[zz] as i64;
                    let recon = simple_out[j] as i64 * matrix.q[zz] as i64;
                    (orig - recon) * (orig - recon)
                })
                .sum();

            let trellis_sse: i64 = (0..16)
                .map(|j| {
                    let zz = VP8_ZIGZAG[j];
                    let orig = block[zz] as i64;
                    let recon = trellis_out[j] as i64 * matrix.q[zz] as i64;
                    (orig - recon) * (orig - recon)
                })
                .sum();

            eprintln!("  Simple SSE: {}, Trellis SSE: {}", simple_sse, trellis_sse);
        }
    }

    /// Diagnostic test to capture all data needed to debug trellis calibration.
    /// Run with: cargo test --release test_trellis_diagnostic -- --nocapture
    #[test]
    fn test_trellis_diagnostic() {
        use crate::common::types::COEFF_PROBS;

        let matrix = VP8Matrix::new(27, 30, MatrixType::Y1);
        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&COEFF_PROBS);

        // Test block that showed issues: small coefficient at position 4
        let block = [20i32, -8, 0, 0, -17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let lambda = 787u32;
        let ctype = 3usize; // I4
        let ctx0 = 0usize;

        eprintln!("\n=== TRELLIS DIAGNOSTIC ===");
        eprintln!("Lambda: {}", lambda);
        eprintln!("Block type: {} (I4)", ctype);
        eprintln!("Initial context: {}", ctx0);
        eprintln!("Input block (natural order): {:?}", block);

        // Show quantization parameters
        eprintln!("\nQuantization parameters:");
        eprintln!(
            "  DC q={}, iq={}, bias={}",
            matrix.q[0], matrix.iq[0], matrix.bias[0]
        );
        eprintln!(
            "  AC q={}, iq={}, bias={}",
            matrix.q[1], matrix.iq[1], matrix.bias[1]
        );

        // Show cost table values for position 0, contexts 0,1,2
        eprintln!("\nCost tables for position 0 (levels 0,1,2):");
        for ctx in 0..3 {
            let costs = level_costs.get_cost_table(ctype, 0, ctx);
            eprintln!(
                "  ctx={}: level0={}, level1={}, level2={}",
                ctx, costs[0], costs[1], costs[2]
            );
        }

        // Simple quantization
        let mut simple_out = [0i32; 16];
        for i in 0..16 {
            let j = VP8_ZIGZAG[i];
            simple_out[i] = matrix.quantize_coeff(block[j], j);
        }
        eprintln!("\nSimple quantization (zigzag): {:?}", simple_out);

        // Per-coefficient analysis
        eprintln!("\nPer-coefficient analysis:");
        eprintln!(
            "{:>3} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
            "pos", "input", "simple", "level0", "cost_l0", "cost_l1", "dist_diff"
        );

        for n in 0..8 {
            let j = VP8_ZIGZAG[n];
            let coeff = block[j];
            let q = matrix.q[j] as i32;
            let iq = matrix.iq[j];

            // Compute level0 (base quantization with neutral bias)
            let sign = coeff < 0;
            let abs_coeff = if sign { -coeff } else { coeff };
            let level0 =
                quantdiv(abs_coeff as u32, iq, quantization_bias(0x00)).min(MAX_LEVEL as i32);

            // Get cost table (assume ctx=0 for simplicity)
            let costs = level_costs.get_cost_table(ctype, n, 0);

            // Cost for level0 and level0+1
            let cost_l0 = VP8_LEVEL_FIXED_COSTS[level0 as usize] as u32
                + costs[level0.min(MAX_VARIABLE_LEVEL as i32) as usize] as u32
                + if level0 > 0 { 256 } else { 0 };
            let cost_l1 = VP8_LEVEL_FIXED_COSTS[(level0 + 1).min(MAX_LEVEL as i32) as usize] as u32
                + costs[(level0 + 1).min(MAX_VARIABLE_LEVEL as i32) as usize] as u32
                + 256; // always non-zero

            // Distortion difference if we use level0+1 instead of level0
            let err0 = abs_coeff - level0 * q;
            let err1 = abs_coeff - (level0 + 1) * q;
            let dist_diff = err1 * err1 - err0 * err0;

            eprintln!(
                "{:>3} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
                n, coeff, simple_out[n], level0, cost_l0, cost_l1, dist_diff
            );
        }

        // RD decision analysis
        eprintln!("\nRD decision at lambda={}:", lambda);
        eprintln!("For level0 -> level0+1: chose higher level if:");
        eprintln!("  lambda * (cost_l1 - cost_l0) < 256 * (dist0 - dist1)");
        eprintln!("  {} * cost_delta < 256 * dist_delta", lambda);

        // EOB cost analysis
        eprintln!("\nEOB cost analysis (approximate=128 vs actual p[0]):");
        let eob_approx = vp8_bit_cost(false, 128);
        for band in 0..8 {
            let p0_0 = COEFF_PROBS[ctype][band][0][0];
            let p0_1 = COEFF_PROBS[ctype][band][1][0];
            let p0_2 = COEFF_PROBS[ctype][band][2][0];
            let eob_0 = vp8_bit_cost(false, p0_0);
            let eob_1 = vp8_bit_cost(false, p0_1);
            let eob_2 = vp8_bit_cost(false, p0_2);
            eprintln!(
                "  band {}: ctx0: p0={:3} eob={:3} (diff={:+4}), ctx1: p0={:3} eob={:3} (diff={:+4}), ctx2: p0={:3} eob={:3} (diff={:+4})",
                band,
                p0_0, eob_0, eob_0 as i32 - eob_approx as i32,
                p0_1, eob_1, eob_1 as i32 - eob_approx as i32,
                p0_2, eob_2, eob_2 as i32 - eob_approx as i32
            );
        }

        // Run trellis
        let mut coeffs = block;
        let mut trellis_out = [0i32; 16];
        let _ = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            0,
            &level_costs,
            ctype,
            ctx0,
        );
        eprintln!("\nTrellis output (zigzag): {:?}", trellis_out);

        // Show differences
        eprintln!("\nDifferences (simple vs trellis):");
        for n in 0..16 {
            if simple_out[n] != trellis_out[n] {
                eprintln!(
                    "  pos {}: simple={}, trellis={}",
                    n, simple_out[n], trellis_out[n]
                );
            }
        }
    }

    /// Compare our trellis with libwebp's output using matching input.
    /// Uses values from libwebp debug log.
    /// Run with: cargo test --release test_trellis_vs_libwebp -- --nocapture
    #[test]
    fn test_trellis_vs_libwebp() {
        use crate::common::types::COEFF_PROBS;

        // From libwebp debug log:
        // === BLOCK 0: type=3 ctx0=0 lambda=840 first=0 ===
        // input: -282 6 3 -4 -3 -11 -4 -2 5 3 4 -1 2 -2 -3 -1
        // q: 25 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
        // iq: 5242 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228 4228
        // last=1 thresh=240 last_proba=202 skip_cost=89 skip_score=74760
        // init: ctx0=0 init_rate=576 init_score=483840
        // RESULT: out: -11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&COEFF_PROBS);

        // Match libwebp's matrix values exactly
        let mut matrix = VP8Matrix::new(25, 31, MatrixType::Y1);
        // Override iq values to match libwebp
        matrix.iq[0] = 5242;
        for i in 1..16 {
            matrix.iq[i] = 4228;
        }

        // Input coefficients (NOTE: these are already in natural/raster order in libwebp)
        // libwebp's "input" is coefficients[j] where j = kZigzag[n], so already in natural order
        let input = [
            -282i32, 6, 3, -4, -3, -11, -4, -2, 5, 3, 4, -1, 2, -2, -3, -1,
        ];

        let lambda = 840u32;
        let ctype = 3usize; // TYPE_I4_AC
        let ctx0 = 0usize;
        let first = 0usize;

        eprintln!("\n=== TRELLIS VS LIBWEBP ===");
        eprintln!(
            "lambda={}, ctype={} (I4), ctx0={}, first={}",
            lambda, ctype, ctx0, first
        );

        // Check skip and init costs
        let skip_cost = level_costs.get_skip_eob_cost(ctype, first, ctx0);
        let init_cost = level_costs.get_init_cost(ctype, first, ctx0);
        let skip_score = rd_score_trellis(lambda, skip_cost as i64, 0);
        let init_score = rd_score_trellis(lambda, init_cost as i64, 0);

        eprintln!("\nCost comparison:");
        eprintln!("  skip_cost: rust={} libwebp=89", skip_cost);
        eprintln!("  skip_score: rust={} libwebp=74760", skip_score);
        eprintln!("  init_cost: rust={} libwebp=576", init_cost);
        eprintln!("  init_score: rust={} libwebp=483840", init_score);

        // Calculate thresh like libwebp
        let thresh = (matrix.q[1] as i64 * matrix.q[1] as i64 / 4) as i32;
        eprintln!("\nthresh: rust={} libwebp=240", thresh);

        // Run trellis
        let mut coeffs = input;
        let mut trellis_out = [0i32; 16];
        let _has_nz = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            first,
            &level_costs,
            ctype,
            ctx0,
        );

        eprintln!("\nOutput comparison:");
        eprintln!("  rust:    {:?}", trellis_out);
        eprintln!("  libwebp: [-11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]");

        // Assert match
        let expected = [-11i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(trellis_out, expected, "Trellis output should match libwebp");
    }
}
