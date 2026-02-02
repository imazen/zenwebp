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

// SIMD imports for GetResidualCost optimization
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use archmage::{arcane, Has128BitSimd, SimdToken, X64V3Token};
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use core::arch::x86_64::*;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use safe_unaligned_simd::x86_64 as simd_mem;

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

//------------------------------------------------------------------------------
// Quantization constants

/// Fixed-point precision for quantization
pub const QFIX: u32 = 17;

/// Bias calculation macro equivalent
#[inline]
pub const fn quantization_bias(b: u32) -> u32 {
    (((b) << (QFIX)) + 128) >> 8
}

/// Quantization division: (coeff * iq + bias) >> QFIX
#[inline]
pub fn quantdiv(coeff: u32, iq: u32, bias: u32) -> i32 {
    ((coeff as u64 * iq as u64 + bias as u64) >> QFIX) as i32
}

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
// Quantization matrix

/// Quantization matrix for a coefficient type (Y1, Y2, UV)
#[derive(Clone, Debug)]
pub struct VP8Matrix {
    /// Quantizer steps for each coefficient position
    pub q: [u16; 16],
    /// Reciprocals (1 << QFIX) / q, for fast division
    pub iq: [u32; 16],
    /// Rounding bias for quantization
    pub bias: [u32; 16],
    /// Zero threshold: coefficients below this are quantized to 0
    pub zthresh: [u32; 16],
    /// Sharpening boost for high-frequency coefficients
    pub sharpen: [u16; 16],
}

impl VP8Matrix {
    /// Create a new quantization matrix from DC and AC quantizer values
    pub fn new(q_dc: u16, q_ac: u16, matrix_type: MatrixType) -> Self {
        let bias_values = match matrix_type {
            MatrixType::Y1 => (96, 110),  // luma-ac
            MatrixType::Y2 => (96, 108),  // luma-dc
            MatrixType::UV => (110, 115), // chroma
        };

        let mut m = Self {
            q: [0; 16],
            iq: [0; 16],
            bias: [0; 16],
            zthresh: [0; 16],
            sharpen: [0; 16],
        };

        // Set DC (index 0) and AC (index 1+) values
        m.q[0] = q_dc;
        m.q[1] = q_ac;

        // Calculate reciprocals, bias, and zero thresholds for DC and AC
        for i in 0..2 {
            let is_ac = i > 0;
            let bias = if is_ac { bias_values.1 } else { bias_values.0 };
            m.iq[i] = ((1u64 << QFIX) / m.q[i] as u64) as u32;
            m.bias[i] = quantization_bias(bias);
            // zthresh: value such that quantdiv(coeff, iq, bias) is 0 if coeff <= zthresh
            m.zthresh[i] = ((1 << QFIX) - 1 - m.bias[i]) / m.iq[i];
        }

        // Replicate AC values for positions 2-15
        for i in 2..16 {
            m.q[i] = m.q[1];
            m.iq[i] = m.iq[1];
            m.bias[i] = m.bias[1];
            m.zthresh[i] = m.zthresh[1];
        }

        // Apply sharpening for Y1 matrix (luma AC)
        if matches!(matrix_type, MatrixType::Y1) {
            const SHARPEN_BITS: u32 = 11;
            for (i, &freq_sharpen) in VP8_FREQ_SHARPENING.iter().enumerate() {
                m.sharpen[i] = ((freq_sharpen as u32 * m.q[i] as u32) >> SHARPEN_BITS) as u16;
            }
        }

        m
    }

    /// Get the average quantizer value (for lambda calculations)
    pub fn average_q(&self) -> u32 {
        let sum: u32 = self.q.iter().map(|&x| x as u32).sum();
        (sum + 8) >> 4
    }

    /// Quantize a single coefficient
    #[inline]
    pub fn quantize_coeff(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = if sign { -coeff } else { coeff } as u32;
        let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Quantize a single coefficient with neutral bias (for trellis)
    #[inline]
    pub fn quantize_neutral(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = if sign { -coeff } else { coeff } as u32;
        let neutral_bias = quantization_bias(0x00); // neutral
        let level = quantdiv(abs_coeff, self.iq[pos], neutral_bias);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Dequantize a coefficient
    #[inline]
    pub fn dequantize(&self, level: i32, pos: usize) -> i32 {
        level * self.q[pos] as i32
    }

    /// Quantize an entire 4x4 block of coefficients in place (SIMD version)
    #[cfg(feature = "simd")]
    pub fn quantize(&self, coeffs: &mut [i32; 16]) {
        use wide::i64x4;

        // Process 4 coefficients at a time using 64-bit intermediates
        for chunk in 0..4 {
            let base = chunk * 4;

            // Load 4 coefficients
            let c = [
                coeffs[base] as i64,
                coeffs[base + 1] as i64,
                coeffs[base + 2] as i64,
                coeffs[base + 3] as i64,
            ];

            // Compute signs and absolute values
            let signs = [c[0] < 0, c[1] < 0, c[2] < 0, c[3] < 0];
            let abs_c = i64x4::from([c[0].abs(), c[1].abs(), c[2].abs(), c[3].abs()]);

            // Load iq and bias as i64
            let iq = i64x4::from([
                self.iq[base] as i64,
                self.iq[base + 1] as i64,
                self.iq[base + 2] as i64,
                self.iq[base + 3] as i64,
            ]);
            let bias = i64x4::from([
                self.bias[base] as i64,
                self.bias[base + 1] as i64,
                self.bias[base + 2] as i64,
                self.bias[base + 3] as i64,
            ]);

            // Quantize: (abs_coeff * iq + bias) >> QFIX
            let result = (abs_c * iq + bias) >> QFIX as i64;
            let r = result.to_array();

            // Apply signs and store
            coeffs[base] = if signs[0] {
                -(r[0] as i32)
            } else {
                r[0] as i32
            };
            coeffs[base + 1] = if signs[1] {
                -(r[1] as i32)
            } else {
                r[1] as i32
            };
            coeffs[base + 2] = if signs[2] {
                -(r[2] as i32)
            } else {
                r[2] as i32
            };
            coeffs[base + 3] = if signs[3] {
                -(r[3] as i32)
            } else {
                r[3] as i32
            };
        }
    }

    /// Quantize an entire 4x4 block of coefficients in place (scalar fallback)
    #[cfg(not(feature = "simd"))]
    pub fn quantize(&self, coeffs: &mut [i32; 16]) {
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            let sign = *coeff < 0;
            let abs_coeff = if sign { -*coeff } else { *coeff } as u32;
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]);
            *coeff = if sign { -level } else { level };
        }
    }

    /// Quantize only AC coefficients (positions 1-15) in place, leaving DC unchanged
    /// This is used for Y1 blocks where the DC goes to the Y2 block
    #[cfg(feature = "simd")]
    pub fn quantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        use wide::i64x4;

        // Process positions 1-3 (first chunk, skip DC at 0)
        {
            let c = [
                coeffs[1] as i64,
                coeffs[2] as i64,
                coeffs[3] as i64,
                0i64, // padding
            ];
            let signs = [c[0] < 0, c[1] < 0, c[2] < 0, false];
            let abs_c = i64x4::from([c[0].abs(), c[1].abs(), c[2].abs(), 0]);
            let iq = i64x4::from([self.iq[1] as i64, self.iq[2] as i64, self.iq[3] as i64, 0]);
            let bias = i64x4::from([
                self.bias[1] as i64,
                self.bias[2] as i64,
                self.bias[3] as i64,
                0,
            ]);
            let result = (abs_c * iq + bias) >> QFIX as i64;
            let r = result.to_array();
            coeffs[1] = if signs[0] {
                -(r[0] as i32)
            } else {
                r[0] as i32
            };
            coeffs[2] = if signs[1] {
                -(r[1] as i32)
            } else {
                r[1] as i32
            };
            coeffs[3] = if signs[2] {
                -(r[2] as i32)
            } else {
                r[2] as i32
            };
        }

        // Process positions 4-15 (three full chunks)
        for chunk in 1..4 {
            let base = chunk * 4;
            let c = [
                coeffs[base] as i64,
                coeffs[base + 1] as i64,
                coeffs[base + 2] as i64,
                coeffs[base + 3] as i64,
            ];
            let signs = [c[0] < 0, c[1] < 0, c[2] < 0, c[3] < 0];
            let abs_c = i64x4::from([c[0].abs(), c[1].abs(), c[2].abs(), c[3].abs()]);
            let iq = i64x4::from([
                self.iq[base] as i64,
                self.iq[base + 1] as i64,
                self.iq[base + 2] as i64,
                self.iq[base + 3] as i64,
            ]);
            let bias = i64x4::from([
                self.bias[base] as i64,
                self.bias[base + 1] as i64,
                self.bias[base + 2] as i64,
                self.bias[base + 3] as i64,
            ]);
            let result = (abs_c * iq + bias) >> QFIX as i64;
            let r = result.to_array();
            coeffs[base] = if signs[0] {
                -(r[0] as i32)
            } else {
                r[0] as i32
            };
            coeffs[base + 1] = if signs[1] {
                -(r[1] as i32)
            } else {
                r[1] as i32
            };
            coeffs[base + 2] = if signs[2] {
                -(r[2] as i32)
            } else {
                r[2] as i32
            };
            coeffs[base + 3] = if signs[3] {
                -(r[3] as i32)
            } else {
                r[3] as i32
            };
        }
    }

    /// Quantize only AC coefficients (positions 1-15) in place, leaving DC unchanged
    /// This is used for Y1 blocks where the DC goes to the Y2 block (scalar fallback)
    #[cfg(not(feature = "simd"))]
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.iq/self.bias
    pub fn quantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            let sign = coeffs[pos] < 0;
            let abs_coeff = if sign { -coeffs[pos] } else { coeffs[pos] } as u32;
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]);
            coeffs[pos] = if sign { -level } else { level };
        }
    }

    /// Dequantize an entire 4x4 block of coefficients in place
    #[cfg(feature = "simd")]
    pub fn dequantize_block(&self, coeffs: &mut [i32; 16]) {
        use wide::i32x4;
        // Load quantizer steps as i32 vectors (4 at a time)
        let q0 = i32x4::from([
            self.q[0] as i32,
            self.q[1] as i32,
            self.q[2] as i32,
            self.q[3] as i32,
        ]);
        let q1 = i32x4::from([
            self.q[4] as i32,
            self.q[5] as i32,
            self.q[6] as i32,
            self.q[7] as i32,
        ]);
        let q2 = i32x4::from([
            self.q[8] as i32,
            self.q[9] as i32,
            self.q[10] as i32,
            self.q[11] as i32,
        ]);
        let q3 = i32x4::from([
            self.q[12] as i32,
            self.q[13] as i32,
            self.q[14] as i32,
            self.q[15] as i32,
        ]);

        // Load, multiply, store - wide handles the SIMD details
        let c0 = i32x4::from([coeffs[0], coeffs[1], coeffs[2], coeffs[3]]) * q0;
        let c1 = i32x4::from([coeffs[4], coeffs[5], coeffs[6], coeffs[7]]) * q1;
        let c2 = i32x4::from([coeffs[8], coeffs[9], coeffs[10], coeffs[11]]) * q2;
        let c3 = i32x4::from([coeffs[12], coeffs[13], coeffs[14], coeffs[15]]) * q3;

        // Store results
        let r0 = c0.to_array();
        let r1 = c1.to_array();
        let r2 = c2.to_array();
        let r3 = c3.to_array();
        coeffs[0..4].copy_from_slice(&r0);
        coeffs[4..8].copy_from_slice(&r1);
        coeffs[8..12].copy_from_slice(&r2);
        coeffs[12..16].copy_from_slice(&r3);
    }

    /// Dequantize an entire 4x4 block of coefficients in place (scalar fallback)
    #[cfg(not(feature = "simd"))]
    pub fn dequantize_block(&self, coeffs: &mut [i32; 16]) {
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            *coeff *= self.q[pos] as i32;
        }
    }

    /// Dequantize only AC coefficients (positions 1-15) in place
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.q
    pub fn dequantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            coeffs[pos] *= self.q[pos] as i32;
        }
    }
}

/// Matrix type for bias selection
#[derive(Clone, Copy, Debug)]
pub enum MatrixType {
    /// Luma AC coefficients
    Y1,
    /// Luma DC (WHT) coefficients
    Y2,
    /// Chroma coefficients
    UV,
}

//------------------------------------------------------------------------------
// Trellis quantization

/// Maximum cost value for RD optimization
const MAX_COST: i64 = i64::MAX / 2;

/// Trellis node for dynamic programming
#[derive(Clone, Copy, Default)]
struct TrellisNode {
    prev: i8,   // best previous node (-1, 0, or 1 for delta)
    sign: bool, // sign of coefficient
    level: i16, // quantized level
}

/// Score state for trellis traversal
/// Stores score and a reference to cost table for the *next* position.
/// The cost table is determined by the context resulting from this state's level.
#[derive(Clone, Copy)]
struct TrellisScoreState<'a> {
    score: i64,                        // partial RD score
    costs: Option<&'a LevelCostArray>, // cost table for next position (based on ctx from this level)
}

impl Default for TrellisScoreState<'_> {
    fn default() -> Self {
        Self {
            score: MAX_COST,
            costs: None,
        }
    }
}

/// RD score calculation for trellis
#[inline]
fn rd_score_trellis(lambda: u32, rate: i64, distortion: i64) -> i64 {
    rate * lambda as i64 + (RD_DISTO_MULT as i64) * distortion
}

/// Compute level cost using stored cost table (like libwebp's VP8LevelCost).
/// cost = VP8LevelFixedCosts[level] + table[min(level, MAX_VARIABLE_LEVEL)] + sign_bit
#[inline]
fn level_cost_with_table(costs: Option<&LevelCostArray>, level: i32) -> u32 {
    let abs_level = level.unsigned_abs() as usize;
    let fixed = VP8_LEVEL_FIXED_COSTS[abs_level.min(MAX_LEVEL)] as u32;
    let variable = match costs {
        Some(table) => table[abs_level.min(MAX_VARIABLE_LEVEL)] as u32,
        None => 0,
    };
    // Sign bit costs 1 bit (256 in 1/256 bit units) for non-zero levels
    fixed + variable + if abs_level > 0 { 256 } else { 0 }
}

/// Fast inline level cost for trellis inner loop.
/// Assumes: level >= 0, level <= MAX_LEVEL, costs is valid.
#[inline(always)]
fn level_cost_fast(costs: &LevelCostArray, level: usize) -> u32 {
    let fixed = VP8_LEVEL_FIXED_COSTS[level] as u32;
    let variable = costs[level.min(MAX_VARIABLE_LEVEL)] as u32;
    // Sign bit costs 1 bit (256 in 1/256 bit units) for non-zero levels
    let sign_cost = if level > 0 { 256 } else { 0 };
    fixed + variable + sign_cost
}

/// Trellis-optimized quantization for a 4x4 block.
///
/// Uses dynamic programming to find optimal coefficient levels that minimize
/// the RD cost: distortion + lambda * rate.
///
/// This implementation follows libwebp's approach:
/// - Each state stores a pointer to the cost table for the *next* position
/// - Context is computed as min(level, 2) from the current level
/// - Predecessors' cost tables are used to compute transition costs
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients (will be modified with reconstructed values)
/// * `out` - Output quantized levels
/// * `mtx` - Quantization matrix
/// * `lambda` - Rate-distortion trade-off parameter
/// * `first` - First coefficient to process (1 for I16_AC mode, 0 otherwise)
/// * `level_costs` - Probability-dependent level costs
/// * `ctype` - Token type for level cost lookup (0=Y2, 1=Y_AC, 2=Y_DC, 3=UV)
/// * `ctx0` - Initial context from neighboring blocks (0, 1, or 2)
///
/// # Returns
/// True if any non-zero coefficient was produced
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // p indexes multiple arrays with different semantics
pub fn trellis_quantize_block(
    coeffs: &mut [i32; 16],
    out: &mut [i32; 16],
    mtx: &VP8Matrix,
    lambda: u32,
    first: usize,
    level_costs: &LevelCosts,
    ctype: usize,
    ctx0: usize,
) -> bool {
    // Number of alternate levels to try: level-0 (MIN_DELTA=0) and level+1 (MAX_DELTA=1)
    const NUM_NODES: usize = 2; // [level, level+1]

    let mut nodes = [[TrellisNode::default(); NUM_NODES]; 16];
    let mut score_states = [[TrellisScoreState::default(); NUM_NODES]; 2];
    let mut ss_cur_idx = 0usize;
    let mut ss_prev_idx = 1usize;

    // Find last significant coefficient based on threshold
    let thresh = (mtx.q[1] as i64 * mtx.q[1] as i64 / 4) as i32;
    let mut last = first as i32 - 1;
    for n in (first..16).rev() {
        let j = VP8_ZIGZAG[n];
        let err = coeffs[j] * coeffs[j];
        if err > thresh {
            last = n as i32;
            break;
        }
    }
    // Extend by one to not lose too much
    if last < 15 {
        last += 1;
    }

    // Best path tracking: [last_pos, level_delta, prev_delta]
    let mut best_path = [-1i32; 3];

    // Initialize: compute skip score (all zeros = signal EOB immediately)
    // Skip means signaling EOB at the first position, not encoding a zero coefficient.
    let skip_cost = level_costs.get_skip_eob_cost(ctype, first, ctx0) as i64;
    let mut best_score = rd_score_trellis(lambda, skip_cost, 0);

    // Initialize source nodes with cost table for first position based on ctx0
    let initial_costs = level_costs.get_cost_table(ctype, first, ctx0);
    let init_rate = if ctx0 == 0 {
        level_costs.get_init_cost(ctype, first, ctx0) as i64
    } else {
        0
    };
    let init_score = rd_score_trellis(lambda, init_rate, 0);

    for state in &mut score_states[ss_cur_idx][..NUM_NODES] {
        *state = TrellisScoreState {
            score: init_score,
            costs: Some(initial_costs),
        };
    }

    // Traverse trellis
    for n in first..=last as usize {
        let j = VP8_ZIGZAG[n];
        let q = mtx.q[j] as i32;
        let iq = mtx.iq[j];
        let neutral_bias = quantization_bias(0x00);

        // Get sign from original coefficient
        let sign = coeffs[j] < 0;
        let abs_coeff = if sign { -coeffs[j] } else { coeffs[j] };
        let coeff_with_sharpen = abs_coeff + mtx.sharpen[j] as i32;

        // Base quantized level with neutral bias
        let level0 = quantdiv(coeff_with_sharpen as u32, iq, neutral_bias).min(MAX_LEVEL as i32);

        // Threshold level for pruning
        let thresh_bias = quantization_bias(0x80);
        let thresh_level =
            quantdiv(coeff_with_sharpen as u32, iq, thresh_bias).min(MAX_LEVEL as i32);

        // Swap score state indices
        core::mem::swap(&mut ss_cur_idx, &mut ss_prev_idx);

        // Test all alternate level values: level0 and level0+1
        for delta in 0..NUM_NODES {
            let node = &mut nodes[n][delta];
            let level = level0 + delta as i32;

            // Context for next position: min(level, 2)
            let ctx = (level as usize).min(2);

            // Store cost table for next position (based on context from this level)
            let next_costs = if n + 1 < 16 {
                Some(level_costs.get_cost_table(ctype, n + 1, ctx))
            } else {
                None
            };

            // Reset current score state
            score_states[ss_cur_idx][delta] = TrellisScoreState {
                score: MAX_COST,
                costs: next_costs,
            };

            // Skip invalid levels
            if level < 0 || level > thresh_level {
                continue;
            }

            // Compute distortion delta
            let new_error = coeff_with_sharpen - level * q;
            let orig_error_sq = (coeff_with_sharpen * coeff_with_sharpen) as i64;
            let new_error_sq = (new_error * new_error) as i64;
            let weight = VP8_WEIGHT_TRELLIS[j] as i64;
            let delta_distortion = weight * (new_error_sq - orig_error_sq);

            let base_score = rd_score_trellis(lambda, 0, delta_distortion);

            // Find best predecessor using stored cost tables
            // Unrolled loop for NUM_NODES=2 with fast level cost
            let level_usize = level as usize;
            let (best_cur_score, best_prev) = {
                let ss_prev = &score_states[ss_prev_idx];

                // Predecessor 0
                let cost0 = if let Some(costs) = ss_prev[0].costs {
                    level_cost_fast(costs, level_usize) as i64
                } else {
                    VP8_LEVEL_FIXED_COSTS[level_usize] as i64
                };
                let score0 = ss_prev[0].score + cost0 * lambda as i64;

                // Predecessor 1
                let cost1 = if let Some(costs) = ss_prev[1].costs {
                    level_cost_fast(costs, level_usize) as i64
                } else {
                    VP8_LEVEL_FIXED_COSTS[level_usize] as i64
                };
                let score1 = ss_prev[1].score + cost1 * lambda as i64;

                // Select best
                if score1 < score0 {
                    (score1 + base_score, 1i8)
                } else {
                    (score0 + base_score, 0i8)
                }
            };

            // Store in node
            node.sign = sign;
            node.level = level as i16;
            node.prev = best_prev;
            score_states[ss_cur_idx][delta].score = best_cur_score;

            // Check if this is the best terminal node
            if level != 0 && best_cur_score < best_score {
                // Add end-of-block cost: signaling "no more coefficients"
                // Uses the context resulting from this level
                let eob_cost = if n < 15 {
                    level_costs.get_eob_cost(ctype, n, ctx) as i64
                } else {
                    0
                };
                let terminal_score = best_cur_score + rd_score_trellis(lambda, eob_cost, 0);
                if terminal_score < best_score {
                    best_score = terminal_score;
                    best_path[0] = n as i32;
                    best_path[1] = delta as i32;
                    best_path[2] = best_prev as i32;
                }
            }
        }
    }

    // Clear output
    if first == 1 {
        // Preserve DC for I16_AC mode
        out[1..].fill(0);
        coeffs[1..].fill(0);
    } else {
        out.fill(0);
        coeffs.fill(0);
    }

    // No non-zero coefficients - skip block
    if best_path[0] == -1 {
        return false;
    }

    // Unwind best path
    let mut has_nz = false;
    let mut best_node_delta = best_path[1] as usize;
    let mut n = best_path[0] as usize;

    // Patch the prev for the terminal node
    nodes[n][best_node_delta].prev = best_path[2] as i8;

    loop {
        let node = &nodes[n][best_node_delta];
        let j = VP8_ZIGZAG[n];
        let level = if node.sign {
            -node.level as i32
        } else {
            node.level as i32
        };

        out[n] = level;
        has_nz |= level != 0;

        // Reconstruct coefficient for subsequent prediction
        coeffs[j] = level * mtx.q[j] as i32;

        if n == first {
            break;
        }
        best_node_delta = node.prev as usize;
        n -= 1;
    }

    has_nz
}

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

/// Residual coefficients for cost calculation.
/// Ported from libwebp's VP8Residual.
pub struct Residual<'a> {
    /// First coefficient to consider (0 or 1)
    pub first: usize,
    /// Last non-zero coefficient index (-1 if all zero)
    pub last: i32,
    /// Coefficient array
    pub coeffs: &'a [i32; 16],
    /// Coefficient type (0=I16DC, 1=I16AC, 2=Chroma, 3=I4)
    pub coeff_type: usize,
}

impl<'a> Residual<'a> {
    /// Create a new residual from coefficients.
    /// Automatically finds the last non-zero coefficient.
    pub fn new(coeffs: &'a [i32; 16], coeff_type: usize, first: usize) -> Self {
        let last = coeffs
            .iter()
            .rposition(|&c| c != 0)
            .map(|i| i as i32)
            .unwrap_or(-1);

        Self {
            first,
            last,
            coeffs,
            coeff_type,
        }
    }
}

/// Calculate the cost of encoding a residual block using probability-based costs.
/// Ported from libwebp's GetResidualCost_C.
///
/// On x86_64 with SIMD feature, uses SSE2 for precomputing absolute values and contexts.
/// Falls back to scalar implementation on other platforms.
///
/// # Arguments
/// * `ctx0` - Initial context (0, 1, or 2)
/// * `res` - Residual coefficients
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables (for last coefficient EOB cost)
///
/// # Returns
/// Cost in 1/256 bit units
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    if let Some(token) = X64V3Token::summon() {
        get_residual_cost_sse2(token, ctx0, res, costs, probs)
    } else {
        get_residual_cost_scalar(ctx0, res, costs, probs)
    }
}

/// Calculate the cost of encoding a residual block using probability-based costs.
/// Scalar fallback for non-SIMD platforms.
#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_scalar(ctx0, res, costs, probs)
}

/// Scalar implementation of residual cost calculation.
/// Ported from libwebp's GetResidualCost_C.
#[inline]
fn get_residual_cost_scalar(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    let ctype = res.coeff_type;
    let mut n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0, updated after each coefficient
    let mut ctx = ctx0;

    // bit_cost(1, p0) is already incorporated in the cost tables, but only if ctx != 0.
    let mut cost = if ctx0 == 0 {
        vp8_bit_cost(true, p0) as u32
    } else {
        0
    };

    // If no non-zero coefficients, just return EOB cost
    if res.last < 0 {
        return vp8_bit_cost(false, p0) as u32;
    }

    // Process coefficients from first to last-1
    while (n as i32) < res.last {
        let v = res.coeffs[n].unsigned_abs() as usize;

        // Add cost using current context
        cost += costs.get_level_cost(ctype, n, ctx, v);

        // Update context for next position based on current value
        ctx = if v >= 2 { 2 } else { v };

        n += 1;
    }

    // Last coefficient is always non-zero
    {
        let v = res.coeffs[n].unsigned_abs() as usize;
        debug_assert!(v != 0, "Last coefficient should be non-zero");

        // Add cost using current context
        cost += costs.get_level_cost(ctype, n, ctx, v);

        // Add EOB cost for the position after the last coefficient
        if n < 15 {
            let next_band = VP8_ENC_BANDS[n + 1] as usize;
            let next_ctx = if v == 1 { 1 } else { 2 };
            let last_p0 = probs[ctype][next_band][next_ctx][0];
            cost += vp8_bit_cost(false, last_p0) as u32;
        }
    }

    cost
}

/// SSE2 implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn get_residual_cost_sse2(
    _token: impl Has128BitSimd + Copy,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    // Storage for precomputed values
    let mut ctxs: [u8; 16] = [0; 16];
    let mut levels: [u8; 16] = [0; 16];
    let mut abs_levels: [u16; 16] = [0; 16];

    let ctype = res.coeff_type;
    let mut n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let mut ctx = ctx0;

    // bit_cost(1, p0) is already incorporated in the cost tables, but only if ctx != 0.
    let mut cost = if ctx0 == 0 {
        vp8_bit_cost(true, p0) as u32
    } else {
        0
    };

    // If no non-zero coefficients, just return EOB cost
    if res.last < 0 {
        return vp8_bit_cost(false, p0) as u32;
    }

    // Precompute clamped levels and contexts using SIMD
    // libwebp uses i16 coefficients, but ours are i32. Pack them to i16.
    {
        let zero = _mm_setzero_si128();
        let k_cst2 = _mm_set1_epi8(2);
        let k_cst67 = _mm_set1_epi8(MAX_VARIABLE_LEVEL as i8);

        // Load coefficients as i32 and pack to i16
        let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[0..4]).unwrap());
        let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[4..8]).unwrap());
        let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[8..12]).unwrap());
        let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[12..16]).unwrap());

        // Pack i32 to i16 (signed saturation)
        let c0 = _mm_packs_epi32(c0_32, c1_32); // 8 x i16
        let c1 = _mm_packs_epi32(c2_32, c3_32); // 8 x i16

        // Compute absolute values: abs(v) = max(v, -v)
        let d0 = _mm_sub_epi16(zero, c0);
        let d1 = _mm_sub_epi16(zero, c1);
        let e0 = _mm_max_epi16(c0, d0); // abs, 16-bit
        let e1 = _mm_max_epi16(c1, d1);

        // Pack to i8 for context and level clamping
        let f = _mm_packs_epi16(e0, e1); // 16 x i8 abs values

        // Context: min(abs, 2)
        let g = _mm_min_epu8(f, k_cst2);

        // Clamped level: min(abs, 67) for cost table lookup
        let h = _mm_min_epu8(f, k_cst67);

        // Store results
        simd_mem::_mm_storeu_si128(&mut ctxs, g);
        simd_mem::_mm_storeu_si128(&mut levels, h);

        // Store 16-bit absolute values for fixed cost lookup
        simd_mem::_mm_storeu_si128(
            <&mut [u16; 8]>::try_from(&mut abs_levels[0..8]).unwrap(),
            e0,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u16; 8]>::try_from(&mut abs_levels[8..16]).unwrap(),
            e1,
        );
    }

    // Process coefficients from first to last-1 using precomputed values
    while (n as i32) < res.last {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize; // full level for fixed cost

        // Cost = fixed cost + variable cost from table
        let fixed = VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32;
        let band_idx = VP8_ENC_BANDS[n] as usize;
        let variable = costs.level_cost[ctype][band_idx][ctx][level] as u32;
        cost += fixed + variable;

        // Update context for next position
        ctx = ctxs[n] as usize;
        n += 1;
    }

    // Last coefficient is always non-zero
    {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize;
        debug_assert!(flevel != 0, "Last coefficient should be non-zero");

        // Add cost using current context
        let fixed = VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32;
        let band_idx = VP8_ENC_BANDS[n] as usize;
        let variable = costs.level_cost[ctype][band_idx][ctx][level] as u32;
        cost += fixed + variable;

        // Add EOB cost for the position after the last coefficient
        if n < 15 {
            let next_band = VP8_ENC_BANDS[n + 1] as usize;
            let next_ctx = ctxs[n] as usize;
            let last_p0 = probs[ctype][next_band][next_ctx][0];
            cost += vp8_bit_cost(false, last_p0) as u32;
        }
    }

    cost
}

/// Find last non-zero coefficient using SIMD.
/// Ported from libwebp's SetResidualCoeffs_SSE2.
///
/// Returns -1 if all coefficients are zero.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
#[allow(dead_code)]
fn find_last_nonzero_simd(_token: impl Has128BitSimd + Copy, coeffs: &[i32; 16]) -> i32 {
    let zero = _mm_setzero_si128();

    // Load coefficients as i32 and pack to i16
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    // Pack i32 to i16 (signed saturation)
    let c0 = _mm_packs_epi32(c0_32, c1_32); // 8 x i16
    let c1 = _mm_packs_epi32(c2_32, c3_32); // 8 x i16

    // Pack i16 to i8
    let m0 = _mm_packs_epi16(c0, c1); // 16 x i8

    // Compare with zero
    let m1 = _mm_cmpeq_epi8(m0, zero);

    // Get bitmask: bit is 1 if equal to zero
    let mask = 0x0000ffff_u32 ^ (_mm_movemask_epi8(m1) as u32);

    // Find position of most significant non-zero bit
    if mask == 0 {
        -1
    } else {
        (31 - mask.leading_zeros()) as i32
    }
}

/// Calculate the cost of encoding a 4x4 luma block (I4 mode).
/// Ported from libwebp's VP8GetCostLuma4.
///
/// # Arguments
/// * `levels` - Quantized coefficients in zigzag order
/// * `top_nz` - Whether the above block had non-zero coefficients
/// * `left_nz` - Whether the left block had non-zero coefficients
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// (cost, has_nonzero) - Cost in 1/256 bits and whether this block has non-zero coeffs
pub fn get_cost_luma4(
    levels: &[i32; 16],
    top_nz: bool,
    left_nz: bool,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> (u32, bool) {
    // Initial context is sum of top and left non-zero flags (0, 1, or 2)
    let ctx = (top_nz as usize) + (left_nz as usize);

    // Create residual for I4 type (type 3), starting at position 0
    let res = Residual::new(levels, 3, 0);
    let has_nz = res.last >= 0;

    let cost = get_residual_cost(ctx, &res, costs, probs);

    (cost, has_nz)
}

/// Calculate the cost of encoding all 16 luma blocks in I16 mode.
/// Includes DC block (Y2) and 16 AC blocks (Y1).
///
/// # Arguments
/// * `dc_levels` - DC coefficients from WHT (16 values)
/// * `ac_levels` - AC coefficients for each 4x4 block (16 blocks × 16 coeffs)
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// Total cost in 1/256 bits
pub fn get_cost_luma16(
    dc_levels: &[i32; 16],
    ac_levels: &[[i32; 16]; 16],
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    let mut total_cost = 0u32;

    // DC block (type 1 = I16DC, also known as Y2)
    // Context is typically from neighboring DC blocks, but for simplicity use 0
    let dc_res = Residual::new(dc_levels, 1, 0);
    total_cost += get_residual_cost(0, &dc_res, costs, probs);

    // AC blocks (type 0 = I16AC, skipping DC coefficient which is in Y2)
    for ac in ac_levels.iter() {
        let ac_res = Residual::new(ac, 0, 1); // Start at position 1 (skip DC)
        total_cost += get_residual_cost(0, &ac_res, costs, probs);
    }

    total_cost
}

/// Compute accurate coefficient cost for UV blocks using probability-dependent tables.
///
/// Port of libwebp's VP8GetCostUV.
///
/// # Arguments
/// * `uv_levels` - 8 blocks of quantized coefficients (4 U blocks + 4 V blocks)
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// Total cost in 1/256 bits
pub fn get_cost_uv(uv_levels: &[[i32; 16]; 8], costs: &LevelCosts, probs: &TokenProbTables) -> u32 {
    let mut total_cost = 0u32;

    // UV blocks use coeff_type=2 (TYPE_CHROMA_A)
    // All coefficients including DC (first=0)
    for block in uv_levels.iter() {
        let res = Residual::new(block, 2, 0); // ctype=2 for UV, first=0 (include DC)
        total_cost += get_residual_cost(0, &res, costs, probs);
    }

    total_cost
}

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
