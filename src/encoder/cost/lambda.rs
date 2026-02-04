//! Lambda calculation for rate-distortion optimization.
//!
//! Lambda values control the rate-distortion trade-off in mode selection.
//! Higher lambda = prefer lower rate (smaller files).
//! Lower lambda = prefer lower distortion (better quality).
//!
//! These formulas are ported from libwebp's RefineUsingDistortion.

#![allow(dead_code)]

use super::super::tables::{LEVELS_FROM_DELTA, MAX_DELTA_SIZE, VP8_AC_TABLE};

//------------------------------------------------------------------------------
// Fixed lambda constants (from libwebp)

/// Fixed lambda for Intra16 mode selection (distortion method)
pub const LAMBDA_I16: u32 = 106;

/// Fixed lambda for Intra4 mode selection (distortion method)
pub const LAMBDA_I4: u32 = 11;

/// Fixed lambda for UV mode selection (distortion method)
pub const LAMBDA_UV: u32 = 120;

/// Default SNS strength value (matches libwebp default)
pub const DEFAULT_SNS_STRENGTH: u32 = 50;

//------------------------------------------------------------------------------
// Dynamic lambda calculation (based on quantizer)

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

//------------------------------------------------------------------------------
// Filter level calculation

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
