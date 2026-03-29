//! Residual coefficient cost estimation using probability-dependent tables.
//!
//! Contains SIMD-optimized GetResidualCost and block-level cost functions
//! (get_cost_luma4, get_cost_luma16, get_cost_uv).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

//------------------------------------------------------------------------------
// Fixed-size array splitting helpers (zero-cost, all checks elided at compile time)

/// Split `&[T; 16]` into four `&[T; 4]` without runtime bounds checks.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn split4_ref<T>(arr: &[T; 16]) -> (&[T; 4], &[T; 4], &[T; 4], &[T; 4]) {
    let (a, rest) = arr.split_first_chunk::<4>().unwrap();
    let rest: &[T; 12] = rest.try_into().unwrap();
    let (b, rest) = rest.split_first_chunk::<4>().unwrap();
    let rest: &[T; 8] = rest.try_into().unwrap();
    let (c, d) = rest.split_first_chunk::<4>().unwrap();
    let d: &[T; 4] = d.try_into().unwrap();
    (a, b, c, d)
}

/// Split `&mut [T; 16]` into two `&mut [T; 8]` without runtime bounds checks.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn split2_mut<T>(arr: &mut [T; 16]) -> (&mut [T; 8], &mut [T; 8]) {
    let (a, b) = arr.split_first_chunk_mut::<8>().unwrap();
    let b: &mut [T; 8] = b.try_into().unwrap();
    (a, b)
}

// SIMD imports for GetResidualCost optimization
#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use archmage::{Wasm128Token, arcane, rite};
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[cfg(target_arch = "aarch64")]
use archmage::intrinsics::aarch64 as simd_mem;
#[cfg(target_arch = "aarch64")]
use archmage::{NeonToken, SimdToken, arcane, rite};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::cost::{LevelCosts, vp8_bit_cost};
use super::tables::{MAX_LEVEL, MAX_VARIABLE_LEVEL, VP8_ENC_BANDS, VP8_LEVEL_FIXED_COSTS};
use crate::common::types::TokenProbTables;

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
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    if let Some(token) = X64V3Token::summon() {
        get_residual_cost_entry(token, ctx0, res, costs, probs)
    } else {
        get_residual_cost_scalar(ctx0, res, costs, probs)
    }
}

/// WASM SIMD128 dispatch for residual cost calculation.
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    use archmage::SimdToken;
    if let Some(token) = Wasm128Token::summon() {
        return get_residual_cost_wasm_entry(token, ctx0, res, costs, probs);
    }
    get_residual_cost_scalar(ctx0, res, costs, probs)
}

/// NEON dispatch for residual cost calculation.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    let token = NeonToken::summon().unwrap();
    get_residual_cost_neon_entry(token, ctx0, res, costs, probs)
}

/// Calculate the cost of encoding a residual block using probability-based costs.
/// Scalar fallback for non-SIMD platforms.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
)))]
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
pub(crate) fn get_residual_cost_scalar(
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

    // Pre-index by ctype once for the inner loop
    let costs_for_type = &costs.level_cost[ctype];
    let band = VP8_ENC_BANDS[n] as usize;
    let mut t = &costs_for_type[band][ctx];

    // Process coefficients from first to last-1
    let last = res.last as usize;
    while n < last {
        let v = res.coeffs[n].unsigned_abs() as usize;

        // VP8LevelCost: fixed cost + variable cost from table
        cost +=
            VP8_LEVEL_FIXED_COSTS[v.min(MAX_LEVEL)] as u32 + t[v.min(MAX_VARIABLE_LEVEL)] as u32;

        // Update context for next position based on current value
        ctx = if v >= 2 { 2 } else { v };

        n += 1;
        let next_band = (VP8_ENC_BANDS[n] as usize) & 7;
        t = &costs_for_type[next_band][ctx.min(2)];
    }

    // Last coefficient is always non-zero
    {
        let v = res.coeffs[n].unsigned_abs() as usize;
        debug_assert!(v != 0, "Last coefficient should be non-zero");

        cost +=
            VP8_LEVEL_FIXED_COSTS[v.min(MAX_LEVEL)] as u32 + t[v.min(MAX_VARIABLE_LEVEL)] as u32;

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

use super::cost::level_costs::LevelCostArray;

/// Shared inner loop for residual cost calculation.
///
/// Processes coefficients from position `first` through `last` (inclusive),
/// accumulating level costs. Uses the padded LevelCostArray (128 entries)
/// with `& 0x7F` to eliminate bounds checks on level indexing.
///
/// `initial_ctx` is the context (0/1/2) for the first coefficient position.
/// `costs_for_type` is `&level_cost[ctype]` -- the [8][3] cost table for one type.
///
/// Returns the accumulated cost for all coefficients first..=last.
#[inline(always)]
fn residual_cost_loop(
    first: usize,
    last: usize,
    initial_ctx: usize,
    levels: &[u8; 16],
    abs_levels: &[u16; 16],
    ctxs: &[u8; 16],
    costs_for_type: &[[LevelCostArray; 3]; 8],
) -> u32 {
    let mut cost = 0u32;
    let mut n = first;

    // Get cost table pointer for position n, context from previous coefficient.
    // VP8_ENC_BANDS is a const [u8; 17] with values in 0..8.
    let mut band = VP8_ENC_BANDS[n] as usize;
    let mut t: &LevelCostArray = &costs_for_type[band][initial_ctx.min(2)];

    // Process first..last-1: each iteration reads coeff at n, updates context.
    // `level & 0x7F` eliminates the bounds check on t[] (padded to 128 entries).
    // `ctx.min(2)` clamps to [0,2] for the 3-element cost arrays.
    // `band & 7` proves band < 8 for the 8-element costs_for_type.
    while n < last {
        let level = (levels[n] as usize) & 0x7F;
        let flevel = abs_levels[n] as usize;

        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;

        let ctx = (ctxs[n] as usize).min(2);
        n += 1;
        band = (VP8_ENC_BANDS[n] as usize) & 7;
        t = &costs_for_type[band][ctx];
    }

    // Last coefficient (always non-zero)
    {
        let level = (levels[n] as usize) & 0x7F;
        let flevel = abs_levels[n] as usize;
        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;
    }

    cost
}

/// Entry shim for get_residual_cost_sse2
#[cfg(target_arch = "x86_64")]
#[arcane]
fn get_residual_cost_entry(
    _token: X64V3Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_sse2(_token, ctx0, res, costs, probs)
}

/// SSE2 implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn get_residual_cost_sse2(
    _token: X64V3Token,
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
    let n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let ctx = ctx0;

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
        let (c0_arr, c1_arr, c2_arr, c3_arr) = split4_ref(res.coeffs);
        let c0_32 = simd_mem::_mm_loadu_si128(c0_arr);
        let c1_32 = simd_mem::_mm_loadu_si128(c1_arr);
        let c2_32 = simd_mem::_mm_loadu_si128(c2_arr);
        let c3_32 = simd_mem::_mm_loadu_si128(c3_arr);

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
        let (al0, al1) = split2_mut(&mut abs_levels);
        simd_mem::_mm_storeu_si128(al0, e0);
        simd_mem::_mm_storeu_si128(al1, e1);
    }

    // Pre-index by ctype once, then use VP8_ENC_BANDS for band lookup in the loop.
    let costs_for_type = &costs.level_cost[ctype];
    let last = res.last as usize;

    // Delegate to the shared inner loop (main loop + last coefficient).
    cost += residual_cost_loop(n, last, ctx, &levels, &abs_levels, &ctxs, costs_for_type);

    // Add EOB cost for the position after the last coefficient
    if last < 15 {
        let next_band = VP8_ENC_BANDS[last + 1] as usize;
        let next_ctx = ctxs[last] as usize;
        let last_p0 = probs[ctype][next_band][next_ctx][0];
        cost += vp8_bit_cost(false, last_p0) as u32;
    }

    cost
}

/// Entry shim for find_last_nonzero_simd
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn find_last_nonzero_simd_entry(_token: X64V3Token, coeffs: &[i32; 16]) -> i32 {
    find_last_nonzero_simd(_token, coeffs)
}

/// Find last non-zero coefficient using SIMD.
/// Ported from libwebp's SetResidualCoeffs_SSE2.
///
/// Returns -1 if all coefficients are zero.
#[cfg(target_arch = "x86_64")]
#[rite]
#[allow(dead_code)]
fn find_last_nonzero_simd(_token: X64V3Token, coeffs: &[i32; 16]) -> i32 {
    let zero = _mm_setzero_si128();

    // Load coefficients as i32 and pack to i16
    let (c0_arr, c1_arr, c2_arr, c3_arr) = split4_ref(coeffs);
    let c0_32 = simd_mem::_mm_loadu_si128(c0_arr);
    let c1_32 = simd_mem::_mm_loadu_si128(c1_arr);
    let c2_32 = simd_mem::_mm_loadu_si128(c2_arr);
    let c3_32 = simd_mem::_mm_loadu_si128(c3_arr);

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

// =============================================================================
// NEON (aarch64) residual cost implementation
// =============================================================================

/// Entry shim for get_residual_cost_neon
#[cfg(target_arch = "aarch64")]
#[arcane]
fn get_residual_cost_neon_entry(
    _token: NeonToken,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_neon(_token, ctx0, res, costs, probs)
}

/// NEON implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(target_arch = "aarch64")]
#[rite]
pub(crate) fn get_residual_cost_neon(
    _token: NeonToken,
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
    let n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let ctx = ctx0;

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

    // Precompute clamped levels and contexts using NEON
    {
        let k_cst2 = vdupq_n_u8(2);
        let k_cst67 = vdupq_n_u8(MAX_VARIABLE_LEVEL as u8);

        // Load coefficients as i32 and pack to i16 (signed saturation via vqmovn)
        let (c0_arr, c1_arr, c2_arr, c3_arr) = split4_ref(res.coeffs);
        let c0 = simd_mem::vld1q_s32(c0_arr);
        let c1 = simd_mem::vld1q_s32(c1_arr);
        let c2 = simd_mem::vld1q_s32(c2_arr);
        let c3 = simd_mem::vld1q_s32(c3_arr);

        // Pack i32 to i16 (signed saturation)
        let s0 = vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1)); // 8 x i16
        let s1 = vcombine_s16(vqmovn_s32(c2), vqmovn_s32(c3)); // 8 x i16

        // Absolute value (i16)
        let e0 = vabsq_s16(s0);
        let e1 = vabsq_s16(s1);

        // Pack abs i16 to u8 (unsigned saturation via vqmovun — values are positive)
        let f = vcombine_u8(vqmovun_s16(e0), vqmovun_s16(e1)); // 16 x u8

        // Context: min(abs, 2)
        let g = vminq_u8(f, k_cst2);

        // Clamped level: min(abs, 67) for cost table lookup
        let h = vminq_u8(f, k_cst67);

        // Store results
        simd_mem::vst1q_u8(&mut ctxs, g);
        simd_mem::vst1q_u8(&mut levels, h);

        // Store 16-bit absolute values (reinterpret signed abs as unsigned)
        let (al0, al1) = split2_mut(&mut abs_levels);
        simd_mem::vst1q_u16(al0, vreinterpretq_u16_s16(e0));
        simd_mem::vst1q_u16(al1, vreinterpretq_u16_s16(e1));
    }

    // Pre-index by ctype once, delegate to shared inner loop.
    let costs_for_type = &costs.level_cost[ctype];
    let last = res.last as usize;

    cost += residual_cost_loop(n, last, ctx, &levels, &abs_levels, &ctxs, costs_for_type);

    // Add EOB cost for the position after the last coefficient
    if last < 15 {
        let next_band = VP8_ENC_BANDS[last + 1] as usize;
        let next_ctx = ctxs[last] as usize;
        let last_p0 = probs[ctype][next_band][next_ctx][0];
        cost += vp8_bit_cost(false, last_p0) as u32;
    }

    cost
}

/// Entry shim for find_last_nonzero_neon
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(dead_code)]
fn find_last_nonzero_neon_entry(_token: NeonToken, coeffs: &[i32; 16]) -> i32 {
    find_last_nonzero_neon(_token, coeffs)
}

/// Find last non-zero coefficient using NEON.
/// Returns -1 if all coefficients are zero.
#[cfg(target_arch = "aarch64")]
#[rite]
#[allow(dead_code)]
fn find_last_nonzero_neon(_token: NeonToken, coeffs: &[i32; 16]) -> i32 {
    let zero = vdupq_n_s32(0);

    // Load coefficients as i32
    let (c0_arr, c1_arr, c2_arr, c3_arr) = split4_ref(coeffs);
    let c0 = simd_mem::vld1q_s32(c0_arr);
    let c1 = simd_mem::vld1q_s32(c1_arr);
    let c2 = simd_mem::vld1q_s32(c2_arr);
    let c3 = simd_mem::vld1q_s32(c3_arr);

    // Pack i32 to i16 (signed saturation)
    let s0 = vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1)); // 8 x i16
    let s1 = vcombine_s16(vqmovn_s32(c2), vqmovn_s32(c3)); // 8 x i16

    // Pack i16 to i8 (signed saturation)
    let m0 = vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)); // 16 x i8

    // Compare with zero: result is 0xFF where equal, 0x00 where not
    let eq_zero = vceqq_s8(m0, vdupq_n_s8(0));

    // Invert: 0xFF where non-zero
    let ne_zero = vmvnq_s8(vreinterpretq_s8_u8(eq_zero));

    // Create index vector [0, 1, 2, ..., 15]
    // For each non-zero position, keep its index; for zero positions, keep 0
    let indices: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let idx_vec = simd_mem::vld1q_u8(&indices);

    // Mask: keep index where coefficient is non-zero, 0 otherwise
    let masked = vandq_u8(idx_vec, vreinterpretq_u8_s8(ne_zero));

    // Find max index among non-zero positions
    let max_idx = vmaxvq_u8(masked);

    // Check if any coefficient is non-zero
    // If all are zero, masked is all zeros and max_idx is 0
    // But index 0 could also be a valid non-zero position, so check separately
    if max_idx > 0 {
        max_idx as i32
    } else {
        // max_idx == 0: either coeff[0] is the only non-zero, or all are zero
        let _ = zero; // suppress unused warning
        if coeffs[0] != 0 { 0 } else { -1 }
    }
}

// =============================================================================
// WASM SIMD128 residual cost implementation
// =============================================================================

/// Entry shim for get_residual_cost_wasm
#[cfg(target_arch = "wasm32")]
#[arcane]
fn get_residual_cost_wasm_entry(
    _token: Wasm128Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_wasm(_token, ctx0, res, costs, probs)
}

/// WASM SIMD128 implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(target_arch = "wasm32")]
#[rite]
pub(crate) fn get_residual_cost_wasm(
    _token: Wasm128Token,
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
    let n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let ctx = ctx0;

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
    {
        let k_cst2 = u8x16_splat(2);
        let k_cst67 = u8x16_splat(MAX_VARIABLE_LEVEL as u8);

        // Load coefficients as i32
        let c0 = i32x4(res.coeffs[0], res.coeffs[1], res.coeffs[2], res.coeffs[3]);
        let c1 = i32x4(res.coeffs[4], res.coeffs[5], res.coeffs[6], res.coeffs[7]);
        let c2 = i32x4(res.coeffs[8], res.coeffs[9], res.coeffs[10], res.coeffs[11]);
        let c3 = i32x4(
            res.coeffs[12],
            res.coeffs[13],
            res.coeffs[14],
            res.coeffs[15],
        );

        // Pack i32 → i16 (signed saturation)
        let s0 = i16x8_narrow_i32x4(c0, c1);
        let s1 = i16x8_narrow_i32x4(c2, c3);

        // Absolute value
        let e0 = i16x8_abs(s0);
        let e1 = i16x8_abs(s1);

        // Pack abs i16 → u8 (unsigned saturation, values are positive)
        let f = u8x16_narrow_i16x8(e0, e1);

        // Context: min(abs, 2)
        let g = u8x16_min(f, k_cst2);

        // Clamped level: min(abs, 67) for cost table lookup
        let h = u8x16_min(f, k_cst67);

        // Store ctxs (extract all 16 lanes)
        ctxs[0] = u8x16_extract_lane::<0>(g);
        ctxs[1] = u8x16_extract_lane::<1>(g);
        ctxs[2] = u8x16_extract_lane::<2>(g);
        ctxs[3] = u8x16_extract_lane::<3>(g);
        ctxs[4] = u8x16_extract_lane::<4>(g);
        ctxs[5] = u8x16_extract_lane::<5>(g);
        ctxs[6] = u8x16_extract_lane::<6>(g);
        ctxs[7] = u8x16_extract_lane::<7>(g);
        ctxs[8] = u8x16_extract_lane::<8>(g);
        ctxs[9] = u8x16_extract_lane::<9>(g);
        ctxs[10] = u8x16_extract_lane::<10>(g);
        ctxs[11] = u8x16_extract_lane::<11>(g);
        ctxs[12] = u8x16_extract_lane::<12>(g);
        ctxs[13] = u8x16_extract_lane::<13>(g);
        ctxs[14] = u8x16_extract_lane::<14>(g);
        ctxs[15] = u8x16_extract_lane::<15>(g);

        // Store levels
        levels[0] = u8x16_extract_lane::<0>(h);
        levels[1] = u8x16_extract_lane::<1>(h);
        levels[2] = u8x16_extract_lane::<2>(h);
        levels[3] = u8x16_extract_lane::<3>(h);
        levels[4] = u8x16_extract_lane::<4>(h);
        levels[5] = u8x16_extract_lane::<5>(h);
        levels[6] = u8x16_extract_lane::<6>(h);
        levels[7] = u8x16_extract_lane::<7>(h);
        levels[8] = u8x16_extract_lane::<8>(h);
        levels[9] = u8x16_extract_lane::<9>(h);
        levels[10] = u8x16_extract_lane::<10>(h);
        levels[11] = u8x16_extract_lane::<11>(h);
        levels[12] = u8x16_extract_lane::<12>(h);
        levels[13] = u8x16_extract_lane::<13>(h);
        levels[14] = u8x16_extract_lane::<14>(h);
        levels[15] = u8x16_extract_lane::<15>(h);

        // Store 16-bit absolute values
        abs_levels[0] = u16x8_extract_lane::<0>(e0);
        abs_levels[1] = u16x8_extract_lane::<1>(e0);
        abs_levels[2] = u16x8_extract_lane::<2>(e0);
        abs_levels[3] = u16x8_extract_lane::<3>(e0);
        abs_levels[4] = u16x8_extract_lane::<4>(e0);
        abs_levels[5] = u16x8_extract_lane::<5>(e0);
        abs_levels[6] = u16x8_extract_lane::<6>(e0);
        abs_levels[7] = u16x8_extract_lane::<7>(e0);
        abs_levels[8] = u16x8_extract_lane::<0>(e1);
        abs_levels[9] = u16x8_extract_lane::<1>(e1);
        abs_levels[10] = u16x8_extract_lane::<2>(e1);
        abs_levels[11] = u16x8_extract_lane::<3>(e1);
        abs_levels[12] = u16x8_extract_lane::<4>(e1);
        abs_levels[13] = u16x8_extract_lane::<5>(e1);
        abs_levels[14] = u16x8_extract_lane::<6>(e1);
        abs_levels[15] = u16x8_extract_lane::<7>(e1);
    }

    // Pre-index by ctype once, delegate to shared inner loop.
    let costs_for_type = &costs.level_cost[ctype];
    let last = res.last as usize;

    cost += residual_cost_loop(n, last, ctx, &levels, &abs_levels, &ctxs, costs_for_type);

    // Add EOB cost for the position after the last coefficient
    if last < 15 {
        let next_band = VP8_ENC_BANDS[last + 1] as usize;
        let next_ctx = ctxs[last] as usize;
        let last_p0 = probs[ctype][next_band][next_ctx][0];
        cost += vp8_bit_cost(false, last_p0) as u32;
    }

    cost
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
    // Track non-zero context like libwebp's VP8GetCostLuma16
    let mut top_nz = [false; 4];
    let mut left_nz = [false; 4];

    for y in 0..4 {
        for x in 0..4 {
            let block_idx = y * 4 + x;
            let ctx = (top_nz[x] as usize) + (left_nz[y] as usize);
            let ac_res = Residual::new(&ac_levels[block_idx], 0, 1); // Start at position 1 (skip DC)
            total_cost += get_residual_cost(ctx, &ac_res, costs, probs);
            let has_nz = ac_res.last >= 0;
            top_nz[x] = has_nz;
            left_nz[y] = has_nz;
        }
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
