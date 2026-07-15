//! Intra prediction mode selection for macroblocks.
//!
//! Contains methods for selecting optimal I16, I4, and UV prediction modes
//! using rate-distortion cost evaluation.
#![allow(clippy::too_many_arguments)]

use crate::common::prediction::*;
use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::{
    FIXED_COSTS_I16, FIXED_COSTS_UV, FLATNESS_LIMIT_I4, FLATNESS_LIMIT_I16, FLATNESS_LIMIT_UV,
    FLATNESS_PENALTY, RD_DISTO_MULT, estimate_dc16_cost, estimate_residual_cost, get_cost_luma4,
    get_cost_luma16, get_cost_uv, is_flat_coeffs, is_flat_source_16, tdisto_4x4, tdisto_8x8,
    tdisto_16x16, trellis_quantize_block,
};

use crate::encoder::psy;

use archmage::prelude::*;

use super::{MacroblockInfo, sse_8x8_chroma, sse_16x16_luma};

// =============================================================================
// Helper dispatch functions for inline SSE computation
// =============================================================================

/// Dispatch SSE4x4 computation to best available SIMD path.
#[inline(always)]
fn sse4x4_dispatch(src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    incant!(sse4x4_impl(src, pred), [v3, neon, wasm128, scalar])
}

/// Test-only re-export of `sse4x4_dispatch` for cross-arch SIMD parity
/// tests in `tests/simd_dispatch_arch_parity.rs`.
#[doc(hidden)]
pub fn __test_only_sse4x4_dispatch(src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    sse4x4_dispatch(src, pred)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse4x4_impl_v3(_token: X64V3Token, src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    crate::common::simd_sse::sse4x4_sse2(_token, src, pred)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse4x4_impl_neon(token: NeonToken, src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    crate::common::simd_neon::sse4x4_neon(token, src, pred)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse4x4_impl_wasm128(token: Wasm128Token, src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    crate::common::simd_wasm::sse4x4_wasm(token, src, pred)
}

#[inline(always)]
fn sse4x4_impl_scalar(_token: ScalarToken, src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for k in 0..16 {
        let diff = i32::from(src[k]) - i32::from(pred[k]);
        sum += (diff * diff) as u32;
    }
    sum
}

/// Dispatch SSE4x4 with residual computation to best available SIMD path.
#[inline(always)]
fn sse4x4_with_residual_dispatch(src: &[u8; 16], pred: &[u8; 16], dequantized: &[i32; 16]) -> u32 {
    incant!(
        sse4x4_with_residual_impl(src, pred, dequantized),
        [v3, neon, wasm128, scalar]
    )
}

/// Test-only re-export for cross-arch SIMD parity tests.
#[doc(hidden)]
pub fn __test_only_sse4x4_with_residual_dispatch(
    src: &[u8; 16],
    pred: &[u8; 16],
    dequantized: &[i32; 16],
) -> u32 {
    sse4x4_with_residual_dispatch(src, pred, dequantized)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse4x4_with_residual_impl_v3(
    _token: X64V3Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    dequantized: &[i32; 16],
) -> u32 {
    crate::common::simd_sse::sse4x4_with_residual_sse2(_token, src, pred, dequantized)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse4x4_with_residual_impl_neon(
    token: NeonToken,
    src: &[u8; 16],
    pred: &[u8; 16],
    dequantized: &[i32; 16],
) -> u32 {
    crate::common::simd_neon::sse4x4_with_residual_neon(token, src, pred, dequantized)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse4x4_with_residual_impl_wasm128(
    token: Wasm128Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    dequantized: &[i32; 16],
) -> u32 {
    crate::common::simd_wasm::sse4x4_with_residual_wasm(token, src, pred, dequantized)
}

#[inline(always)]
fn sse4x4_with_residual_impl_scalar(
    _token: ScalarToken,
    src: &[u8; 16],
    pred: &[u8; 16],
    dequantized: &[i32; 16],
) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let reconstructed = (i32::from(pred[i]) + dequantized[i]).clamp(0, 255) as u8;
        let diff = i32::from(src[i]) - i32::from(reconstructed);
        sum += (diff * diff) as u32;
    }
    sum
}

// =============================================================================
// Arcane (SIMD-hoisted) inner functions for I4 mode selection
//
// These free functions run entirely within an #[arcane] context, allowing the
// compiler to inline all SIMD leaf functions (ftransform, quantize, sse4x4,
// get_residual_cost, etc.) and keep values in SIMD registers across calls.
// This eliminates the per-call dispatch overhead (~9.7M instructions) and
// enables cross-function optimizations.
// =============================================================================

/// Result of evaluating a single I4 block mode
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
struct I4BlockResult {
    mode_idx: usize,
    rd_score: u64,
    has_nz: bool,
    dequantized: [i32; 16],
    sse: u32,
    spectral_disto: i32,
    psy_cost: i32,
    coeff_cost: u32,
}

/// Pre-sort I4 prediction modes by prediction SSE (ascending).
/// Runs sse4x4 for all 10 modes using direct SIMD calls (no dispatch overhead).
#[archmage::arcane]
fn presort_i4_modes_sse2(
    _token: archmage::X64V3Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
) -> [(u32, usize); 10] {
    let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
    for (mode_idx, entry) in mode_sse.iter_mut().enumerate() {
        let pred = preds.get(mode_idx);
        let sse = crate::common::simd_sse::sse4x4_sse2(_token, src_block, pred);
        *entry = (sse, mode_idx);
    }
    mode_sse.sort_unstable_by_key(|&(sse, _)| sse);
    mode_sse
}

/// Evaluate I4 block modes within a single arcane context.
///
/// This is the hot inner loop of pick_best_intra4, with all SIMD calls
/// going directly to `_sse2` variants (inlinable within arcane context).
///
/// Returns the best mode result, or None if no mode beats `best_block_score_limit`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
fn evaluate_i4_modes_sse2(
    _token: archmage::X64V3Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
    mode_sse_order: &[(u32, usize)],
    max_modes: usize,
    mode_costs: &[u16; 10],
    nz_top: bool,
    nz_left: bool,
    // RD parameters
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    lambda_i4: u32,
    tlambda: u32,
    trellis_lambda_i4: Option<u32>,
    level_costs: &crate::encoder::cost::LevelCosts,
    probs: &TokenProbTables,
    psy_config: &crate::encoder::psy::PsyConfig,
    luma_csf: &[u16; 16],
) -> Option<I4BlockResult> {
    use crate::encoder::residual_cost::Residual;

    let mut best: Option<I4BlockResult> = None;
    let mut best_block_score = u64::MAX;

    for &(_, mode_idx) in mode_sse_order[..max_modes].iter() {
        let pred = preds.get(mode_idx);

        // Fused residual + DCT using direct SIMD call
        let mut residual =
            crate::common::transform::ftransform_from_u8_4x4_sse2(_token, src_block, pred);

        // Quantize - use trellis if enabled, otherwise fused quantize+dequantize
        let mut quantized_zigzag = [0i32; 16];
        let mut quantized_natural = [0i32; 16];
        let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
            // Trellis quantization (scalar, called from arcane context is fine)
            let ctx0 = usize::from(nz_top) + usize::from(nz_left);
            const CTYPE_I4_AC: usize = 3;
            let nz = trellis_quantize_block(
                &mut residual,
                &mut quantized_zigzag,
                y1_matrix,
                lambda,
                0,
                level_costs,
                CTYPE_I4_AC,
                ctx0,
                psy_config,
            );
            // Convert zigzag to natural order
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_natural[j] = quantized_zigzag[n];
            }
            // Dequantize + IDCT for trellis path
            let mut dq = quantized_natural;
            for (idx, val) in dq.iter_mut().enumerate() {
                *val = y1_matrix.dequantize(*val, idx);
            }
            crate::common::transform::idct4x4_sse2(_token, &mut dq);
            (nz, dq)
        } else {
            // Fused quantize+dequantize using direct SIMD call
            let mut dequant_natural = [0i32; 16];
            let nz = crate::encoder::quantize::quantize_dequantize_block_sse2(
                _token,
                &residual,
                y1_matrix,
                true,
                &mut quantized_natural,
                &mut dequant_natural,
            );
            // Convert natural to zigzag for cost estimation
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_zigzag[n] = quantized_natural[j];
            }
            // IDCT the dequantized values (direct sse2, skip token re-summon)
            crate::common::transform::idct4x4_sse2(_token, &mut dequant_natural);
            (nz, dequant_natural)
        };

        // Compute SSE using direct SIMD call (cheap, needed for early exit)
        let sse = crate::common::simd_sse::sse4x4_with_residual_sse2(
            _token,
            src_block,
            pred,
            &dequantized,
        );

        // Flatness penalty for non-DC modes (cheap SIMD check)
        let flatness_penalty: u32 = if mode_idx > 0 {
            let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
            if crate::encoder::cost::distortion::is_flat_coeffs_sse2(
                _token,
                &levels_i16,
                1,
                FLATNESS_LIMIT_I4,
            ) {
                FLATNESS_PENALTY
            } else {
                0
            }
        } else {
            0
        };

        // Early exit #1 (libwebp-style): check score WITHOUT coefficient cost.
        // Since coeff_cost >= 0 and spectral_disto >= 0, the actual score can
        // only be >= this lower bound. Skip expensive residual_cost if hopeless.
        let mode_cost = mode_costs[mode_idx];
        let lower_bound =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, flatness_penalty, lambda_i4)
                as u64;
        if lower_bound >= best_block_score {
            continue;
        }

        // Get coefficient cost using direct SIMD call (expensive)
        let ctx0 = (nz_top as usize) + (nz_left as usize);
        let res = Residual::new(&quantized_zigzag, 3, 0); // CTYPE_I4_AC=3, first=0
        let coeff_cost = crate::encoder::residual_cost::get_residual_cost_sse2(
            _token,
            ctx0,
            &res,
            level_costs,
            probs,
        );

        // Early exit #2: check with coefficient cost but without spectral/psy.
        // Skip spectral distortion and psy-rd computation if already losing.
        let total_rate_cost = coeff_cost + flatness_penalty;
        let base_rd_score =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, total_rate_cost, lambda_i4)
                as u64;
        if base_rd_score >= best_block_score {
            continue;
        }

        // Spectral distortion + psy-rd
        let (spectral_disto, psy_cost) = if tlambda > 0 || psy_config.psy_rd_strength > 0 {
            // Build reconstructed 4x4 block
            let mut rec_block = [0u8; 16];
            for k in 0..16 {
                rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
            }

            let td = if tlambda > 0 {
                let td_raw = crate::common::simd_sse::tdisto_4x4_fused_sse2(
                    _token, src_block, &rec_block, 4, luma_csf,
                );
                (tlambda as i32 * td_raw + 128) >> 8
            } else {
                0
            };

            let psy = if psy_config.psy_rd_strength > 0 {
                let src_satd = psy::satd_4x4(src_block, 4);
                let rec_satd = psy::satd_4x4(&rec_block, 4);
                psy::psy_rd_cost(src_satd, rec_satd, psy_config.psy_rd_strength)
            } else {
                0
            };

            (td, psy)
        } else {
            (0, 0)
        };

        // Final RD score
        let rd_score = crate::encoder::cost::rd_score_full(
            sse,
            spectral_disto + psy_cost,
            mode_cost,
            total_rate_cost,
            lambda_i4,
        ) as u64;

        if rd_score < best_block_score {
            best_block_score = rd_score;
            best = Some(I4BlockResult {
                mode_idx,
                rd_score,
                has_nz,
                dequantized,
                sse,
                spectral_disto,
                psy_cost,
                coeff_cost,
            });
        }
    }

    best
}

/// Pre-sort I4 prediction modes by prediction SSE (ascending) for wasm SIMD128.
/// Runs sse4x4 for all 10 modes using direct SIMD calls (no dispatch overhead).
#[archmage::arcane]
fn presort_i4_modes_wasm(
    _token: archmage::Wasm128Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
) -> [(u32, usize); 10] {
    let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
    for (mode_idx, entry) in mode_sse.iter_mut().enumerate() {
        let pred = preds.get(mode_idx);
        let sse = crate::common::simd_wasm::sse4x4_wasm(_token, src_block, pred);
        *entry = (sse, mode_idx);
    }
    mode_sse.sort_unstable_by_key(|&(sse, _)| sse);
    mode_sse
}

/// Evaluate I4 block modes within a single arcane context for wasm SIMD128.
///
/// This is the hot inner loop of pick_best_intra4, with all SIMD calls
/// going directly to `_wasm` #[rite] variants (inlinable within arcane context).
///
/// Returns the best mode result, or None if no mode beats `best_block_score_limit`.
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn evaluate_i4_modes_wasm(
    _token: archmage::Wasm128Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
    mode_sse_order: &[(u32, usize)],
    max_modes: usize,
    mode_costs: &[u16; 10],
    nz_top: bool,
    nz_left: bool,
    // RD parameters
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    lambda_i4: u32,
    tlambda: u32,
    trellis_lambda_i4: Option<u32>,
    level_costs: &crate::encoder::cost::LevelCosts,
    probs: &TokenProbTables,
    psy_config: &crate::encoder::psy::PsyConfig,
    luma_csf: &[u16; 16],
) -> Option<I4BlockResult> {
    use crate::encoder::residual_cost::Residual;

    let mut best: Option<I4BlockResult> = None;
    let mut best_block_score = u64::MAX;

    for &(_, mode_idx) in mode_sse_order[..max_modes].iter() {
        let pred = preds.get(mode_idx);

        // Fused residual + DCT using direct SIMD call
        let mut residual =
            crate::common::transform::ftransform_from_u8_4x4_wasm_impl(_token, src_block, pred);

        // Quantize - use trellis if enabled, otherwise fused quantize+dequantize
        let mut quantized_zigzag = [0i32; 16];
        let mut quantized_natural = [0i32; 16];
        let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
            // Trellis quantization (scalar, called from arcane context is fine)
            let ctx0 = usize::from(nz_top) + usize::from(nz_left);
            const CTYPE_I4_AC: usize = 3;
            let nz = trellis_quantize_block(
                &mut residual,
                &mut quantized_zigzag,
                y1_matrix,
                lambda,
                0,
                level_costs,
                CTYPE_I4_AC,
                ctx0,
                psy_config,
            );
            // Convert zigzag to natural order
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_natural[j] = quantized_zigzag[n];
            }
            // Dequantize + IDCT for trellis path
            let mut dq = quantized_natural;
            for (idx, val) in dq.iter_mut().enumerate() {
                *val = y1_matrix.dequantize(*val, idx);
            }
            transform::idct4x4(&mut dq);
            (nz, dq)
        } else {
            // Fused quantize+dequantize using direct SIMD call
            let mut dequant_natural = [0i32; 16];
            let nz = crate::common::simd_wasm::quantize_dequantize_block_wasm(
                _token,
                &residual,
                y1_matrix,
                true,
                &mut quantized_natural,
                &mut dequant_natural,
            );
            // Convert natural to zigzag for cost estimation
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_zigzag[n] = quantized_natural[j];
            }
            // IDCT the dequantized values
            transform::idct4x4(&mut dequant_natural);
            (nz, dequant_natural)
        };

        // Compute SSE using direct SIMD call (cheap, needed for early exit)
        let sse = crate::common::simd_wasm::sse4x4_with_residual_wasm(
            _token,
            src_block,
            pred,
            &dequantized,
        );

        // Flatness penalty for non-DC modes (cheap SIMD check)
        let flatness_penalty: u32 = if mode_idx > 0 {
            let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
            if crate::common::simd_wasm::is_flat_coeffs_wasm(
                _token,
                &levels_i16,
                1,
                FLATNESS_LIMIT_I4,
            ) {
                FLATNESS_PENALTY
            } else {
                0
            }
        } else {
            0
        };

        // Early exit #1 (libwebp-style): check score WITHOUT coefficient cost.
        // Since coeff_cost >= 0 and spectral_disto >= 0, the actual score can
        // only be >= this lower bound. Skip expensive residual_cost if hopeless.
        let mode_cost = mode_costs[mode_idx];
        let lower_bound =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, flatness_penalty, lambda_i4)
                as u64;
        if lower_bound >= best_block_score {
            continue;
        }

        // Get coefficient cost using direct SIMD call (expensive)
        let ctx0 = (nz_top as usize) + (nz_left as usize);
        let res = Residual::new(&quantized_zigzag, 3, 0); // CTYPE_I4_AC=3, first=0
        let coeff_cost = crate::encoder::residual_cost::get_residual_cost_wasm(
            _token,
            ctx0,
            &res,
            level_costs,
            probs,
        );

        // Early exit #2: check with coefficient cost but without spectral/psy.
        // Skip spectral distortion and psy-rd computation if already losing.
        let total_rate_cost = coeff_cost + flatness_penalty;
        let base_rd_score =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, total_rate_cost, lambda_i4)
                as u64;
        if base_rd_score >= best_block_score {
            continue;
        }

        // Spectral distortion + psy-rd
        let (spectral_disto, psy_cost) = if tlambda > 0 || psy_config.psy_rd_strength > 0 {
            // Build reconstructed 4x4 block
            let mut rec_block = [0u8; 16];
            for k in 0..16 {
                rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
            }

            let td = if tlambda > 0 {
                let td_raw = crate::common::simd_wasm::tdisto_4x4_fused_wasm(
                    _token, src_block, &rec_block, 4, luma_csf,
                );
                (tlambda as i32 * td_raw + 128) >> 8
            } else {
                0
            };

            let psy = if psy_config.psy_rd_strength > 0 {
                let src_satd = psy::satd_4x4(src_block, 4);
                let rec_satd = psy::satd_4x4(&rec_block, 4);
                psy::psy_rd_cost(src_satd, rec_satd, psy_config.psy_rd_strength)
            } else {
                0
            };

            (td, psy)
        } else {
            (0, 0)
        };

        // Final RD score
        let rd_score = crate::encoder::cost::rd_score_full(
            sse,
            spectral_disto + psy_cost,
            mode_cost,
            total_rate_cost,
            lambda_i4,
        ) as u64;

        if rd_score < best_block_score {
            best_block_score = rd_score;
            best = Some(I4BlockResult {
                mode_idx,
                rd_score,
                has_nz,
                dequantized,
                sse,
                spectral_disto,
                psy_cost,
                coeff_cost,
            });
        }
    }

    best
}

impl<'a> super::Vp8Encoder<'a> {
    /// Select the best 16x16 luma prediction mode using full RD (rate-distortion) cost.
    ///
    /// This implements libwebp's full RD path for PickBestIntra16:
    /// 1. For each mode: generate prediction, forward transform, quantize
    /// 2. Dequantize and inverse transform to get reconstructed block
    /// 3. Compute SSE between reconstructed and source (NOT prediction vs source!)
    /// 4. Compute spectral distortion (TDisto) if tlambda > 0
    /// 5. Include coefficient cost in rate term
    /// 6. Apply flat source penalty if applicable
    ///
    /// RD formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
    /// Where: R = coeff cost, H = mode cost, D = SSE, SD = spectral distortion
    ///
    /// Returns `(best_mode, rd_score, d_raw)`:
    ///   - `rd_score` — combined RD score for comparison against Intra4x4.
    ///   - `d_raw` — winning-mode raw source-vs-reconstruction SSE before
    ///     the flat-source doubling. Matches libwebp's `rd->D` (the value
    ///     fed into `dqm->min_disto` gating of `StoreMaxDelta`,
    ///     `quant_enc.c:1111`). Threaded into `MacroblockInfo` so the
    ///     `store_max_delta` call site can apply the `D > min_disto` gate
    ///     (issue #44).
    fn pick_best_intra16(&self, mbx: usize, mby: usize) -> (LumaMode, u64, u32) {
        // Check for debug mode
        #[cfg(feature = "mode_debug")]
        let debug_i16 = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let _debug_i16 = false;
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // Fast path for method 0-1: DC mode only with SSE-based scoring
        // This avoids the full RD evaluation loop for maximum speed
        if self.method <= 1 {
            return self.pick_intra16_fast_dc(mbx, mby);
        }

        // The 4 modes to try for 16x16 luma prediction (order matches FIXED_COSTS_I16)
        const MODES: [LumaMode; 4] = [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM];

        let segment = self.get_segment_for_mb(mbx, mby);
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        let lambda = segment.lambda_i16;
        let tlambda = segment.tlambda;

        // Use updated probabilities if available (for consistent mode selection)
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Check if source block is flat (for flat source penalty)
        let src_base = mby * 16 * src_width + mbx * 16;
        let is_flat = is_flat_source_16(&self.frame.ybuf[src_base..], src_width);

        // Pre-extract source block for TDisto/psy-rd (avoid repeated extraction per mode)
        let need_spectral = tlambda > 0 || segment.psy_config.psy_rd_strength > 0;
        let src_block = if need_spectral {
            let mut block = [0u8; 256];
            for y in 0..16 {
                let src_row = (mby * 16 + y) * src_width + mbx * 16;
                block[y * 16..(y + 1) * 16]
                    .copy_from_slice(&self.frame.ybuf[src_row..src_row + 16]);
            }
            Some(block)
        } else {
            None
        };

        let mut best_mode = LumaMode::DC;
        let mut best_rd_score = i64::MAX;
        // libwebp evaluates I16 modes in the order DC, TM, V, H (its internal mode
        // numbers 0,1,2,3) and keeps the first with a strictly-smaller score, so an
        // exact tie resolves to the lowest libwebp mode number. Our `MODES` array is
        // ordered [DC, V, H, TM]; the corresponding libwebp mode numbers are
        // [0, 2, 3, 1]. Under StrictLibwebpParity we tie-break by this rank so an
        // H/TM (or any) tie picks the same mode as libwebp (#38). At frame edges TM
        // and H predict identically and tie constantly, so this drives a large share
        // of the top-row/left-column I16 mode agreement.
        const I16_LIB_RANK: [u8; 4] = [0, 2, 3, 1];
        let parity_tiebreak =
            self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity;
        let mut best_lib_rank = u8::MAX;
        // Store best mode's cost components for final score recalculation with lambda_mode
        let mut best_coeff_cost = 0u32;
        let mut best_mode_cost = 0u16;
        let mut best_sse = 0u32;
        let mut best_spectral_disto = 0i32;
        let mut best_psy_cost = 0i32;
        // Raw winning-mode source-vs-reconstruction SSE *before* the flat-source
        // doubling. This is the `D` libwebp's `quant_enc.c:1111` per-MB gate
        // compares against `dqm->min_disto`. Threaded through MacroblockInfo
        // so `store_max_delta` can apply the gate (issue #44).
        let mut best_d_raw = 0u32;

        // Pre-allocate scratch buffers outside mode loop to avoid redundant zero-init.
        // All elements are written before read in each iteration.
        let mut luma_blocks = [0i32; 256];
        let mut y1_quant = [[0i32; 16]; 16];
        // Zigzag-scan-order views used only for the coefficient rate
        // (`get_cost_luma16`) and the flatness test — libwebp codes `y_dc_levels`
        // / `y_ac_levels` in zigzag order and `VP8GetCostLuma16` / `IsFlat` walk
        // that order. Natural (raster) order stays in `y1_quant` / `y2_dequant`
        // for the IDCT reconstruction path. Matches the I4 path (#38).
        let mut y1_quant_zz = [[0i32; 16]; 16];
        let mut y2_quant_zz = [0i32; 16];
        let mut recon_dequant_block = [0i32; 16];
        let mut rec_block = [0u8; 256];
        let mut all_levels = [0i16; 256];

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // All four I16 modes (DC, V, H, TM) are evaluated at every position,
            // including MB borders. `create_border_luma` substitutes 127 above and
            // 129 left when the neighbour is out of frame; RD scoring picks the
            // winner. Matches libwebp's PickBestIntra16 (`quant_enc.c:1072-1100`),
            // which never skips a mode based on position. Removed in #28 to allow
            // the border-value-padded prediction to compete on top-row, left-column,
            // and corner MBs (small images and frame edges).

            // Generate prediction for this mode
            let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);

            // === Full reconstruction for RD evaluation ===
            // 1. Compute residuals and forward DCT (luma_blocks hoisted outside mode loop)
            self.fill_luma_blocks_from_predicted_16x16(&pred, mbx, mby, &mut luma_blocks);

            // 2. Extract DC coefficients and do WHT
            let mut dc_coeffs = [0i32; 16];
            for (i, dc) in dc_coeffs.iter_mut().enumerate() {
                *dc = luma_blocks[i * 16];
            }
            let mut y2_coeffs = dc_coeffs;
            transform::wht4x4(&mut y2_coeffs);

            // 3. Quantize Y2 (DC) coefficients using SIMD
            let mut y2_quant = y2_coeffs;
            crate::encoder::quantize::quantize_block_simd(&mut y2_quant, y2_matrix, true);

            // 4. Quantize Y1 (AC) coefficients
            // At m6 (RD_OPT_TRELLIS_ALL) the I16 mode-selection path also runs
            // trellis quantization — see libwebp's ReconstructIntra16
            // (`quant_enc.c:830-879`), which switches on `DO_TRELLIS_I16` and
            // calls `TrellisQuantizeBlock` instead of `VP8EncQuantizeBlock`.
            // The resulting levels feed both the cost estimate (step 5) and the
            // distortion measurement (step 7) so both halves of the RD score
            // see the trellis-optimized coefficients.
            // (y1_quant hoisted outside mode loop to avoid redundant zero-init)
            if self.do_trellis_i4_mode {
                // m6 RD_OPT_TRELLIS_ALL: use trellis quantization for I16 AC blocks.
                // Track per-block has_nz context locally — `get_cost_luma16` itself
                // resets these to all-false at function entry, so the ctx0 we feed
                // into the trellis call needs to mirror that ordering exactly.
                let mut top_nz_t = [false; 4];
                let mut left_nz_t = [false; 4];
                #[allow(clippy::needless_range_loop)]
                for block_idx in 0..16 {
                    let bx = block_idx % 4;
                    let by = block_idx / 4;
                    let block_start = block_idx * 16;
                    let mut coeffs: [i32; 16] = luma_blocks[block_start..block_start + 16]
                        .try_into()
                        .unwrap();
                    coeffs[0] = 0; // DC handled by Y2
                    let ctx0 = (u8::from(top_nz_t[bx]) + u8::from(left_nz_t[by])).min(2) as usize;
                    let mut zigzag_levels = [0i32; 16];
                    let has_nz = trellis_quantize_block(
                        &mut coeffs,
                        &mut zigzag_levels,
                        y1_matrix,
                        segment.lambda_trellis_i16,
                        1, // first=1 for I16_AC (DC lives in Y2)
                        &self.level_costs,
                        0, // ctype=0 for I16_AC
                        ctx0,
                        &segment.psy_config,
                    );
                    top_nz_t[bx] = has_nz;
                    left_nz_t[by] = has_nz;
                    // Convert zigzag-ordered levels back to natural index so the
                    // downstream `y1_matrix.dequantize(level, i)` and cost paths
                    // see the same storage convention as the simple-quant branch.
                    let mut natural = [0i32; 16];
                    for n in 1..16 {
                        natural[crate::encoder::tables::VP8_ZIGZAG[n]] = zigzag_levels[n];
                    }
                    y1_quant[block_idx] = natural;
                }
            } else {
                // m0..m5: simple SIMD quantization (no trellis during mode selection)
                #[allow(clippy::needless_range_loop)]
                for block_idx in 0..16 {
                    // Copy block from luma_blocks (DC will be zeroed by quantize_ac_only)
                    let block_start = block_idx * 16;
                    let mut block: [i32; 16] = luma_blocks[block_start..block_start + 16]
                        .try_into()
                        .unwrap();
                    block[0] = 0; // DC is handled by Y2
                    crate::encoder::quantize::quantize_ac_only_simd(&mut block, y1_matrix, true);
                    y1_quant[block_idx] = block;
                }
            }

            // 5. Compute coefficient cost using probability-dependent tables.
            // libwebp's VP8GetCostLuma16 imports cross-MB non-zero context from
            // `it->top_nz[]`/`it->left_nz[]` (slot 8 reserved for Y2 DC) and walks
            // each AC block updating local state. Mirrored here using zenwebp's
            // top_complexity/left_complexity (#23).
            let top_y_nz = [
                self.top_complexity[mbx].y[0] != 0,
                self.top_complexity[mbx].y[1] != 0,
                self.top_complexity[mbx].y[2] != 0,
                self.top_complexity[mbx].y[3] != 0,
            ];
            let left_y_nz = [
                self.left_complexity.y[0] != 0,
                self.left_complexity.y[1] != 0,
                self.left_complexity.y[2] != 0,
                self.left_complexity.y[3] != 0,
            ];
            let top_y2_nz = self.top_complexity[mbx].y2 != 0;
            let left_y2_nz = self.left_complexity.y2 != 0;

            // Build zigzag-scan-order views for the coefficient rate (see the
            // `y1_quant_zz` / `y2_quant_zz` declarations). Natural order stays in
            // `y1_quant` / `y2_dequant` for the IDCT reconstruction below.
            for n in 0..16 {
                y2_quant_zz[n] = y2_quant[ZIGZAG[n] as usize];
            }
            for (blk, blk_zz) in y1_quant.iter().zip(y1_quant_zz.iter_mut()) {
                for n in 0..16 {
                    blk_zz[n] = blk[ZIGZAG[n] as usize];
                }
            }
            let coeff_cost = get_cost_luma16(
                &y2_quant_zz,
                &y1_quant_zz,
                top_y_nz,
                left_y_nz,
                top_y2_nz,
                left_y2_nz,
                &self.level_costs,
                probs,
            );

            // 6. Dequantize Y2 and do inverse WHT using SIMD
            let mut y2_dequant = y2_quant;
            y2_matrix.dequantize_block(&mut y2_dequant);
            transform::iwht4x4(&mut y2_dequant);

            // 7. Dequantize Y1, add DC from Y2, and do fused inverse DCT + add residue
            let mut reconstructed = pred;
            for block_idx in 0..16 {
                let bx = block_idx % 4;
                let by = block_idx / 4;

                // AC from Y1 (recon_dequant_block hoisted outside mode loop)
                for i in 1..16 {
                    recon_dequant_block[i] = y1_matrix.dequantize(y1_quant[block_idx][i], i);
                }
                // DC from Y2
                recon_dequant_block[0] = y2_dequant[block_idx];

                // Fused IDCT + add residue to prediction (reads pred, adds IDCT, clamps, stores)
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = recon_dequant_block[1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut recon_dequant_block,
                    &mut reconstructed,
                    y0,
                    x0,
                    LUMA_STRIDE,
                    dc_only,
                );
            }

            // 8. Compute SSE between source and reconstructed (NOT prediction!)
            let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &reconstructed);

            // 9. Compute spectral distortion (TDisto) + psy-rd if enabled
            let (spectral_disto, psy_cost) = if let Some(ref src_block) = src_block {
                // Extract reconstructed block only (source already cached)
                // (rec_block hoisted outside mode loop to avoid redundant zero-init)
                for y in 0..16 {
                    let rec_row = (y + 1) * LUMA_STRIDE + 1;
                    rec_block[y * 16..(y + 1) * 16]
                        .copy_from_slice(&reconstructed[rec_row..rec_row + 16]);
                }

                let td = if tlambda > 0 {
                    let td_raw =
                        tdisto_16x16(src_block, &rec_block, 16, &segment.psy_config.luma_csf);
                    (tlambda as i32 * td_raw + 128) >> 8
                } else {
                    0
                };

                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_16x16(src_block, 16);
                    let rec_satd = psy::satd_16x16(&rec_block, 16);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };

                (td, psy)
            } else {
                (0, 0)
            };

            // 10. Apply flat source penalty if applicable
            let (d_final, sd_final) = if is_flat {
                // Check if coefficients are also flat
                // (all_levels hoisted outside mode loop to avoid redundant zero-init)
                for block_idx in 0..16 {
                    for i in 1..16 {
                        all_levels[block_idx * 16 + i] = y1_quant_zz[block_idx][i] as i16;
                    }
                }
                if is_flat_coeffs(&all_levels, 16, FLATNESS_LIMIT_I16) {
                    // Double distortion to penalize I16 for flat sources
                    (sse * 2, spectral_disto * 2)
                } else {
                    (sse, spectral_disto)
                }
            } else {
                (sse, spectral_disto)
            };

            // 11. Compute full RD score
            // score = (R + H) * lambda + RD_DISTO_MULT * (D + SD + PSY)
            let mode_cost = FIXED_COSTS_I16[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost)) * i64::from(lambda);
            let distortion = i64::from(RD_DISTO_MULT)
                * (i64::from(d_final) + i64::from(sd_final) + i64::from(psy_cost));
            let rd_score = rate + distortion;

            #[cfg(feature = "mode_debug")]
            if debug_i16 {
                eprintln!(
                    "  I16 {:?}: H={}, R={}, D={}, SD={}, rate={}, disto={}, score={}",
                    mode, mode_cost, coeff_cost, d_final, sd_final, rate, distortion, rd_score
                );
            }

            let is_better = rd_score < best_rd_score
                || (parity_tiebreak
                    && rd_score == best_rd_score
                    && I16_LIB_RANK[mode_idx] < best_lib_rank);
            if is_better {
                best_rd_score = rd_score;
                best_mode = mode;
                best_lib_rank = I16_LIB_RANK[mode_idx];
                // Store components for final score recalculation
                best_coeff_cost = coeff_cost;
                best_mode_cost = mode_cost;
                best_sse = d_final;
                best_spectral_disto = sd_final;
                best_psy_cost = psy_cost;
                // Raw SSE (pre flat-source doubling) for the #44 gate.
                best_d_raw = sse;
            }
        }

        // Recalculate final score using lambda_mode for I4 vs I16 comparison
        // This matches libwebp's: SetRDScore(dqm->lambda_mode, rd);
        let lambda_mode = segment.lambda_mode;
        let final_rate =
            (i64::from(best_mode_cost) + i64::from(best_coeff_cost)) * i64::from(lambda_mode);
        let final_distortion = i64::from(RD_DISTO_MULT)
            * (i64::from(best_sse) + i64::from(best_spectral_disto) + i64::from(best_psy_cost));
        let final_score = final_rate + final_distortion;

        #[cfg(feature = "mode_debug")]
        if debug_i16 {
            eprintln!(
                "  I16 FINAL: mode={:?}, H={}, R={}, D={}, SD={}, lambda_mode={}, tlambda={}, rate={}, disto={}, score={}",
                best_mode,
                best_mode_cost,
                best_coeff_cost,
                best_sse,
                best_spectral_disto,
                lambda_mode,
                tlambda,
                final_rate,
                final_distortion,
                final_score
            );
        }

        // Convert to u64 for interface compatibility (score should be positive)
        (best_mode, final_score.max(0) as u64, best_d_raw)
    }

    /// Fast DC-only mode selection for method 0.
    ///
    /// Uses simplified scoring (SSE + fixed mode cost) without full reconstruction.
    /// This is much faster than the full RD path but may not find the optimal mode.
    ///
    /// Returns `(mode, score, d_raw)`. `d_raw` is the prediction-vs-source SSE,
    /// reused as the `D` proxy for the #44 `D > min_disto` gate (the fast path
    /// has no quantize+reconstruct step). At m0/m1, residue energy approximates
    /// reconstruction error closely enough for the gate to be sound — flat
    /// regions still produce small `D` and stay below `min_disto`, while
    /// real-edge MBs produce large `D` and pass through.
    fn pick_intra16_fast_dc(&self, mbx: usize, mby: usize) -> (LumaMode, u64, u32) {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;
        let segment = self.get_segment_for_mb(mbx, mby);
        let lambda = segment.lambda_i16;

        // For method 0, we use DC mode with simple SSE scoring
        let pred = self.get_predicted_luma_block_16x16(LumaMode::DC, mbx, mby);
        let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &pred);

        // Simple score: SSE + lambda * mode_cost
        // DC mode has the lowest fixed cost (FIXED_COSTS_I16[0])
        let mode_cost = FIXED_COSTS_I16[0] as u32;
        let score = u64::from(sse) + u64::from(lambda) * u64::from(mode_cost);

        (LumaMode::DC, score, sse)
    }

    /// libwebp `RefineUsingDistortion` Intra16 arm at RD_OPT_NONE (m0/m1):
    /// score all four I16 modes by plain SSE plus the fixed mode cost weighted
    /// by libwebp's empirical constant `lambda_d_i16 = 106` (quant_enc.c:1266)
    /// — no per-mode quantization or coefficient-cost machinery. Includes the
    /// libwebp bug-#432 border guard: flat sources on the first row/column are
    /// pinned to DC/V to avoid seeding a checkerboard resonance.
    ///
    /// Returns the winning mode and its SSE (feeds the #44 `D > min_disto`
    /// skip gate via `MacroblockInfo::intra16_d`).
    fn pick_intra16_sse(&self, mbx: usize, mby: usize) -> (LumaMode, u32, u64, bool) {
        const LAMBDA_D_I16: u64 = 106;
        // libwebp's RefineUsingDistortion iterates modes in VP8 index order
        // (0 = DC, 1 = TM, 2 = V, 3 = H) with a strict `<`, so an exact SSE
        // tie resolves to the lowest mode index. Match that order (not
        // zenwebp's `[DC, V, H, TM]`) so tie-breaking agrees, and pair each
        // mode with its cost from VP8FixedCostsI16 in the same order.
        const MODES: [LumaMode; 4] = [LumaMode::DC, LumaMode::TM, LumaMode::V, LumaMode::H];
        const COSTS: [u16; 4] = [663, 919, 872, 919]; // VP8FixedCostsI16, mode order
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        let mut best_mode = LumaMode::DC;
        let mut best_sse = 0u32;
        let mut best_score = u64::MAX;
        for (idx, &mode) in MODES.iter().enumerate() {
            let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);
            let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &pred);
            let score =
                u64::from(sse) * u64::from(RD_DISTO_MULT) + u64::from(COSTS[idx]) * LAMBDA_D_I16;
            if score < best_score {
                best_score = score;
                best_mode = mode;
                best_sse = sse;
            }
        }

        let mut flat_locked = false;
        if mbx == 0 || mby == 0 {
            let src_base = mby * 16 * src_width + mbx * 16;
            if is_flat_source_16(&self.frame.ybuf[src_base..], src_width) {
                // libwebp: best_mode = (it->x == 0) ? DC : V, and
                // try_both_modes = 0 ("stick to i16") — bug #432.
                best_mode = if mbx == 0 { LumaMode::DC } else { LumaMode::V };
                let pred = self.get_predicted_luma_block_16x16(best_mode, mbx, mby);
                best_sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &pred);
                flat_locked = true;
            }
        }

        (best_mode, best_sse, best_score, flat_locked)
    }

    /// libwebp `RefineUsingDistortion` Intra4 arm at RD_OPT_NONE (m0/m1):
    /// each sub-block's mode is picked from all 10 candidates by plain SSE
    /// plus the context-dependent fixed mode cost weighted by libwebp's
    /// `lambda_d_i4 = 11` (quant_enc.c:1267). Only the winning mode is
    /// transformed/quantized (once, for the reconstruction that feeds the
    /// next sub-block's predictors) — the RD path quantizes every candidate.
    /// No bail-out to I16: with `try_both_modes == 0` libwebp's early-exit
    /// threshold is MAX_COST, so the analysis hint fully decides I16 vs I4.
    fn pick_intra4_sse(&self, mbx: usize, mby: usize) -> [IntraMode; 16] {
        // No-bail variant (m0/m1: libwebp try_both_modes == 0 semantics).
        self.pick_intra4_sse_with_bail(mbx, mby, u64::MAX, 0, u64::MAX)
            .expect("no-bail I4 pick always returns modes")
    }

    /// SSE-scored Intra4 pick with libwebp's accumulate-and-bail (m2,
    /// try_both_modes == 1): `score_i4` starts at `i4_penalty` and grows by
    /// each sub-block's best SSE-domain score; returns `None` (use I16) when
    /// it reaches `i16_score` or the header-bit sum exceeds `bit_limit`.
    fn pick_intra4_sse_with_bail(
        &self,
        mbx: usize,
        mby: usize,
        i16_score: u64,
        i4_penalty: u64,
        bit_limit: u64,
    ) -> Option<[IntraMode; 16]> {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;
        let mut y_with_border =
            create_border_luma(mbx, mby, mbw, &self.top_border_y, &self.left_border_y);
        let segment = self.get_segment_for_mb(mbx, mby);
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();

        let top_ctx0: [usize; 4] =
            core::array::from_fn(|sbx| self.top_b_pred[mbx * 4 + sbx] as usize);
        let left_ctx0: [usize; 4] = core::array::from_fn(|sby| self.left_b_pred[sby] as usize);

        // Single-arcane hoist (739bc14 recipe): the whole sub-block loop runs
        // inside one target_feature region so the #[rite] SSE/transform/quant
        // primitives inline and values stay in registers — the dispatched
        // version paid ~14 arcane boundaries per sub-block.
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            return pick_intra4_sse_arcane(
                token,
                &self.frame.ybuf,
                src_width,
                mbx,
                mby,
                &mut y_with_border,
                y1_matrix,
                &top_ctx0,
                &left_ctx0,
                i16_score,
                i4_penalty,
                bit_limit,
            );
        }

        pick_intra4_sse_portable(
            &self.frame.ybuf,
            src_width,
            mbx,
            mby,
            &mut y_with_border,
            y1_matrix,
            &top_ctx0,
            &left_ctx0,
            i16_score,
            i4_penalty,
            bit_limit,
        )
    }

    /// Estimate coefficient cost for a 16x16 luma macroblock (I16 mode).
    ///
    /// Quantizes coefficients and estimates their encoding cost without
    /// permanently modifying state.
    fn estimate_luma16_coeff_cost(&self, luma_blocks: &[i32; 256], segment: &Segment) -> u32 {
        let mut total_cost = 0u32;

        // Extract DC coefficients and estimate Y2 (DC transform) cost
        let mut dc_coeffs = [0i32; 16];
        for (i, dc) in dc_coeffs.iter_mut().enumerate() {
            *dc = luma_blocks[i * 16];
        }

        // WHT transform on DC coefficients
        let mut y2_coeffs = dc_coeffs;
        transform::wht4x4(&mut y2_coeffs);

        // Quantize Y2 coefficients and estimate cost
        for (idx, coeff) in y2_coeffs.iter_mut().enumerate() {
            let quant = if idx > 0 { segment.y2ac } else { segment.y2dc };
            *coeff /= i32::from(quant);
        }
        total_cost += estimate_dc16_cost(&y2_coeffs);

        // Estimate AC coefficient cost for each 4x4 block (skip DC at index 0)
        for block_idx in 0..16 {
            let block_start = block_idx * 16;
            let mut block = [0i32; 16];

            // Copy and quantize AC coefficients (DC is handled separately in I16 mode)
            for (i, coeff) in block.iter_mut().enumerate() {
                if i == 0 {
                    *coeff = 0; // DC is in Y2 block
                } else {
                    *coeff = luma_blocks[block_start + i] / i32::from(segment.yac);
                }
            }

            // Estimate cost (starting from position 1, DC is separate)
            total_cost += estimate_residual_cost(&block, 1);
        }

        total_cost
    }

    /// Apply a 4x4 intra prediction mode to the working buffer
    #[allow(dead_code)] // Kept for future Intra4 mode selection with coefficient-level costs
    fn apply_intra4_prediction(
        ws: &mut [u8; LUMA_BLOCK_SIZE],
        mode: IntraMode,
        x0: usize,
        y0: usize,
    ) {
        let stride = LUMA_STRIDE;
        match mode {
            IntraMode::TM => predict_tmpred(ws, 4, x0, y0, stride),
            IntraMode::VE => predict_bvepred(ws, x0, y0, stride),
            IntraMode::HE => predict_bhepred(ws, x0, y0, stride),
            IntraMode::DC => predict_bdcpred(ws, x0, y0, stride),
            IntraMode::LD => predict_bldpred(ws, x0, y0, stride),
            IntraMode::RD => predict_brdpred(ws, x0, y0, stride),
            IntraMode::VR => predict_bvrpred(ws, x0, y0, stride),
            IntraMode::VL => predict_bvlpred(ws, x0, y0, stride),
            IntraMode::HD => predict_bhdpred(ws, x0, y0, stride),
            IntraMode::HU => predict_bhupred(ws, x0, y0, stride),
        }
    }

    /// Compute SSE for a 4x4 subblock between source image and prediction buffer
    #[allow(dead_code)] // Kept for future Intra4 mode selection with coefficient-level costs
    fn sse_4x4_subblock(
        &self,
        pred: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
        sbx: usize,
        sby: usize,
    ) -> u32 {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        let mut sse = 0u32;
        let pred_y0 = sby * 4 + 1;
        let pred_x0 = sbx * 4 + 1;
        let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;

        for y in 0..4 {
            let pred_row = (pred_y0 + y) * LUMA_STRIDE + pred_x0;
            let src_row = src_base + y * src_width;
            for x in 0..4 {
                let diff = i32::from(self.frame.ybuf[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
    }

    /// Select the best Intra4 modes for all 16 subblocks using accurate coefficient cost estimation.
    ///
    /// Returns `Some((modes, rd_score))` if Intra4 is better than `i16_score`,
    /// or `None` if Intra16 should be used (early-exit optimization).
    ///
    /// The comparison includes an i4_penalty (1000 * q²) to account for the
    /// typically higher bit cost of Intra4 mode signaling.
    ///
    /// Uses proper probability-based coefficient cost estimation ported from
    /// libwebp's VP8GetCostLuma4 with remapped_costs tables.
    fn pick_best_intra4(
        &self,
        mbx: usize,
        mby: usize,
        i16_score: u64,
    ) -> Option<([IntraMode; 16], u64)> {
        // Check for debug mode
        #[cfg(feature = "mode_debug")]
        let debug_i4 = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let _debug_i4 = false;

        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // All 10 intra4 modes (used by SIMD path on x86_64 and wasm32)
        #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
        const MODES: [IntraMode; 10] = [
            IntraMode::DC,
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::LD,
            IntraMode::RD,
            IntraMode::VR,
            IntraMode::VL,
            IntraMode::HD,
            IntraMode::HU,
        ];

        let mut best_modes = [IntraMode::DC; 16];
        let mut best_mode_indices = [0usize; 16]; // Track indices for context lookup

        // Create working buffer with border
        let mut y_with_border =
            create_border_luma(mbx, mby, mbw, &self.top_border_y, &self.left_border_y);

        let segment = self.get_segment_for_mb(mbx, mby);

        // Get quantizer-dependent lambdas for I4 mode RD scoring
        // lambda_i4 is used for selecting the best mode within each block
        // lambda_mode is used for accumulation and comparison against I16 score
        // tlambda is used for spectral distortion (TDisto) weighting
        // (This matches libwebp's SetRDScore flow in PickBestIntra4)
        let lambda_i4 = segment.lambda_i4;
        let lambda_mode = segment.lambda_mode;
        let tlambda = segment.tlambda;

        // Initialize I4 running score with an I4 penalty
        // libwebp uses i4_penalty = 1000 * q² with fixed lambda_d_i4 = 11
        // Our scoring system uses q-dependent lambdas, so we need to scale differently
        //
        // Approach: Use a penalty proportional to lambda_mode (which is q²/128)
        // This gives: penalty ≈ SCALE * q² where SCALE is tuned empirically
        //
        // libwebp's PickBestIntra4 (`quant_enc.c:1144-1145`) initializes the
        // running score with just `H = 211` (the bit-cost of the is_intra4 flag,
        // i.e. `VP8BitCost(0, 145) = 211`) — no separate i4_penalty:
        //
        //     rd_best.H = 211;  // 211 = VP8BitCost(0, 145)
        //     SetRDScore(dqm->lambda_mode, &rd_best);  // running_score = 211 * lambda_mode
        //
        // (The `1000 * q²` `i4_penalty` in `quant_enc.c:267` is only used by
        // `RefineUsingDistortion` at m0/m1, not by `PickBestIntra4` at m2+.)
        //
        // zenwebp previously used `3000 * lambda_mode` which is ~14× larger
        // than libwebp's value, biasing strongly away from I4 and triggering
        // the early-exit `running_score >= i16_score` even when I4 would
        // actually win. Fixed in #22.
        //
        // partition_limit (0-100) still scales the penalty up to prevent
        // partition-0 overflow on very large images at high limits — that
        // mechanism is orthogonal to the libwebp default and remains in
        // place; only the base constant changes from 3000 to 211.
        const H_INTRA4_BIT_COST: u64 = 211;
        let base_penalty = H_INTRA4_BIT_COST * u64::from(lambda_mode);
        let limit_scale = u64::from(self.partition_limit); // 0-100
        let i4_penalty = base_penalty + base_penalty * limit_scale * limit_scale / 400;
        let mut running_score = i4_penalty;

        #[cfg(feature = "mode_debug")]
        if debug_i4 {
            eprintln!(
                "  I4 i4_penalty={}, running_score={}",
                i4_penalty, running_score
            );
        }

        // Track total mode cost for header bit limiting
        let mut total_mode_cost = 0u32;
        // Maximum header bits for I4 modes (from libwebp)
        let max_header_bits: u32 = 256 * 16 * 16 / 4;

        // Track non-zero context for accurate coefficient cost estimation
        // top_nz[x] = whether block above has non-zero coefficients
        // left_nz[y] = whether block to left has non-zero coefficients
        // Initialize from cross-macroblock context (top_complexity/left_complexity)
        // so edge blocks use the correct context from neighboring macroblocks
        let mut top_nz = [
            self.top_complexity[mbx].y[0] != 0,
            self.top_complexity[mbx].y[1] != 0,
            self.top_complexity[mbx].y[2] != 0,
            self.top_complexity[mbx].y[3] != 0,
        ];
        let mut left_nz = [
            self.left_complexity.y[0] != 0,
            self.left_complexity.y[1] != 0,
            self.left_complexity.y[2] != 0,
            self.left_complexity.y[3] != 0,
        ];

        // Get probability tables for coefficient cost estimation
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Process each subblock in raster order
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                // Get mode context from neighboring blocks
                // For edge blocks, use cross-macroblock context from previous MB's I4 modes
                let top_ctx = if sby == 0 {
                    // Top edge: use mode from macroblock above (stored in top_b_pred)
                    // Index: mbx * 4 + sbx gives the correct column's top context
                    self.top_b_pred[mbx * 4 + sbx] as usize
                } else {
                    best_mode_indices[(sby - 1) * 4 + sbx]
                };
                let left_ctx = if sbx == 0 {
                    // Left edge: use mode from macroblock to the left (stored in left_b_pred)
                    // Index: sby gives the correct row's left context
                    self.left_b_pred[sby] as usize
                } else {
                    best_mode_indices[sby * 4 + (sbx - 1)]
                };

                // Get non-zero context from neighboring blocks
                // For edge blocks (sby==0 or sbx==0), the cross-macroblock context
                // is already in top_nz/left_nz from initialization above
                let nz_top = top_nz[sbx];
                let nz_left = left_nz[sby];

                // Precompute mode costs for all 10 modes (context is constant for this block)
                // This avoids repeated 3D table lookup inside the tight mode loop
                let mode_costs: [u16; 10] = core::array::from_fn(|mode_idx| {
                    crate::encoder::tables::VP8_FIXED_COSTS_I4[top_ctx][left_ctx][mode_idx]
                });

                let mut best_mode = IntraMode::DC;
                let mut best_mode_idx = 0usize;
                let mut best_block_score = u64::MAX;
                let mut best_has_nz = false;
                // Save best dequantized+IDCT result to avoid recomputing in post-loop
                let mut best_dequantized = [0i32; 16];
                // Track best block's SSE, spectral distortion, psy cost, and coeff cost for recalculating with lambda_mode
                let mut best_sse = 0u32;
                let mut best_spectral_disto = 0i32;
                let mut best_psy_cost = 0i32;
                let mut best_coeff_cost = 0u32;

                // Pre-compute all 10 I4 prediction modes at once
                let preds = I4Predictions::compute(&y_with_border, x0, y0, LUMA_STRIDE);

                // Compute source block once (row-wise copy for better cache/vectorization)
                let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;
                let mut src_block = [0u8; 16];
                for y in 0..4 {
                    let src_row = src_base + y * src_width;
                    src_block[y * 4..y * 4 + 4]
                        .copy_from_slice(&self.frame.ybuf[src_row..src_row + 4]);
                }

                let y1_matrix = segment.y1_matrix.as_ref().unwrap();

                // Number of modes to try depends on method:
                // - method 0-2: 3 modes (fast, RD_OPT_NONE equivalent)
                // - method 3+: 10 modes (full search, matches libwebp RD_OPT_BASIC+)
                let max_modes_to_try = match self.method {
                    0..=2 => 3,
                    _ => 10, // method 3+: try all modes (matches libwebp RD_OPT_BASIC+)
                };

                // Get trellis lambda if enabled for mode selection (method >= 6)
                let trellis_lambda_i4 = if self.do_trellis_i4_mode {
                    Some(segment.lambda_trellis_i4)
                } else {
                    None
                };

                // === SIMD-hoisted path: single arcane context for all mode evaluation ===
                // This eliminates per-call dispatch overhead by running all SIMD operations
                // (ftransform, quantize, dequantize, sse, tdisto, is_flat, residual_cost)
                // within a single #[target_feature] context where they can be inlined.
                #[cfg(target_arch = "x86_64")]
                let simd_result = {
                    use archmage::SimdToken;
                    archmage::X64V3Token::summon().and_then(|token| {
                        // Pre-sort modes by prediction SSE using direct SIMD calls
                        let mode_sse = presort_i4_modes_sse2(token, &src_block, &preds);

                        // Evaluate all candidate modes in a single arcane context
                        evaluate_i4_modes_sse2(
                            token,
                            &src_block,
                            &preds,
                            &mode_sse,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            &self.level_costs,
                            probs,
                            &segment.psy_config,
                            &segment.psy_config.luma_csf,
                        )
                    })
                };

                #[cfg(target_arch = "x86_64")]
                if let Some(result) = simd_result {
                    // Use the arcane result
                    best_mode = MODES[result.mode_idx];
                    best_mode_idx = result.mode_idx;
                    let _ = result.rd_score; // best_block_score not needed after eval
                    best_has_nz = result.has_nz;
                    best_dequantized = result.dequantized;
                    best_sse = result.sse;
                    best_spectral_disto = result.spectral_disto;
                    best_psy_cost = result.psy_cost;
                    best_coeff_cost = result.coeff_cost;
                } else {
                    // Scalar fallback (no SIMD available at runtime or not x86_64)
                    #[cfg(target_arch = "x86_64")]
                    {
                        self.evaluate_i4_modes_scalar(
                            &src_block,
                            &preds,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            probs,
                            segment,
                            &mut best_mode,
                            &mut best_mode_idx,
                            &mut best_block_score,
                            &mut best_has_nz,
                            &mut best_dequantized,
                            &mut best_sse,
                            &mut best_spectral_disto,
                            &mut best_psy_cost,
                            &mut best_coeff_cost,
                        );
                    }
                }

                // === WASM SIMD128 path ===
                #[cfg(target_arch = "wasm32")]
                let simd_result = {
                    use archmage::SimdToken;
                    archmage::Wasm128Token::summon().and_then(|token| {
                        let mode_sse = presort_i4_modes_wasm(token, &src_block, &preds);

                        evaluate_i4_modes_wasm(
                            token,
                            &src_block,
                            &preds,
                            &mode_sse,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            &self.level_costs,
                            probs,
                            &segment.psy_config,
                            &segment.psy_config.luma_csf,
                        )
                    })
                };

                #[cfg(target_arch = "wasm32")]
                if let Some(result) = simd_result {
                    best_mode = MODES[result.mode_idx];
                    best_mode_idx = result.mode_idx;
                    let _ = result.rd_score;
                    best_has_nz = result.has_nz;
                    best_dequantized = result.dequantized;
                    best_sse = result.sse;
                    best_spectral_disto = result.spectral_disto;
                    best_psy_cost = result.psy_cost;
                    best_coeff_cost = result.coeff_cost;
                } else {
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.evaluate_i4_modes_scalar(
                            &src_block,
                            &preds,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            probs,
                            segment,
                            &mut best_mode,
                            &mut best_mode_idx,
                            &mut best_block_score,
                            &mut best_has_nz,
                            &mut best_dequantized,
                            &mut best_sse,
                            &mut best_spectral_disto,
                            &mut best_psy_cost,
                            &mut best_coeff_cost,
                        );
                    }
                }

                // Fallback: no SIMD available
                #[cfg(not(any(target_arch = "x86_64", target_arch = "wasm32")))]
                {
                    self.evaluate_i4_modes_scalar(
                        &src_block,
                        &preds,
                        max_modes_to_try,
                        &mode_costs,
                        nz_top,
                        nz_left,
                        y1_matrix,
                        lambda_i4,
                        tlambda,
                        trellis_lambda_i4,
                        probs,
                        segment,
                        &mut best_mode,
                        &mut best_mode_idx,
                        &mut best_block_score,
                        &mut best_has_nz,
                        &mut best_dequantized,
                        &mut best_sse,
                        &mut best_spectral_disto,
                        &mut best_psy_cost,
                        &mut best_coeff_cost,
                    );
                }

                best_modes[i] = best_mode;
                best_mode_indices[i] = best_mode_idx;

                // Update non-zero context for subsequent blocks
                top_nz[sbx] = best_has_nz;
                left_nz[sby] = best_has_nz;

                let best_mode_cost = mode_costs[best_mode_idx];
                total_mode_cost += u32::from(best_mode_cost);

                // Recalculate the block score with lambda_mode for accumulation
                // (matching libwebp's SetRDScore(lambda_mode, &rd_i4) before AddScore)
                let block_score_for_comparison = crate::encoder::cost::rd_score_full(
                    best_sse,
                    best_spectral_disto + best_psy_cost,
                    best_mode_cost,
                    best_coeff_cost,
                    lambda_mode,
                ) as u64;

                #[cfg(feature = "mode_debug")]
                if debug_i4 {
                    eprintln!(
                        "  I4 blk[{:2}]: mode={:?}, H={}, R={}, D={}, SD={}, block_score={}, running={}",
                        i,
                        best_mode,
                        best_mode_cost,
                        best_coeff_cost,
                        best_sse,
                        best_spectral_disto,
                        block_score_for_comparison,
                        running_score + block_score_for_comparison
                    );
                }

                // Add this block's score to running total
                running_score += block_score_for_comparison;

                // Early-exit: if I4 already exceeds I16, bail out
                if running_score >= i16_score {
                    return None;
                }

                // Check header bit limit
                if total_mode_cost > max_header_bits {
                    return None;
                }

                // Apply the selected mode and reconstruct for next blocks
                Self::apply_intra4_prediction(&mut y_with_border, best_mode, x0, y0);

                // Add back saved dequantized+IDCT result (already computed in inner loop)
                // This eliminates a redundant dequantize + IDCT per block
                add_residue(&mut y_with_border, &best_dequantized, y0, x0, LUMA_STRIDE);
            }
        }

        // I4 wins! Return the modes and final score
        #[cfg(feature = "mode_debug")]
        if debug_i4 {
            eprintln!(
                "  I4 FINAL: score={}, i16_score={}, margin={}",
                running_score,
                i16_score,
                i16_score as i64 - running_score as i64
            );
        }

        Some((best_modes, running_score))
    }

    /// libwebp `RefineUsingDistortion` UV arm (refine_uv_mode, m1/m2): pick
    /// among the four chroma modes by plain SSE over both planes plus the
    /// fixed mode cost weighted by `lambda_d_uv = 120` (quant_enc.c:1268) —
    /// no per-mode quantization.
    fn pick_uv_sse(&self, mbx: usize, mby: usize) -> ChromaMode {
        const LAMBDA_D_UV: u64 = 120;
        // Order matches FIXED_COSTS_UV.
        const MODES: [ChromaMode; 4] =
            [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];
        let mbw = usize::from(self.macroblock_width);
        let chroma_width = mbw * 8;

        let mut best_mode = ChromaMode::DC;
        let mut best_score = u64::MAX;
        for (idx, &mode) in MODES.iter().enumerate() {
            // libwebp's RefineUsingDistortion evaluates ALL four chroma modes,
            // including V/H/TM at frame edges — there the prediction reads the
            // default borders (top=127, left=129), which the decoder uses too,
            // so an edge V/H/TM pick is legal and round-trips. (Skipping them
            // here diverged from libwebp at edge MBs — the m1/m2 UV gap.)
            let pred_u = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_u,
                &self.left_border_u,
            );
            let pred_v = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_v,
                &self.left_border_v,
            );
            let sse = sse_8x8_chroma(&self.frame.ubuf, chroma_width, mbx, mby, &pred_u)
                + sse_8x8_chroma(&self.frame.vbuf, chroma_width, mbx, mby, &pred_v);
            let score = u64::from(sse) * u64::from(RD_DISTO_MULT)
                + u64::from(FIXED_COSTS_UV[idx]) * LAMBDA_D_UV;
            if score < best_score {
                best_score = score;
                best_mode = mode;
            }
        }
        best_mode
    }

    /// Select the best chroma (UV) prediction mode using full RD scoring.
    ///
    /// This implements libwebp's full RD path for PickBestUV:
    /// 1. For each mode: generate prediction, forward DCT, quantize
    /// 2. Dequantize and inverse DCT to get reconstructed block
    /// 3. Compute SSE between reconstructed and source (NOT prediction vs source!)
    /// 4. Include coefficient cost in rate term
    /// 5. Apply flatness penalty for non-DC modes with flat coefficients
    ///
    /// RD formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD + PSY)
    /// UV spectral distortion and psy-rd are enabled by PsyConfig at method >= 3.
    #[allow(clippy::needless_range_loop)] // block_idx used for both indexing and coordinate computation
    /// The chroma mode chosen by the analysis pass (`MBAnalyzeBestUVMode`,
    /// 0 = DC / 1 = TM), used at m0 under `StrictLibwebpParity` where libwebp
    /// leaves the UV mode unrefined. Falls back to the RD pick if the hint
    /// vector is unavailable (e.g. `partition_limit >= 100` suppressed it).
    fn analysis_uv_mode(&self, mbx: usize, mby: usize) -> ChromaMode {
        let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
        match self.fast_mb_uv_hints.get(mb_idx) {
            Some(0) => ChromaMode::DC,
            Some(1) => ChromaMode::TM,
            _ => self.pick_best_uv(mbx, mby),
        }
    }

    fn pick_best_uv(&self, mbx: usize, mby: usize) -> ChromaMode {
        let mbw = usize::from(self.macroblock_width);
        let chroma_width = mbw * 8;

        // Order matches FIXED_COSTS_UV
        const MODES: [ChromaMode; 4] =
            [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

        let segment = self.get_segment_for_mb(mbx, mby);
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();
        let lambda = segment.lambda_uv;
        let tlambda = segment.tlambda;

        // Use updated probabilities if available (for consistent mode selection)
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Pre-extract source blocks for TDisto/psy-rd (avoid repeated extraction per mode).
        // libwebp's PickBestUV sets `rd_uv.SD = 0` unconditionally ("not calling TDisto
        // here: it tends to flatten areas", `quant_enc.c:1222`) — chroma never gets a
        // spectral-distortion term. Under StrictLibwebpParity we match that exactly by
        // suppressing the UV TDisto/psy-rd path; the tuned default keeps zenwebp's
        // perceptual UV extension unchanged. (#38)
        let need_spectral = (tlambda > 0 || segment.psy_config.psy_rd_strength > 0)
            && self.cost_model != crate::encoder::api::CostModel::StrictLibwebpParity;
        let (src_u_block, src_v_block) = if need_spectral {
            let mut u_block = [0u8; 64];
            let mut v_block = [0u8; 64];
            for y in 0..8 {
                let src_row = (mby * 8 + y) * chroma_width + mbx * 8;
                u_block[y * 8..(y + 1) * 8].copy_from_slice(&self.frame.ubuf[src_row..src_row + 8]);
                v_block[y * 8..(y + 1) * 8].copy_from_slice(&self.frame.vbuf[src_row..src_row + 8]);
            }
            (Some(u_block), Some(v_block))
        } else {
            (None, None)
        };

        let mut best_mode = ChromaMode::DC;
        let mut best_rd_score = i64::MAX;
        // Match libwebp's PickBestUV tie-break: it evaluates chroma modes in the
        // order DC, TM, V, H (internal mode numbers 0,1,2,3) and keeps the first
        // strictly-smaller score, so ties resolve to the lowest libwebp mode
        // number. Our `MODES` is [DC, V, H, TM] -> libwebp ranks [0, 2, 3, 1].
        // (#38 — same edge TM/H tie phenomenon as the I16 path.)
        const UV_LIB_RANK: [u8; 4] = [0, 2, 3, 1];
        let parity_tiebreak =
            self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity;
        let mut best_lib_rank = u8::MAX;

        #[cfg(feature = "mode_debug")]
        let debug_uv = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: alloc::vec::Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);

        // Pre-allocate scratch buffers outside mode loop to avoid redundant zero-init.
        // All elements are written before read in each iteration.
        let mut u_blocks = [0i32; 64];
        let mut v_blocks = [0i32; 64];
        let mut uv_quant = [[0i32; 16]; 8];
        // Zigzag-scan-order view of `uv_quant`, used only for the coefficient
        // rate (`get_cost_uv`) and the flatness test. libwebp stores `uv_levels`
        // in zigzag order (`quant.c` writes `out[]` zigzagged) and both
        // `VP8GetCostUV` and `IsFlat` walk that scan order; our quant buffer is
        // natural (raster) order for the IDCT path, so cost/flatness computed on
        // it diverged from libwebp (wrong per-position bands + `last`), flipping
        // UV mode picks at m3+ (#38). The I4 path already builds this zigzag view
        // (see `evaluate_i4_modes_*`); this brings I-UV to parity with it.
        let mut uv_quant_zz = [[0i32; 16]; 8];
        let mut uv_dequant = [[0i32; 16]; 8];
        let mut rec_u_block = [0u8; 64];
        let mut rec_v_block = [0u8; 64];
        let mut all_levels_uv = [0i16; 128];

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // libwebp's PickBestUV (`quant_enc.c:1214`) evaluates ALL four chroma
            // modes at every MB, including V/H/TM at frame edges — the prediction
            // there reads the default borders (top=127, left=129), which the decoder
            // uses too, so an edge V/H/TM pick is legal and round-trips. Skipping
            // them diverged from libwebp at edge MBs (the m3+ UV gap, #38); this is
            // the RD-path analog of the RefineUsingDistortion edge fix in
            // `pick_uv_sse`.

            // Generate predictions for U and V
            let pred_u = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_u,
                &self.left_border_u,
            );
            let pred_v = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_v,
                &self.left_border_v,
            );

            // === Full reconstruction for RD evaluation ===
            // 1. Compute residuals and forward DCT (buffers hoisted outside mode loop)
            self.fill_chroma_blocks_from_predicted(
                &pred_u,
                &self.frame.ubuf,
                mbx,
                mby,
                &mut u_blocks,
            );
            self.fill_chroma_blocks_from_predicted(
                &pred_v,
                &self.frame.vbuf,
                mbx,
                mby,
                &mut v_blocks,
            );

            // Under StrictLibwebpParity, apply the chroma DC error diffusion that
            // libwebp's ReconstructUV runs for every candidate mode
            // (`CorrectDCValues`, gated on `it->top_derr != NULL`, i.e. quality
            // <= 98). Scoring each mode on the *diffused* reconstruction is what
            // makes the m3+ UV mode pick match libwebp — without it, the RD sees
            // a different DC (and different levels) than the emitted bitstream,
            // flipping edge/near-boundary picks (#38). Read-only on the diffusion
            // state; the final emission still performs the actual store. The
            // tuned default keeps scoring on the undiffused reconstruction.
            if self.do_error_diffusion
                && self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity
            {
                super::residuals::diffuse_chroma_dc_inplace(
                    &mut u_blocks,
                    &mut v_blocks,
                    &self.top_derr[mbx],
                    &self.left_derr,
                    uv_matrix,
                );
            }

            // 2. Fused quantize+dequantize coefficients using SIMD
            // (uv_quant/uv_dequant hoisted outside mode loop to avoid redundant zero-init)

            // Process U blocks (indices 0-3)
            for block_idx in 0..4 {
                let block_start = block_idx * 16;
                let coeffs: [i32; 16] = u_blocks[block_start..block_start + 16].try_into().unwrap();
                crate::encoder::quantize::quantize_dequantize_block_simd(
                    &coeffs,
                    uv_matrix,
                    false,
                    &mut uv_quant[block_idx],
                    &mut uv_dequant[block_idx],
                );
            }

            // Process V blocks (indices 4-7)
            for block_idx in 0..4 {
                let block_start = block_idx * 16;
                let coeffs: [i32; 16] = v_blocks[block_start..block_start + 16].try_into().unwrap();
                crate::encoder::quantize::quantize_dequantize_block_simd(
                    &coeffs,
                    uv_matrix,
                    false,
                    &mut uv_quant[4 + block_idx],
                    &mut uv_dequant[4 + block_idx],
                );
            }

            // Build the zigzag-scan-order view for cost + flatness (see the
            // `uv_quant_zz` declaration above). Natural order stays in `uv_quant`
            // / `uv_dequant` for the IDCT reconstruction below.
            for b in 0..8 {
                for n in 0..16 {
                    uv_quant_zz[b][n] = uv_quant[b][ZIGZAG[n] as usize];
                }
            }

            // 3. Compute coefficient cost using probability-dependent tables.
            // libwebp's VP8GetCostUV walks U then V channels (`ch ∈ {0,2}`) over a 2x2 grid
            // updating `it->top_nz[4+ch+x]`/`it->left_nz[4+ch+y]`. Import the live
            // cross-MB context from zenwebp's complexity tracker (#23).
            let top_u_nz = [
                self.top_complexity[mbx].u[0] != 0,
                self.top_complexity[mbx].u[1] != 0,
            ];
            let left_u_nz = [
                self.left_complexity.u[0] != 0,
                self.left_complexity.u[1] != 0,
            ];
            let top_v_nz = [
                self.top_complexity[mbx].v[0] != 0,
                self.top_complexity[mbx].v[1] != 0,
            ];
            let left_v_nz = [
                self.left_complexity.v[0] != 0,
                self.left_complexity.v[1] != 0,
            ];
            let coeff_cost = get_cost_uv(
                &uv_quant_zz,
                top_u_nz,
                left_u_nz,
                top_v_nz,
                left_v_nz,
                &self.level_costs,
                probs,
            );

            // 4. Fused inverse DCT + add residue for reconstruction
            let mut reconstructed_u = pred_u;
            let mut reconstructed_v = pred_v;

            // Reconstruct U blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = uv_dequant[block_idx][1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut uv_dequant[block_idx],
                    &mut reconstructed_u,
                    y0,
                    x0,
                    CHROMA_STRIDE,
                    dc_only,
                );
            }

            // Reconstruct V blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = uv_dequant[4 + block_idx][1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut uv_dequant[4 + block_idx],
                    &mut reconstructed_v,
                    y0,
                    x0,
                    CHROMA_STRIDE,
                    dc_only,
                );
            }

            // 4. Compute SSE between source and reconstructed (NOT prediction!)
            let sse_u = sse_8x8_chroma(&self.frame.ubuf, chroma_width, mbx, mby, &reconstructed_u);
            let sse_v = sse_8x8_chroma(&self.frame.vbuf, chroma_width, mbx, mby, &reconstructed_v);
            let sse = sse_u + sse_v;

            // 4b. Compute UV spectral distortion + psy-rd if enabled
            let uv_spectral_disto = if let (Some(src_u), Some(src_v)) = (&src_u_block, &src_v_block)
            {
                // Extract reconstructed blocks only (source already cached)
                // (rec_u_block/rec_v_block hoisted outside mode loop)
                for y in 0..8 {
                    let rec_row = (y + 1) * CHROMA_STRIDE + 1;
                    rec_u_block[y * 8..(y + 1) * 8]
                        .copy_from_slice(&reconstructed_u[rec_row..rec_row + 8]);
                    rec_v_block[y * 8..(y + 1) * 8]
                        .copy_from_slice(&reconstructed_v[rec_row..rec_row + 8]);
                }

                let td = if tlambda > 0 {
                    let td_u = tdisto_8x8(src_u, &rec_u_block, 8, &segment.psy_config.chroma_csf);
                    let td_v = tdisto_8x8(src_v, &rec_v_block, 8, &segment.psy_config.chroma_csf);
                    let td_total = td_u + td_v;
                    (tlambda as i32 * td_total + 128) >> 8
                } else {
                    0
                };

                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_8x8(src_u, 8) + psy::satd_8x8(src_v, 8);
                    let rec_satd = psy::satd_8x8(&rec_u_block, 8) + psy::satd_8x8(&rec_v_block, 8);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };

                td + psy
            } else {
                0
            };

            // 5. Apply flatness penalty for non-DC modes
            let rate_penalty = if mode_idx > 0 {
                // Check if coefficients are flat
                // (all_levels_uv hoisted outside mode loop)
                for block_idx in 0..8 {
                    for i in 0..16 {
                        all_levels_uv[block_idx * 16 + i] = uv_quant_zz[block_idx][i] as i16;
                    }
                }
                if is_flat_coeffs(&all_levels_uv, 8, FLATNESS_LIMIT_UV) {
                    // Add flatness penalty: FLATNESS_PENALTY * num_blocks
                    FLATNESS_PENALTY * 8
                } else {
                    0
                }
            } else {
                0
            };

            // 6. Compute full RD score
            // score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
            let mode_cost = FIXED_COSTS_UV[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost) + i64::from(rate_penalty))
                * i64::from(lambda);
            let distortion =
                i64::from(RD_DISTO_MULT) * (i64::from(sse) + i64::from(uv_spectral_disto));
            let rd_score = rate + distortion;

            #[cfg(feature = "mode_debug")]
            if debug_uv {
                eprintln!(
                    "  ZENUV mb({mbx},{mby}) mode_idx={mode_idx} H={} R={} penalty={} sse={} SD={} lambda_uv={} score={rd_score}",
                    mode_cost, coeff_cost, rate_penalty, sse, uv_spectral_disto, lambda
                );
            }

            let is_better = rd_score < best_rd_score
                || (parity_tiebreak
                    && rd_score == best_rd_score
                    && UV_LIB_RANK[mode_idx] < best_lib_rank);
            if is_better {
                best_rd_score = rd_score;
                best_mode = mode;
                best_lib_rank = UV_LIB_RANK[mode_idx];
            }
        }

        #[cfg(feature = "mode_debug")]
        if debug_uv {
            eprintln!("  ZENUV mb({mbx},{mby}) WIN mode={best_mode:?} score={best_rd_score}");
        }

        best_mode
    }

    pub(super) fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        // FastMBAnalyze fast path (libwebp `RefineUsingDistortion(try_both_modes=0,
        // refine_uv_mode=method>=1)` at m0/m1, quant_enc.c:1447). The analysis
        // pass already chose I16 vs I4 via the DC-variance test in
        // `FastMBAnalyze`. Consume that hint directly:
        //   - I16Dc  => LumaMode::DC (matches `pick_intra16_fast_dc`)
        //   - I4AllDc => run `pick_best_intra4` to refine the per-sub-block
        //     modes. libwebp uses an SSE-only loop here; we reuse our
        //     existing RD path which is more accurate but slower. This
        //     resolves #32 (4× size blowup on tiny low-color images at m0):
        //     the previous code unconditionally picked I16-DC at m0/m1
        //     regardless of source variance, missing the I4 case entirely.
        // Chroma still goes through `pick_best_uv` here (zenwebp's path
        // matches libwebp m1's `refine_uv_mode=1`; m0 differs but UV mode
        // experiments showed zero size change on the worst offender).
        if self.method <= 1 && self.partition_limit < 100 && !self.fast_mb_hints.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            if let Some(&hint) = self.fast_mb_hints.get(mb_idx) {
                use crate::encoder::analysis::MbModeHint;
                // For the I16 hint, use the fast DC scorer (current behavior);
                // for I4, run `pick_best_intra4` so each sub-block's mode is
                // chosen rather than left at all-DC (libwebp does the same,
                // just via SSE rather than RD).
                // Track winning-mode D for the #44 `D > min_disto` gate.
                // For the I4AllDc branch we run the fast-DC scorer to obtain
                // an I16 baseline; we keep that D in case I4 doesn't beat it.
                let mut intra16_d: Option<u32> = None;
                let (luma_mode, luma_bpred) = match hint {
                    MbModeHint::I16Dc => {
                        // libwebp RefineUsingDistortion: all four I16 modes
                        // by SSE (not DC-only), lambda_d_i16 = 106.
                        let (mode, d, _score, _flat) = self.pick_intra16_sse(mbx, mby);
                        intra16_d = Some(d);
                        (mode, None)
                    }
                    MbModeHint::I4AllDc => {
                        // libwebp RefineUsingDistortion with try_both == 0:
                        // the hint decides I4; sub-block modes are SSE-picked
                        // (10 candidates each), no bail-out to I16.
                        let modes = self.pick_intra4_sse(mbx, mby);
                        (LumaMode::B, Some(modes))
                    }
                };
                // libwebp refine_uv_mode = (method >= 1): m1 refines by SSE;
                // m0 leaves the chroma mode at the analysis pick
                // (`refine_uv_mode = 0`). Under StrictLibwebpParity, m0 uses
                // that stored analysis UV mode verbatim to match libwebp;
                // otherwise m0 keeps zenwebp's RD pick (measured better — the
                // SSE pick at m0 pushed gradient q90 past the 1.3x size gate).
                let chroma_mode = if self.method == 0 {
                    if self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity {
                        self.analysis_uv_mode(mbx, mby)
                    } else {
                        self.pick_best_uv(mbx, mby)
                    }
                } else {
                    self.pick_uv_sse(mbx, mby)
                };
                let segment_id = self.get_segment_id_for_mb(mbx, mby);
                return MacroblockInfo {
                    luma_mode,
                    luma_bpred,
                    chroma_mode,
                    segment_id,
                    coeffs_skipped: false,
                    intra16_d,
                };
            }
        }

        // Method 2 (RD_OPT_NONE without analysis hints): libwebp's
        // RefineUsingDistortion(try_both_modes=1, refine_uv_mode=1) —
        // SSE-scored I16 pick, SSE-scored I4 with accumulate-and-bail
        // against the I16 score (score_i4 starts at i4_penalty = 1000*q_i4^2,
        // header bits capped by mb_header_limit), SSE-scored UV refine.
        // m3+ keeps the full RD path below (libwebp RD_OPT_BASIC+).
        if self.method == 2 && self.partition_limit < 100 {
            let (i16_mode, i16_d, i16_score, flat_locked) = self.pick_intra16_sse(mbx, mby);
            let (luma_mode, luma_bpred) = if flat_locked {
                (i16_mode, None)
            } else {
                let segment = self.get_segment_for_mb(mbx, mby);
                let q_i4 = (segment.ydc as u64 + 15 * segment.yac as u64 + 8) >> 4;
                let i4_penalty = 1000u64 * q_i4 * q_i4;
                let num_mb = u64::from(self.macroblock_width) * u64::from(self.macroblock_height);
                let bit_limit = (256u64 * 510 * 8 * 1024) / num_mb.max(1);
                match self.pick_intra4_sse_with_bail(mbx, mby, i16_score, i4_penalty, bit_limit) {
                    Some(modes) => (LumaMode::B, Some(modes)),
                    None => (i16_mode, None),
                }
            };
            let chroma_mode = self.pick_uv_sse(mbx, mby);
            let segment_id = self.get_segment_id_for_mb(mbx, mby);
            return MacroblockInfo {
                luma_mode,
                luma_bpred,
                chroma_mode,
                segment_id,
                coeffs_skipped: false,
                intra16_d: Some(i16_d),
            };
        }

        // Pick the best 16x16 luma mode using RD cost selection
        let (luma_mode, i16_score, i16_d) = self.pick_best_intra16(mbx, mby);

        // Debug output for specific macroblock (check MB_DEBUG env var)
        // Set MB_DEBUG=x,y to debug mode selection for that macroblock
        #[cfg(feature = "mode_debug")]
        let debug_mb = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: alloc::vec::Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let debug_mb = false;

        #[allow(unused_variables, clippy::needless_bool)]
        let debug_i16_details = if debug_mb {
            #[cfg(feature = "mode_debug")]
            {
                let segment = self.get_segment_for_mb(mbx, mby);
                eprintln!("=== MB({},{}) Mode Selection Debug ===", mbx, mby);
                eprintln!("I16: mode={:?}, score={}", luma_mode, i16_score);
                eprintln!(
                    "  lambda_mode={}, lambda_i4={}, lambda_i16={}",
                    segment.lambda_mode, segment.lambda_i4, segment.lambda_i16
                );
            }
            true
        } else {
            false
        };

        // Method-based I4 mode selection:
        // - method 0-1: Skip I4 entirely (fastest)
        // - method 2-4: Try I4 with fast filtering
        // - method 5-6: Full I4 search
        // partition_limit >= 100 forces I16-only to prevent partition 0 overflow.
        let (luma_mode, luma_bpred) = if self.method <= 1 || self.partition_limit >= 100 {
            // Fastest / partition overflow prevention: I16 only, no I4 evaluation
            (luma_mode, None)
        } else {
            // For method >= 2, try I4 with early exit optimizations
            let segment = self.get_segment_for_mb(mbx, mby);
            // partition_limit raises the skip threshold, making it harder for I4 to
            // qualify. At partition_limit=0 this is the base threshold (211).
            // At partition_limit=80, threshold is ~5x higher, skipping most I4 attempts.
            let limit_boost = 211u64 + 211u64 * u64::from(self.partition_limit) * 5 / 100;
            let skip_i4_threshold = limit_boost * u64::from(segment.lambda_mode);

            // libwebp's VP8Decimate calls PickBestIntra4 unconditionally at m3+
            // (`quant_enc.c:1428`, gated only on `method >= 2` and
            // `max_i4_header_bits > 0`) — there is no "skip I4 for flat DC blocks"
            // heuristic. Under StrictLibwebpParity we always try I4 so the I16-vs-I4
            // decision matches libwebp exactly (#38). The tuned default keeps the
            // flat-DC skip as a speed optimization.
            //
            // Skip I4 for very flat DC blocks (method 2-4)
            // For method 5-6, always try I4 for best quality (unless partition_limit overrides)
            let should_try_i4 = self.cost_model
                == crate::encoder::api::CostModel::StrictLibwebpParity
                || (self.method >= 5 && self.partition_limit < 50)
                || i16_score > skip_i4_threshold
                || luma_mode != LumaMode::DC;

            if should_try_i4 {
                match self.pick_best_intra4(mbx, mby, i16_score) {
                    #[allow(unused_variables)]
                    Some((modes, i4_score)) => {
                        #[cfg(feature = "mode_debug")]
                        if debug_mb {
                            eprintln!("I4: score={} (beats I16)", i4_score);
                            eprintln!("  modes={:?}", modes);
                            eprintln!(
                                "  RESULT: I4 wins by {} points",
                                i16_score.saturating_sub(i4_score)
                            );
                        }
                        (LumaMode::B, Some(modes))
                    }
                    None => {
                        #[cfg(feature = "mode_debug")]
                        if debug_mb {
                            eprintln!("I4: score >= {} (I16 wins)", i16_score);
                            eprintln!("  RESULT: I16 wins");
                        }
                        (luma_mode, None)
                    }
                }
            } else {
                #[cfg(feature = "mode_debug")]
                if debug_mb {
                    eprintln!("I4: skipped (flat DC block)");
                    eprintln!("  RESULT: I16 wins (I4 not tried)");
                }
                (luma_mode, None)
            }
        };

        // Pick the best chroma mode using RD-based selection
        let chroma_mode = self.pick_best_uv(mbx, mby);

        // Get segment ID from segment map if enabled
        let segment_id = self.get_segment_id_for_mb(mbx, mby);

        // Only carry I16's D forward when I16 actually wins (#44 gate).
        // I4 winners use a different reconstruction path; libwebp's
        // `StoreMaxDelta` is gated on `(nz & 0x100ffff) == 0x1000000` —
        // i.e. an I16 MB with non-zero Y2 and zero Y1 AC — so I4 MBs
        // never reach that call site anyway.
        let intra16_d = if luma_bpred.is_some() {
            None
        } else {
            Some(i16_d)
        };

        MacroblockInfo {
            luma_mode,
            luma_bpred,
            chroma_mode,
            segment_id,
            coeffs_skipped: false,
            intra16_d,
        }
    }

    /// Scalar fallback for I4 mode evaluation.
    /// Used when SIMD is not available or on non-x86_64 platforms.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_i4_modes_scalar(
        &self,
        src_block: &[u8; 16],
        preds: &I4Predictions,
        max_modes_to_try: usize,
        mode_costs: &[u16; 10],
        nz_top: bool,
        nz_left: bool,
        y1_matrix: &crate::encoder::quantize::VP8Matrix,
        lambda_i4: u32,
        tlambda: u32,
        trellis_lambda_i4: Option<u32>,
        probs: &TokenProbTables,
        segment: &Segment,
        best_mode: &mut IntraMode,
        best_mode_idx: &mut usize,
        best_block_score: &mut u64,
        best_has_nz: &mut bool,
        best_dequantized: &mut [i32; 16],
        best_sse: &mut u32,
        best_spectral_disto: &mut i32,
        best_psy_cost: &mut i32,
        best_coeff_cost: &mut u32,
    ) {
        const MODES: [IntraMode; 10] = [
            IntraMode::DC,
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::LD,
            IntraMode::RD,
            IntraMode::VR,
            IntraMode::VL,
            IntraMode::HD,
            IntraMode::HU,
        ];

        // Pre-sort modes by SSE
        let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
        for (mode_idx, _) in MODES.iter().enumerate() {
            let pred = preds.get(mode_idx);
            let sse = sse4x4_dispatch(src_block, pred);
            mode_sse[mode_idx] = (sse, mode_idx);
        }
        mode_sse.sort_unstable_by_key(|&(sse, _)| sse);

        for &(_, mode_idx) in mode_sse[..max_modes_to_try].iter() {
            let pred = preds.get(mode_idx);

            let mut residual = crate::common::transform::ftransform_from_u8_4x4(src_block, pred);

            let mut quantized_zigzag = [0i32; 16];
            let mut quantized_natural = [0i32; 16];
            let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
                let ctx0 = usize::from(nz_top) + usize::from(nz_left);
                const CTYPE_I4_AC: usize = 3;
                let nz = trellis_quantize_block(
                    &mut residual,
                    &mut quantized_zigzag,
                    y1_matrix,
                    lambda,
                    0,
                    &self.level_costs,
                    CTYPE_I4_AC,
                    ctx0,
                    &segment.psy_config,
                );
                for n in 0..16 {
                    let j = ZIGZAG[n] as usize;
                    quantized_natural[j] = quantized_zigzag[n];
                }
                let mut dq = quantized_natural;
                for (idx, val) in dq.iter_mut().enumerate() {
                    *val = y1_matrix.dequantize(*val, idx);
                }
                transform::idct4x4(&mut dq);
                (nz, dq)
            } else {
                let mut dequant_natural = [0i32; 16];
                let nz = crate::encoder::quantize::quantize_dequantize_block_simd(
                    &residual,
                    y1_matrix,
                    true,
                    &mut quantized_natural,
                    &mut dequant_natural,
                );
                for n in 0..16 {
                    let j = ZIGZAG[n] as usize;
                    quantized_zigzag[n] = quantized_natural[j];
                }
                transform::idct4x4(&mut dequant_natural);
                (nz, dequant_natural)
            };

            // Compute SSE (cheap, needed for early exit)
            let sse = sse4x4_with_residual_dispatch(src_block, pred, &dequantized);

            // Flatness penalty for non-DC modes (cheap check)
            let flatness_penalty: u32 = if mode_idx > 0 {
                let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
                if is_flat_coeffs(&levels_i16, 1, FLATNESS_LIMIT_I4) {
                    FLATNESS_PENALTY
                } else {
                    0
                }
            } else {
                0
            };

            // Early exit #1 (libwebp-style): check score WITHOUT coefficient cost.
            // Since coeff_cost >= 0 and spectral_disto >= 0, the actual score can
            // only be >= this lower bound. Skip expensive residual_cost if hopeless.
            let mode_cost = mode_costs[mode_idx];
            let lower_bound =
                crate::encoder::cost::rd_score_full(sse, 0, mode_cost, flatness_penalty, lambda_i4)
                    as u64;
            if lower_bound >= *best_block_score {
                continue;
            }

            // Compute coefficient cost (expensive)
            let (coeff_cost_val, _) =
                get_cost_luma4(&quantized_zigzag, nz_top, nz_left, &self.level_costs, probs);

            // Early exit #2: check with coefficient cost but without spectral/psy.
            let total_rate_cost = coeff_cost_val + flatness_penalty;
            let base_rd_score =
                crate::encoder::cost::rd_score_full(sse, 0, mode_cost, total_rate_cost, lambda_i4)
                    as u64;
            if base_rd_score >= *best_block_score {
                continue;
            }

            let (spectral_disto, psy_cost_val) = if tlambda > 0
                || segment.psy_config.psy_rd_strength > 0
            {
                let mut rec_block = [0u8; 16];
                for k in 0..16 {
                    rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
                }
                let td = if tlambda > 0 {
                    let td_raw = tdisto_4x4(src_block, &rec_block, 4, &segment.psy_config.luma_csf);
                    (tlambda as i32 * td_raw + 128) >> 8
                } else {
                    0
                };
                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_4x4(src_block, 4);
                    let rec_satd = psy::satd_4x4(&rec_block, 4);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };
                (td, psy)
            } else {
                (0, 0)
            };

            let rd_score = crate::encoder::cost::rd_score_full(
                sse,
                spectral_disto + psy_cost_val,
                mode_cost,
                total_rate_cost,
                lambda_i4,
            ) as u64;

            if rd_score < *best_block_score {
                *best_block_score = rd_score;
                *best_mode = MODES[mode_idx];
                *best_mode_idx = mode_idx;
                *best_has_nz = has_nz;
                *best_dequantized = dequantized;
                *best_sse = sse;
                *best_spectral_disto = spectral_disto;
                *best_psy_cost = psy_cost_val;
                *best_coeff_cost = coeff_cost_val;
            }
        }
    }

    /// Estimate coefficient cost for a specific Intra16 mode
    #[allow(dead_code)] // Reserved for Intra4 vs Intra16 RD comparison
    fn estimate_luma16_mode_coeff_cost(
        &self,
        mode: LumaMode,
        mbx: usize,
        mby: usize,
        segment: &Segment,
    ) -> u32 {
        // Get prediction and compute residuals
        let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&pred, mbx, mby);

        // Use the cost estimation function
        self.estimate_luma16_coeff_cost(&luma_blocks, segment)
    }
}

/// Shared body of the SSE-scored Intra4 pick. `$sse4x4`, `$ftransform`,
/// `$quant`, `$idct_add` are the per-target primitives; on x86_64 the arcane
/// wrapper below instantiates it with the `#[rite]` SSE2 kernels inside a
/// single target_feature region, elsewhere the portable wrapper uses the
/// dispatching helpers.
macro_rules! pick_intra4_sse_body {
    ($ybuf:expr, $src_width:expr, $mbx:expr, $mby:expr, $ywb:expr, $y1:expr,
     $top_ctx0:expr, $left_ctx0:expr, $i16_score:expr, $i4_penalty:expr, $bit_limit:expr,
     $sse4x4:expr, $ftransform:expr, $quant:expr, $idct_add:expr) => {{
        const LAMBDA_D_I4: u64 = 11;
        let mut best_modes = [IntraMode::DC; 16];
        let mut best_mode_indices = [0usize; 16];
        let mut score_i4: u64 = $i4_penalty;
        let mut i4_bit_sum: u64 = 0;

        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                let top_ctx = if sby == 0 {
                    $top_ctx0[sbx]
                } else {
                    best_mode_indices[(sby - 1) * 4 + sbx]
                };
                let left_ctx = if sbx == 0 {
                    $left_ctx0[sby]
                } else {
                    best_mode_indices[sby * 4 + (sbx - 1)]
                };
                let mode_costs: [u16; 10] = core::array::from_fn(|mode_idx| {
                    crate::encoder::tables::VP8_FIXED_COSTS_I4[top_ctx][left_ctx][mode_idx]
                });

                let preds = I4Predictions::compute($ywb, x0, y0, LUMA_STRIDE);

                let src_base = ($mby * 16 + sby * 4) * $src_width + $mbx * 16 + sbx * 4;
                let mut src_block = [0u8; 16];
                for y in 0..4 {
                    let src_row = src_base + y * $src_width;
                    src_block[y * 4..y * 4 + 4].copy_from_slice(&$ybuf[src_row..src_row + 4]);
                }

                // Evaluate modes in libwebp's internal B-mode iteration order
                // so an exact SSE tie resolves to the same winner. libwebp
                // loops `mode = 0..NUM_BMODES` in ITS numbering (DC, TM, VE, HE,
                // RD, VR, LD, VL, HD, HU) with a strict `<`, so ties go to the
                // lowest libwebp index. `preds.data`/`mode_costs` are in
                // zenwebp's IntraMode order, so visit their indices in the
                // permutation that reproduces libwebp's order: RD(zen 5),
                // VR(zen 6), LD(zen 4) occupy libwebp slots 4, 5, 6.
                const ITER_ORDER: [usize; 10] = [0, 1, 2, 3, 5, 6, 4, 7, 8, 9];
                let mut best_idx = 0usize;
                let mut best_score = u64::MAX;
                for &m in ITER_ORDER.iter() {
                    let pred = &preds.data[m];
                    let sse = $sse4x4(&src_block, pred);
                    let score = u64::from(sse) * u64::from(RD_DISTO_MULT)
                        + u64::from(mode_costs[m]) * LAMBDA_D_I4;
                    if score < best_score {
                        best_score = score;
                        best_idx = m;
                    }
                }

                // `best_idx` indexes `preds.data` / `mode_costs`, which are in
                // IntraMode declaration order (DC, TM, VE, HE, LD, RD, VR, VL,
                // HD, HU). Derive the IntraMode from that same index via
                // `from_i8` — the single canonical mapping — so the emitted and
                // reconstructed mode always matches the prediction that won.
                let best_mode = IntraMode::from_i8(best_idx as i8)
                    .expect("best_idx is in 0..10, a valid I4 mode index");
                best_modes[i] = best_mode;
                best_mode_indices[i] = best_idx;
                score_i4 = score_i4.saturating_add(best_score);
                i4_bit_sum += u64::from(mode_costs[best_idx]);
                if score_i4 >= $i16_score || i4_bit_sum > $bit_limit {
                    // Intra4 won't beat Intra16 (libwebp quant_enc.c:1310).
                    return None;
                }

                // Reconstruct the winner so later sub-blocks predict from
                // reconstructed (not source) neighbors, matching the decoder.
                super::Vp8Encoder::apply_intra4_prediction($ywb, best_mode, x0, y0);
                let coeffs = $ftransform(&src_block, &preds.data[best_idx]);
                let mut quantized = [0i32; 16];
                let mut dequant = [0i32; 16];
                let has_nz = $quant(&coeffs, $y1, &mut quantized, &mut dequant);
                if has_nz {
                    let dc_only = dequant[1..].iter().all(|&c| c == 0);
                    $idct_add(&mut dequant, $ywb, y0, x0, dc_only);
                }
            }
        }

        Some(best_modes)
    }};
}

/// Single-`#[arcane]` instantiation of the SSE-scored Intra4 pick: every
/// primitive is the `#[rite]` SSE2 kernel, inlined into this one
/// target_feature region (739bc14 pattern).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn pick_intra4_sse_arcane(
    token: X64V3Token,
    ybuf: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    y_with_border: &mut [u8; LUMA_BLOCK_SIZE],
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    top_ctx0: &[usize; 4],
    left_ctx0: &[usize; 4],
    i16_score: u64,
    i4_penalty: u64,
    bit_limit: u64,
) -> Option<[IntraMode; 16]> {
    pick_intra4_sse_body!(
        ybuf,
        src_width,
        mbx,
        mby,
        y_with_border,
        y1_matrix,
        top_ctx0,
        left_ctx0,
        i16_score,
        i4_penalty,
        bit_limit,
        |src: &[u8; 16], pred: &[u8; 16]| crate::common::simd_sse::sse4x4_sse2(token, src, pred),
        |src: &[u8; 16], pred: &[u8; 16]| {
            crate::common::transform::ftransform_from_u8_4x4_sse2(token, src, pred)
        },
        |coeffs: &[i32; 16],
         m: &crate::encoder::quantize::VP8Matrix,
         q: &mut [i32; 16],
         dq: &mut [i32; 16]| {
            crate::encoder::quantize::quantize_dequantize_block_sse2(token, coeffs, m, true, q, dq)
        },
        |dq: &mut [i32; 16],
         ywb: &mut [u8; LUMA_BLOCK_SIZE],
         y0: usize,
         x0: usize,
         dc_only: bool| {
            crate::common::transform::idct_add_residue_inplace_sse2_inner(
                token,
                dq,
                ywb.as_mut_slice(),
                y0,
                x0,
                LUMA_STRIDE,
                dc_only,
            )
        }
    )
}

/// Portable (dispatching) instantiation for non-x86_64 targets.
#[allow(clippy::too_many_arguments)]
fn pick_intra4_sse_portable(
    ybuf: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    y_with_border: &mut [u8; LUMA_BLOCK_SIZE],
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    top_ctx0: &[usize; 4],
    left_ctx0: &[usize; 4],
    i16_score: u64,
    i4_penalty: u64,
    bit_limit: u64,
) -> Option<[IntraMode; 16]> {
    pick_intra4_sse_body!(
        ybuf,
        src_width,
        mbx,
        mby,
        y_with_border,
        y1_matrix,
        top_ctx0,
        left_ctx0,
        i16_score,
        i4_penalty,
        bit_limit,
        |src: &[u8; 16], pred: &[u8; 16]| sse4x4_dispatch(src, pred),
        |src: &[u8; 16], pred: &[u8; 16]| {
            crate::common::transform::ftransform_from_u8_4x4(src, pred)
        },
        |coeffs: &[i32; 16],
         m: &crate::encoder::quantize::VP8Matrix,
         q: &mut [i32; 16],
         dq: &mut [i32; 16]| {
            crate::encoder::quantize::quantize_dequantize_block_simd(coeffs, m, true, q, dq)
        },
        |dq: &mut [i32; 16],
         ywb: &mut [u8; LUMA_BLOCK_SIZE],
         y0: usize,
         x0: usize,
         dc_only: bool| {
            crate::common::transform::idct_add_residue_inplace(
                dq,
                ywb,
                y0,
                x0,
                LUMA_STRIDE,
                dc_only,
            )
        }
    )
}
