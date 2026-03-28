//! Fused prediction + IDCT pipeline for the v2 VP8 decoder.
//!
//! Combines border update, prediction, IDCT, border extraction, and cache
//! write into single `#[arcane]` entry points per architecture. This puts
//! all prediction loops + IDCT in one `target_feature` region, enabling
//! autovectorization and eliminating per-block dispatch overhead.
//!
//! **Non-B luma modes (I16/V/H/TM/DC):**
//! Predict 16x16 in workspace, IDCT add using non_zero_blocks bitmap,
//! then extract borders and copy to cache.
//!
//! **B luma mode (per 4x4 sub-block):**
//! Predict + IDCT interleaved in workspace (sub-blocks reference each
//! other), then extract borders and copy to cache.
//!
//! **Chroma (DC/V/H/TM):**
//! Predict 8x8 in workspace, IDCT add using non_zero_blocks bitmap,
//! then extract borders and copy to cache.

#![allow(clippy::too_many_arguments)]

use super::MbRowEntry;
use crate::common::prediction::*;
use crate::common::types::*;

// =============================================================================
// Public dispatch functions (called by the v2 decode loop)
// =============================================================================

/// Process one macroblock's luma prediction + IDCT.
///
/// Updates borders from `top_border_y`/`left_border_y`, runs prediction + IDCT
/// in workspace, extracts new borders, and copies the result to cache.
#[inline(always)]
pub(super) fn process_luma_mb(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_entry: &MbRowEntry,
    cache_y: &mut [u8],
    cache_y_stride: usize,
    extra_y_rows: usize,
    mbx: usize,
    mby: usize,
    mbwidth: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
) {
    // Update workspace borders from neighbor data
    update_border_luma(ws, mbx, mby, mbwidth, top_border_y, left_border_y);

    let nz = mb_entry.non_zero_blocks;

    // Dispatch to SIMD pipeline when available
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        use archmage::{SimdToken, X64V3Token};
        if let Some(token) = X64V3Token::summon() {
            process_luma_mb_v3(
                token,
                ws,
                coeff_blocks,
                mb_entry.luma_mode,
                &mb_entry.bpred,
                nz,
                mbx,
                mby,
            );
            extract_luma_borders_and_copy_to_cache(
                ws,
                cache_y,
                cache_y_stride,
                extra_y_rows,
                mbx,
                top_border_y,
                left_border_y,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            process_luma_mb_neon(
                token,
                ws,
                coeff_blocks,
                mb_entry.luma_mode,
                &mb_entry.bpred,
                nz,
                mbx,
                mby,
            );
            extract_luma_borders_and_copy_to_cache(
                ws,
                cache_y,
                cache_y_stride,
                extra_y_rows,
                mbx,
                top_border_y,
                left_border_y,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            process_luma_mb_wasm(
                token,
                ws,
                coeff_blocks,
                mb_entry.luma_mode,
                &mb_entry.bpred,
                nz,
                mbx,
                mby,
            );
            extract_luma_borders_and_copy_to_cache(
                ws,
                cache_y,
                cache_y_stride,
                extra_y_rows,
                mbx,
                top_border_y,
                left_border_y,
            );
            return;
        }
    }

    // Scalar fallback
    process_luma_mb_scalar(ws, coeff_blocks, mb_entry, nz, mbx, mby);
    extract_luma_borders_and_copy_to_cache(
        ws,
        cache_y,
        cache_y_stride,
        extra_y_rows,
        mbx,
        top_border_y,
        left_border_y,
    );
}

/// Process one macroblock's chroma prediction + IDCT (both U and V).
///
/// Updates borders, runs prediction + IDCT in workspaces, extracts new
/// borders, and copies the results to cache.
#[inline(always)]
pub(super) fn process_chroma_mb(
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_entry: &MbRowEntry,
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_uv_stride: usize,
    extra_y_rows: usize,
    mbx: usize,
    mby: usize,
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
) {
    // Update workspace borders from neighbor data
    update_border_chroma(uws, mbx, mby, top_border_u, left_border_u);
    update_border_chroma(vws, mbx, mby, top_border_v, left_border_v);

    let nz = mb_entry.non_zero_blocks;

    // Dispatch to SIMD pipeline when available
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        use archmage::{SimdToken, X64V3Token};
        if let Some(token) = X64V3Token::summon() {
            process_chroma_predict_and_idct_v3(
                token,
                uws,
                vws,
                coeff_blocks,
                mb_entry.chroma_mode,
                nz,
                mbx,
                mby,
            );
            extract_chroma_borders_and_copy_to_cache(
                uws,
                cache_u,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_u,
                left_border_u,
            );
            extract_chroma_borders_and_copy_to_cache(
                vws,
                cache_v,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_v,
                left_border_v,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            process_chroma_predict_and_idct_neon(
                token,
                uws,
                vws,
                coeff_blocks,
                mb_entry.chroma_mode,
                nz,
                mbx,
                mby,
            );
            extract_chroma_borders_and_copy_to_cache(
                uws,
                cache_u,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_u,
                left_border_u,
            );
            extract_chroma_borders_and_copy_to_cache(
                vws,
                cache_v,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_v,
                left_border_v,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            process_chroma_predict_and_idct_wasm(
                token,
                uws,
                vws,
                coeff_blocks,
                mb_entry.chroma_mode,
                nz,
                mbx,
                mby,
            );
            extract_chroma_borders_and_copy_to_cache(
                uws,
                cache_u,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_u,
                left_border_u,
            );
            extract_chroma_borders_and_copy_to_cache(
                vws,
                cache_v,
                cache_uv_stride,
                extra_y_rows,
                mbx,
                top_border_v,
                left_border_v,
            );
            return;
        }
    }

    // Scalar fallback
    process_chroma_mb_scalar(uws, vws, coeff_blocks, mb_entry, nz, mbx, mby);
    extract_chroma_borders_and_copy_to_cache(
        uws,
        cache_u,
        cache_uv_stride,
        extra_y_rows,
        mbx,
        top_border_u,
        left_border_u,
    );
    extract_chroma_borders_and_copy_to_cache(
        vws,
        cache_v,
        cache_uv_stride,
        extra_y_rows,
        mbx,
        top_border_v,
        left_border_v,
    );
}

// =============================================================================
// x86_64 / x86: V3 (AVX2+FMA) pipeline
// =============================================================================

/// Luma prediction + IDCT within a single AVX2+FMA target_feature region.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::arcane]
fn process_luma_mb_v3(
    _token: archmage::X64V3Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    luma_predict(ws, luma_mode, bpred, mbx, mby);
    luma_idct_x86(_token, ws, coeff_blocks, luma_mode, bpred, nz);
}

/// Chroma prediction + IDCT within a single AVX2+FMA target_feature region.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::arcane]
fn process_chroma_predict_and_idct_v3(
    _token: archmage::X64V3Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    chroma_predict(uws, vws, chroma_mode, mbx, mby);
    chroma_idct_x86(_token, uws, vws, coeff_blocks, nz);
}

/// Luma IDCT for x86 (rite — inlines into arcane callers).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::rite]
fn luma_idct_x86(
    _token: archmage::X64V3Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
) {
    let stride = LUMA_STRIDE;

    if luma_mode == LumaMode::B {
        // B-mode: interleaved predict + IDCT per 4x4 sub-block.
        // Prediction was skipped in luma_predict for B mode because later
        // sub-blocks reference earlier ones.
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sbx + sby * 4;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                bpred_dispatch(ws, bpred[i], x0, y0, stride);

                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_sse2_inner(
                        _token, rb, ws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    } else {
        // I16/V/H/TM/DC: prediction already done, add IDCT residue
        if nz & 0xFFFF != 0 {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    if nz & (1u32 << i) != 0 {
                        let y0 = 1 + y * 4;
                        let x0 = 1 + x * 4;
                        let rb = coeff_block(coeff_blocks, i);
                        let dc_only = rb[1..].iter().all(|&c| c == 0);
                        crate::common::transform::idct_add_residue_inplace_sse2_inner(
                            _token, rb, ws, y0, x0, stride, dc_only,
                        );
                    }
                }
            }
        }
    }
}

/// Chroma IDCT for x86 (rite).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::rite]
fn chroma_idct_x86(
    _token: archmage::X64V3Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    nz: u32,
) {
    let stride = CHROMA_STRIDE;

    if nz & 0xFF_0000 != 0 {
        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let u_idx = 16 + i;
                let v_idx = 20 + i;
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                if nz & (1u32 << u_idx) != 0 {
                    let urb = coeff_block(coeff_blocks, u_idx);
                    let dc_only = urb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_sse2_inner(
                        _token, urb, uws, y0, x0, stride, dc_only,
                    );
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb = coeff_block(coeff_blocks, v_idx);
                    let dc_only = vrb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_sse2_inner(
                        _token, vrb, vws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    }
}

// =============================================================================
// aarch64: NEON pipeline
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn process_luma_mb_neon(
    _token: archmage::NeonToken,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    luma_predict(ws, luma_mode, bpred, mbx, mby);
    luma_idct_neon(_token, ws, coeff_blocks, luma_mode, bpred, nz);
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn process_chroma_predict_and_idct_neon(
    _token: archmage::NeonToken,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    chroma_predict(uws, vws, chroma_mode, mbx, mby);
    chroma_idct_neon(_token, uws, vws, coeff_blocks, nz);
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn luma_idct_neon(
    _token: archmage::NeonToken,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
) {
    let stride = LUMA_STRIDE;

    if luma_mode == LumaMode::B {
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sbx + sby * 4;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                bpred_dispatch(ws, bpred[i], x0, y0, stride);

                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_neon_inner(
                        _token, rb, ws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    } else {
        if nz & 0xFFFF != 0 {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    if nz & (1u32 << i) != 0 {
                        let y0 = 1 + y * 4;
                        let x0 = 1 + x * 4;
                        let rb = coeff_block(coeff_blocks, i);
                        let dc_only = rb[1..].iter().all(|&c| c == 0);
                        crate::common::transform::idct_add_residue_inplace_neon_inner(
                            _token, rb, ws, y0, x0, stride, dc_only,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn chroma_idct_neon(
    _token: archmage::NeonToken,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    nz: u32,
) {
    let stride = CHROMA_STRIDE;

    if nz & 0xFF_0000 != 0 {
        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let u_idx = 16 + i;
                let v_idx = 20 + i;
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                if nz & (1u32 << u_idx) != 0 {
                    let urb = coeff_block(coeff_blocks, u_idx);
                    let dc_only = urb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_neon_inner(
                        _token, urb, uws, y0, x0, stride, dc_only,
                    );
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb = coeff_block(coeff_blocks, v_idx);
                    let dc_only = vrb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_neon_inner(
                        _token, vrb, vws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    }
}

// =============================================================================
// wasm32: WASM SIMD128 pipeline
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn process_luma_mb_wasm(
    _token: archmage::Wasm128Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    luma_predict(ws, luma_mode, bpred, mbx, mby);
    luma_idct_scalar(ws, coeff_blocks, luma_mode, bpred, nz);
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn process_chroma_predict_and_idct_wasm(
    _token: archmage::Wasm128Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    chroma_predict(uws, vws, chroma_mode, mbx, mby);
    chroma_idct_scalar(uws, vws, coeff_blocks, nz);
}

// =============================================================================
// Shared prediction functions (compiled with caller's target_feature)
// =============================================================================

/// Run prediction for non-B luma modes.
/// B-mode prediction is interleaved with IDCT in the arch-specific IDCT functions.
#[allow(dead_code)]
#[inline(always)]
fn luma_predict(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    luma_mode: LumaMode,
    _bpred: &[IntraMode; 16],
    mbx: usize,
    mby: usize,
) {
    let stride = LUMA_STRIDE;

    match luma_mode {
        LumaMode::V => predict_vpred(ws, 16, 1, 1, stride),
        LumaMode::H => predict_hpred(ws, 16, 1, 1, stride),
        LumaMode::TM => predict_tmpred(ws, 16, 1, 1, stride),
        LumaMode::DC => predict_dcpred(ws, 16, stride, mby != 0, mbx != 0),
        LumaMode::B => {
            // B-mode prediction is interleaved with IDCT
        }
    }
}

/// Run prediction for chroma (DC, V, H, TM).
#[allow(dead_code)]
#[inline(always)]
fn chroma_predict(
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_mode: ChromaMode,
    mbx: usize,
    mby: usize,
) {
    let stride = CHROMA_STRIDE;

    match chroma_mode {
        ChromaMode::DC => {
            predict_dcpred(uws, 8, stride, mby != 0, mbx != 0);
            predict_dcpred(vws, 8, stride, mby != 0, mbx != 0);
        }
        ChromaMode::V => {
            predict_vpred(uws, 8, 1, 1, stride);
            predict_vpred(vws, 8, 1, 1, stride);
        }
        ChromaMode::H => {
            predict_hpred(uws, 8, 1, 1, stride);
            predict_hpred(vws, 8, 1, 1, stride);
        }
        ChromaMode::TM => {
            predict_tmpred(uws, 8, 1, 1, stride);
            predict_tmpred(vws, 8, 1, 1, stride);
        }
    }
}

/// Dispatch B-mode sub-block prediction (shared across all architectures).
#[inline(always)]
fn bpred_dispatch(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    mode: IntraMode,
    x0: usize,
    y0: usize,
    stride: usize,
) {
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

// =============================================================================
// Scalar IDCT fallbacks
// =============================================================================

/// Scalar luma IDCT (used by WASM path and scalar fallback).
#[allow(dead_code)]
#[inline(always)]
fn luma_idct_scalar(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
) {
    let stride = LUMA_STRIDE;

    if luma_mode == LumaMode::B {
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sbx + sby * 4;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                bpred_dispatch(ws, bpred[i], x0, y0, stride);

                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                }
            }
        }
    } else {
        if nz & 0xFFFF != 0 {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    if nz & (1u32 << i) != 0 {
                        let y0 = 1 + y * 4;
                        let x0 = 1 + x * 4;
                        let rb = coeff_block(coeff_blocks, i);
                        idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                    }
                }
            }
        }
    }
}

/// Scalar chroma IDCT (used by WASM path and scalar fallback).
#[allow(dead_code)]
#[inline(always)]
fn chroma_idct_scalar(
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    nz: u32,
) {
    let stride = CHROMA_STRIDE;

    if nz & 0xFF_0000 != 0 {
        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let u_idx = 16 + i;
                let v_idx = 20 + i;
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                if nz & (1u32 << u_idx) != 0 {
                    let urb = coeff_block(coeff_blocks, u_idx);
                    idct_add_residue_and_clear(uws, urb, y0, x0, stride);
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb = coeff_block(coeff_blocks, v_idx);
                    idct_add_residue_and_clear(vws, vrb, y0, x0, stride);
                }
            }
        }
    }
}

// =============================================================================
// Scalar fallback (no SIMD at all)
// =============================================================================

/// Full scalar luma path: predict + IDCT.
#[cold]
#[inline(never)]
fn process_luma_mb_scalar(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_entry: &MbRowEntry,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    luma_predict(ws, mb_entry.luma_mode, &mb_entry.bpred, mbx, mby);
    luma_idct_scalar(ws, coeff_blocks, mb_entry.luma_mode, &mb_entry.bpred, nz);
}

/// Full scalar chroma path: predict + IDCT.
#[cold]
#[inline(never)]
fn process_chroma_mb_scalar(
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_entry: &MbRowEntry,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    chroma_predict(uws, vws, mb_entry.chroma_mode, mbx, mby);
    chroma_idct_scalar(uws, vws, coeff_blocks, nz);
}

// =============================================================================
// Border extraction + cache copy
// =============================================================================

/// Extract luma borders from workspace and copy pixels to cache.
///
/// After prediction + IDCT, the workspace contains the reconstructed 16x16
/// luma block in rows 1..=16, columns 1..=16 (stride = LUMA_STRIDE = 32).
/// This extracts the right column and bottom row for neighbor prediction,
/// then copies the 16x16 interior to the cache row buffer.
#[inline(always)]
fn extract_luma_borders_and_copy_to_cache(
    ws: &[u8; LUMA_BLOCK_SIZE],
    cache_y: &mut [u8],
    cache_y_stride: usize,
    extra_y_rows: usize,
    mbx: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
) {
    let stride = LUMA_STRIDE;

    // Extract left border for next MB:
    // Corner = ws[row 0, col 16] = top-right of border row
    left_border_y[0] = ws[16];
    // Column 16 of rows 1..=16
    for i in 0usize..16 {
        left_border_y[1 + i] = ws[(i + 1) * stride + 16];
    }

    // Extract top border for next row:
    // Bottom row of this MB = row 16, columns 1..=16
    top_border_y[mbx * 16..][..16].copy_from_slice(&ws[16 * stride + 1..][..16]);

    // Copy workspace interior to cache.
    // Cache layout: [extra_y_rows rows][16 rows for current MB row]
    let cache_y_offset = extra_y_rows * cache_y_stride;
    let region_start = cache_y_offset + mbx * 16;
    let region_len = 15 * cache_y_stride + 16;
    let cache_region = &mut cache_y[region_start..region_start + region_len];
    for y in 0usize..16 {
        let src_start = (1 + y) * stride + 1;
        cache_region[y * cache_y_stride..][..16].copy_from_slice(&ws[src_start..][..16]);
    }
}

/// Extract chroma borders from workspace and copy pixels to cache.
///
/// After prediction + IDCT, the workspace contains the reconstructed 8x8
/// chroma block in rows 1..=8, columns 1..=8 (stride = CHROMA_STRIDE = 32).
#[inline(always)]
fn extract_chroma_borders_and_copy_to_cache(
    chroma_ws: &[u8; CHROMA_BLOCK_SIZE],
    cache: &mut [u8],
    cache_uv_stride: usize,
    extra_y_rows: usize,
    mbx: usize,
    top_border: &mut [u8],
    left_border: &mut [u8; 9],
) {
    let stride = CHROMA_STRIDE;

    // Extract left border for next MB:
    // Corner = ws[row 0, col 8] = top-right of border row
    left_border[0] = chroma_ws[8];
    // Column 8 of rows 1..=8
    for i in 0usize..8 {
        left_border[1 + i] = chroma_ws[(i + 1) * stride + 8];
    }

    // Extract top border for next row:
    // Bottom row of this MB = row 8, columns 1..=8
    top_border[mbx * 8..][..8].copy_from_slice(&chroma_ws[8 * stride + 1..][..8]);

    // Copy workspace interior to cache.
    let extra_uv_rows = extra_y_rows / 2;
    let cache_uv_offset = extra_uv_rows * cache_uv_stride;
    let region_start = cache_uv_offset + mbx * 8;
    let region_len = 7 * cache_uv_stride + 8;
    let cache_region = &mut cache[region_start..region_start + region_len];
    for y in 0usize..8 {
        let src_start = (1 + y) * stride + 1;
        cache_region[y * cache_uv_stride..][..8].copy_from_slice(&chroma_ws[src_start..][..8]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a simple DC prediction + zero IDCT produces uniform output.
    #[test]
    fn test_luma_dc_prediction_to_cache() {
        let mut ws = [0u8; LUMA_BLOCK_SIZE];
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];

        // Set up borders: top border all 100, left border all 100
        let mbwidth = 4;
        let mut top_border_y = vec![100u8; mbwidth * 16 + 4 + 16];
        let mut left_border_y = [100u8; 17];

        let mb_entry = MbRowEntry {
            luma_mode: LumaMode::DC,
            chroma_mode: ChromaMode::DC,
            bpred: [IntraMode::DC; 16],
            segmentid: 0,
            coeffs_skipped: true,
            non_zero_blocks: 0,
            non_zero_dct: false,
            has_nonzero_uv_ac: false,
        };

        let cache_y_stride = mbwidth * 16;
        let extra_y_rows = 8;
        let cache_y_size = (extra_y_rows + 16) * cache_y_stride;
        let mut cache_y = vec![0u8; cache_y_size];

        // Process MB at (1, 1) — has both top and left borders
        process_luma_mb(
            &mut ws,
            &mut coeff_blocks,
            &mb_entry,
            &mut cache_y,
            cache_y_stride,
            extra_y_rows,
            1, // mbx
            1, // mby
            mbwidth,
            &mut top_border_y,
            &mut left_border_y,
        );

        // DC prediction with all borders = 100 should produce all 100s
        let cache_y_offset = extra_y_rows * cache_y_stride;
        for y in 0..16 {
            for x in 0..16 {
                let idx = cache_y_offset + y * cache_y_stride + 16 + x; // mbx=1
                assert_eq!(cache_y[idx], 100, "cache pixel ({x}, {y}) != 100");
            }
        }
    }

    /// Verify that vertical prediction copies the top border row.
    #[test]
    fn test_luma_v_prediction_to_cache() {
        let mut ws = [0u8; LUMA_BLOCK_SIZE];
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];

        let mbwidth = 2;
        let mut top_border_y = vec![0u8; mbwidth * 16 + 4 + 16];
        // Set top border for mbx=0 to distinct values
        for i in 0..16 {
            top_border_y[i] = (i * 10 + 20) as u8;
        }
        let mut left_border_y = [50u8; 17];

        let mb_entry = MbRowEntry {
            luma_mode: LumaMode::V,
            chroma_mode: ChromaMode::DC,
            bpred: [IntraMode::DC; 16],
            segmentid: 0,
            coeffs_skipped: true,
            non_zero_blocks: 0,
            non_zero_dct: false,
            has_nonzero_uv_ac: false,
        };

        let cache_y_stride = mbwidth * 16;
        let extra_y_rows = 0;
        let cache_y_size = 16 * cache_y_stride;
        let mut cache_y = vec![0u8; cache_y_size];

        process_luma_mb(
            &mut ws,
            &mut coeff_blocks,
            &mb_entry,
            &mut cache_y,
            cache_y_stride,
            extra_y_rows,
            0, // mbx
            1, // mby (not 0, so it reads top border)
            mbwidth,
            &mut top_border_y,
            &mut left_border_y,
        );

        // Vertical prediction: each column should match the top border value
        for y in 0..16 {
            for x in 0..16 {
                let expected = (x * 10 + 20) as u8;
                let idx = y * cache_y_stride + x;
                assert_eq!(
                    cache_y[idx], expected,
                    "cache pixel ({x}, {y}) = {} != {expected}",
                    cache_y[idx]
                );
            }
        }
    }

    /// Verify chroma DC prediction writes to cache correctly.
    #[test]
    fn test_chroma_dc_prediction_to_cache() {
        let mut uws = [0u8; CHROMA_BLOCK_SIZE];
        let mut vws = [0u8; CHROMA_BLOCK_SIZE];
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];

        let mbwidth = 2;
        let mut top_border_u = vec![80u8; mbwidth * 8];
        let mut left_border_u = [80u8; 9];
        let mut top_border_v = vec![120u8; mbwidth * 8];
        let mut left_border_v = [120u8; 9];

        let mb_entry = MbRowEntry {
            luma_mode: LumaMode::DC,
            chroma_mode: ChromaMode::DC,
            bpred: [IntraMode::DC; 16],
            segmentid: 0,
            coeffs_skipped: true,
            non_zero_blocks: 0,
            non_zero_dct: false,
            has_nonzero_uv_ac: false,
        };

        let cache_uv_stride = mbwidth * 8;
        let extra_y_rows = 0;
        let cache_uv_size = 8 * cache_uv_stride;
        let mut cache_u = vec![0u8; cache_uv_size];
        let mut cache_v = vec![0u8; cache_uv_size];

        process_chroma_mb(
            &mut uws,
            &mut vws,
            &mut coeff_blocks,
            &mb_entry,
            &mut cache_u,
            &mut cache_v,
            cache_uv_stride,
            extra_y_rows,
            1, // mbx
            1, // mby
            &mut top_border_u,
            &mut left_border_u,
            &mut top_border_v,
            &mut left_border_v,
        );

        // DC prediction with all borders = 80 → U cache should be 80
        for y in 0..8 {
            for x in 0..8 {
                let idx = y * cache_uv_stride + 8 + x; // mbx=1
                assert_eq!(cache_u[idx], 80, "U cache pixel ({x}, {y}) != 80");
                assert_eq!(cache_v[idx], 120, "V cache pixel ({x}, {y}) != 120");
            }
        }
    }

    /// Verify that IDCT residue is applied correctly.
    #[test]
    fn test_luma_dc_with_residue() {
        let mut ws = [0u8; LUMA_BLOCK_SIZE];
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];

        let mbwidth = 2;
        let mut top_border_y = vec![100u8; mbwidth * 16 + 4 + 16];
        let mut left_border_y = [100u8; 17];

        // Set a DC coefficient for block 0 (top-left 4x4)
        coeff_blocks[0] = 80; // DC value — IDCT DC: (80+4)>>3 = 10

        let mb_entry = MbRowEntry {
            luma_mode: LumaMode::DC,
            chroma_mode: ChromaMode::DC,
            bpred: [IntraMode::DC; 16],
            segmentid: 0,
            coeffs_skipped: false,
            non_zero_blocks: 1, // only block 0 has coefficients
            non_zero_dct: true,
            has_nonzero_uv_ac: false,
        };

        let cache_y_stride = mbwidth * 16;
        let extra_y_rows = 0;
        let cache_y_size = 16 * cache_y_stride;
        let mut cache_y = vec![0u8; cache_y_size];

        process_luma_mb(
            &mut ws,
            &mut coeff_blocks,
            &mb_entry,
            &mut cache_y,
            cache_y_stride,
            extra_y_rows,
            1, // mbx
            1, // mby
            mbwidth,
            &mut top_border_y,
            &mut left_border_y,
        );

        // Block 0 (rows 0-3, cols 0-3 of MB at mbx=1):
        // DC predict = 100, IDCT DC residue = (80+4)>>3 = 10
        // Result = clamp(100 + 10) = 110
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * cache_y_stride + 16 + x;
                assert_eq!(
                    cache_y[idx], 110,
                    "block 0 pixel ({x}, {y}) = {} != 110",
                    cache_y[idx]
                );
            }
        }

        // Block 1 (rows 0-3, cols 4-7 of MB): no residue, should be 100
        for y in 0..4 {
            for x in 4..8 {
                let idx = y * cache_y_stride + 16 + x;
                assert_eq!(
                    cache_y[idx], 100,
                    "block 1 pixel ({x}, {y}) = {} != 100",
                    cache_y[idx]
                );
            }
        }

        // Coefficient block 0 should be cleared after IDCT
        assert!(coeff_blocks[..16].iter().all(|&c| c == 0));
    }

    /// Verify border extraction after processing.
    #[test]
    fn test_border_extraction() {
        let mut ws = [0u8; LUMA_BLOCK_SIZE];
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];

        let mbwidth = 4;
        let mut top_border_y = vec![100u8; mbwidth * 16 + 4 + 16];
        let mut left_border_y = [100u8; 17];

        let mb_entry = MbRowEntry {
            luma_mode: LumaMode::DC,
            chroma_mode: ChromaMode::DC,
            bpred: [IntraMode::DC; 16],
            segmentid: 0,
            coeffs_skipped: true,
            non_zero_blocks: 0,
            non_zero_dct: false,
            has_nonzero_uv_ac: false,
        };

        let cache_y_stride = mbwidth * 16;
        let extra_y_rows = 0;
        let cache_y_size = 16 * cache_y_stride;
        let mut cache_y = vec![0u8; cache_y_size];

        let mbx = 2;
        process_luma_mb(
            &mut ws,
            &mut coeff_blocks,
            &mb_entry,
            &mut cache_y,
            cache_y_stride,
            extra_y_rows,
            mbx,
            1, // mby
            mbwidth,
            &mut top_border_y,
            &mut left_border_y,
        );

        // After DC prediction with all 100s:
        // left_border should be updated to 100 for all positions
        for i in 0..17 {
            assert_eq!(
                left_border_y[i], 100,
                "left_border_y[{i}] = {} != 100",
                left_border_y[i]
            );
        }

        // top_border at mbx should be updated to 100
        for i in 0..16 {
            assert_eq!(
                top_border_y[mbx * 16 + i],
                100,
                "top_border_y[{}] = {} != 100",
                mbx * 16 + i,
                top_border_y[mbx * 16 + i]
            );
        }
    }
}
