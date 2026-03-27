//! Multi-tier SIMD prediction + IDCT pipeline for the VP8 decoder.
//!
//! Combines prediction and IDCT into single `#[arcane]` entry points per
//! architecture, so that:
//! 1. Scalar prediction loops get autovectorized at higher tiers (AVX2 for V3)
//! 2. IDCT `#[rite]` functions inline without target_feature boundary overhead
//! 3. Per-block `if let Some(token)` dispatch is eliminated (one dispatch per MB)

#![allow(clippy::too_many_arguments)]

use crate::common::prediction::*;
use crate::common::types::*;

// =============================================================================
// x86_64 / x86: V3 (AVX2+FMA) pipeline
// =============================================================================

/// Process all luma prediction + IDCT for a macroblock within a single
/// target_feature(avx2,fma,...) region. Prediction loops get autovectorized
/// with 256-bit instructions, and the SSE2 IDCT inlines without boundary cost.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::arcane]
pub(super) fn process_luma_mb(
    _token: archmage::X64V3Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_luma_mb_predict(ws, luma_mode, bpred, mbx, mby);
    process_luma_mb_idct_x86(_token, ws, coeff_blocks, luma_mode, bpred, nz);
}

/// Process all chroma prediction + IDCT for a macroblock (x86 V3).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::arcane]
pub(super) fn process_chroma_mb(
    _token: archmage::X64V3Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_chroma_mb_predict(uws, vws, chroma_mode, mbx, mby);
    process_chroma_mb_idct_x86(_token, uws, vws, coeff_blocks, nz);
}

/// IDCT for luma B-mode blocks — inlines the `#[rite]` SSE2 IDCT.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::rite]
fn process_luma_mb_idct_x86(
    _token: archmage::X64V3Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
) {
    let stride = LUMA_STRIDE;

    if luma_mode == LumaMode::B {
        // B-mode: IDCT per 4x4 sub-block, interleaved with prediction
        // Note: prediction was already done above, but B-mode needs
        // predict+IDCT interleaved because later blocks reference earlier results.
        // Re-do prediction here within the SIMD region for full inlining benefit.
        // Actually — prediction was NOT done for B-mode in process_luma_mb_predict;
        // it's handled here to keep predict+IDCT interleaved.
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sbx + sby * 4;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                match bpred[i] {
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

                if nz & (1u32 << i) != 0 {
                    let rb: &mut [i32; 16] =
                        (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_simd_intrinsics::idct_add_residue_inplace_sse2_inner(
                        _token, rb, ws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    } else {
        // I16/V/H/TM/DC modes — IDCT after full-block prediction
        if nz & 0xFFFF != 0 {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    if nz & (1u32 << i) != 0 {
                        let y0 = 1 + y * 4;
                        let x0 = 1 + x * 4;
                        let rb: &mut [i32; 16] =
                            (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        let dc_only = rb[1..].iter().all(|&c| c == 0);
                        crate::common::transform_simd_intrinsics::idct_add_residue_inplace_sse2_inner(
                            _token, rb, ws, y0, x0, stride, dc_only,
                        );
                    }
                }
            }
        }
    }
}

/// IDCT for chroma blocks (x86 V3).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[archmage::rite]
fn process_chroma_mb_idct_x86(
    _token: archmage::X64V3Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
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
                    let urb: &mut [i32; 16] =
                        (&mut coeff_blocks[u_idx * 16..][..16]).try_into().unwrap();
                    let dc_only = urb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_simd_intrinsics::idct_add_residue_inplace_sse2_inner(
                        _token, urb, uws, y0, x0, stride, dc_only,
                    );
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb: &mut [i32; 16] =
                        (&mut coeff_blocks[v_idx * 16..][..16]).try_into().unwrap();
                    let dc_only = vrb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_simd_intrinsics::idct_add_residue_inplace_sse2_inner(
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

/// Process all luma prediction + IDCT for a macroblock (NEON).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub(super) fn process_luma_mb(
    _token: archmage::NeonToken,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_luma_mb_predict(ws, luma_mode, bpred, mbx, mby);
    process_luma_mb_idct_neon(_token, ws, coeff_blocks, luma_mode, bpred, nz);
}

/// Process all chroma prediction + IDCT for a macroblock (NEON).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub(super) fn process_chroma_mb(
    _token: archmage::NeonToken,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_chroma_mb_predict(uws, vws, chroma_mode, mbx, mby);
    process_chroma_mb_idct_neon(_token, uws, vws, coeff_blocks, nz);
}

/// IDCT for luma blocks (NEON).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn process_luma_mb_idct_neon(
    _token: archmage::NeonToken,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
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

                match bpred[i] {
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

                if nz & (1u32 << i) != 0 {
                    let rb: &mut [i32; 16] =
                        (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_aarch64::idct_add_residue_inplace_neon_inner(
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
                        let rb: &mut [i32; 16] =
                            (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        let dc_only = rb[1..].iter().all(|&c| c == 0);
                        crate::common::transform_aarch64::idct_add_residue_inplace_neon_inner(
                            _token, rb, ws, y0, x0, stride, dc_only,
                        );
                    }
                }
            }
        }
    }
}

/// IDCT for chroma blocks (NEON).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn process_chroma_mb_idct_neon(
    _token: archmage::NeonToken,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
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
                    let urb: &mut [i32; 16] =
                        (&mut coeff_blocks[u_idx * 16..][..16]).try_into().unwrap();
                    let dc_only = urb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_aarch64::idct_add_residue_inplace_neon_inner(
                        _token, urb, uws, y0, x0, stride, dc_only,
                    );
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb: &mut [i32; 16] =
                        (&mut coeff_blocks[v_idx * 16..][..16]).try_into().unwrap();
                    let dc_only = vrb[1..].iter().all(|&c| c == 0);
                    crate::common::transform_aarch64::idct_add_residue_inplace_neon_inner(
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

/// Process all luma prediction + IDCT for a macroblock (WASM SIMD128).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub(super) fn process_luma_mb(
    _token: archmage::Wasm128Token,
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    luma_mode: LumaMode,
    bpred: &[IntraMode; 16],
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_luma_mb_predict(ws, luma_mode, bpred, mbx, mby);
    process_luma_mb_idct_scalar(ws, coeff_blocks, luma_mode, bpred, nz);
}

/// Process all chroma prediction + IDCT for a macroblock (WASM SIMD128).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub(super) fn process_chroma_mb(
    _token: archmage::Wasm128Token,
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
    chroma_mode: ChromaMode,
    nz: u32,
    mbx: usize,
    mby: usize,
) {
    process_chroma_mb_predict(uws, vws, chroma_mode, mbx, mby);
    process_chroma_mb_idct_scalar(uws, vws, coeff_blocks, nz);
}

// =============================================================================
// Shared prediction functions — compiled with caller's target_feature
// =============================================================================

/// Run prediction for non-B luma modes (V, H, TM, DC).
/// B-mode prediction is interleaved with IDCT in the arch-specific functions
/// because later sub-blocks reference earlier sub-block results.
#[allow(dead_code)]
#[inline(always)]
fn process_luma_mb_predict(
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
            // B-mode prediction is interleaved with IDCT in the IDCT functions
        }
    }
}

/// Run prediction for chroma (DC, V, H, TM).
#[allow(dead_code)]
#[inline(always)]
fn process_chroma_mb_predict(
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

// =============================================================================
// Scalar IDCT fallback (used by WASM path which has WASM SIMD for prediction
// autovectorization but no intrinsic IDCT implementation)
// =============================================================================

/// Scalar IDCT for luma blocks (used by WASM and fallback paths).
#[allow(dead_code)]
#[inline(always)]
fn process_luma_mb_idct_scalar(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
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

                match bpred[i] {
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

                if nz & (1u32 << i) != 0 {
                    let rb: &mut [i32; 16] =
                        (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
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
                        let rb: &mut [i32; 16] =
                            (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                    }
                }
            }
        }
    }
}

/// Scalar IDCT for chroma blocks (used by WASM and fallback paths).
#[allow(dead_code)]
#[inline(always)]
fn process_chroma_mb_idct_scalar(
    uws: &mut [u8; CHROMA_BLOCK_SIZE],
    vws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32],
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
                    let urb: &mut [i32; 16] =
                        (&mut coeff_blocks[u_idx * 16..][..16]).try_into().unwrap();
                    idct_add_residue_and_clear(uws, urb, y0, x0, stride);
                }

                if nz & (1u32 << v_idx) != 0 {
                    let vrb: &mut [i32; 16] =
                        (&mut coeff_blocks[v_idx * 16..][..16]).try_into().unwrap();
                    idct_add_residue_and_clear(vws, vrb, y0, x0, stride);
                }
            }
        }
    }
}
