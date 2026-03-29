//! Fused MB-row pipeline for the v2 VP8 decoder.
//!
//! A single `#[arcane]` entry point per architecture covers all post-parsing
//! work for an entire macroblock row: prediction + IDCT for every MB, then
//! loop filtering. This eliminates multiple target_feature boundaries per row
//! (previously 3+: one per luma MB, one per chroma MB, one for the filter).
//!
//! Inner functions use `#[inline(always)]` to inherit the caller's target
//! features when inlined into the `#[arcane]` body.

#![allow(clippy::too_many_arguments)]

use archmage::prelude::*;

use super::MbRowEntry;
use crate::common::prediction::*;
use crate::common::types::*;
use crate::decoder::loop_filter::{self, MbFilterParams};

// =============================================================================
// Public dispatch: filter an entire MB row
// =============================================================================

/// Filter a row of macroblocks using the best available SIMD path.
///
/// Called after all MBs in the row have been predict+IDCT'd to cache.
#[inline(always)]
pub(super) fn filter_mb_row(
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_filter_params: &[MbFilterParams],
) {
    incant!(
        filter_mb_row_dispatch(
            cache_y,
            cache_u,
            cache_v,
            cache_y_stride,
            cache_uv_stride,
            extra_y_rows,
            filter_type,
            mby,
            mb_filter_params
        ),
        [v3, neon, wasm128, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn filter_mb_row_dispatch_v3(
    token: X64V3Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_filter_params: &[MbFilterParams],
) {
    loop_filter::filter_row_simd(
        token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn filter_mb_row_dispatch_neon(
    token: NeonToken,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_filter_params: &[MbFilterParams],
) {
    loop_filter::filter_row_simd(
        token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn filter_mb_row_dispatch_wasm128(
    token: Wasm128Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_filter_params: &[MbFilterParams],
) {
    loop_filter::filter_row_simd(
        token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

#[inline(always)]
fn filter_mb_row_dispatch_scalar(
    _token: ScalarToken,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_filter_params: &[MbFilterParams],
) {
    filter_row_scalar(
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

// =============================================================================
// Full row pipeline (predict+IDCT+filter) — for future use
// =============================================================================

/// Process an entire macroblock row: predict+IDCT all MBs, then filter.
///
/// Note: Currently unused because the decode loop needs to interleave
/// parsing and predict/IDCT per-MB (coefficients are consumed immediately).
/// Kept for potential future use with per-MB coefficient storage.
#[allow(dead_code)]
#[inline(always)]
fn process_mb_row(
    luma_ws: &mut [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_row_data: &[MbRowEntry],
    mb_filter_params: &[MbFilterParams],
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
    mby: usize,
    mbwidth: usize,
    filter_type: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::{SimdToken, X64V3Token};
        if let Some(token) = X64V3Token::summon() {
            process_mb_row_v3(
                token,
                luma_ws,
                chroma_u_ws,
                chroma_v_ws,
                coeff_blocks,
                mb_row_data,
                mb_filter_params,
                cache_y,
                cache_u,
                cache_v,
                cache_y_stride,
                cache_uv_stride,
                extra_y_rows,
                top_border_y,
                left_border_y,
                top_border_u,
                left_border_u,
                top_border_v,
                left_border_v,
                mby,
                mbwidth,
                filter_type,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            process_mb_row_neon(
                token,
                luma_ws,
                chroma_u_ws,
                chroma_v_ws,
                coeff_blocks,
                mb_row_data,
                mb_filter_params,
                cache_y,
                cache_u,
                cache_v,
                cache_y_stride,
                cache_uv_stride,
                extra_y_rows,
                top_border_y,
                left_border_y,
                top_border_u,
                left_border_u,
                top_border_v,
                left_border_v,
                mby,
                mbwidth,
                filter_type,
            );
            return;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            process_mb_row_wasm(
                token,
                luma_ws,
                chroma_u_ws,
                chroma_v_ws,
                coeff_blocks,
                mb_row_data,
                mb_filter_params,
                cache_y,
                cache_u,
                cache_v,
                cache_y_stride,
                cache_uv_stride,
                extra_y_rows,
                top_border_y,
                left_border_y,
                top_border_u,
                left_border_u,
                top_border_v,
                left_border_v,
                mby,
                mbwidth,
                filter_type,
            );
            return;
        }
    }

    // Scalar fallback
    predict_idct_all_mbs_scalar(
        luma_ws,
        chroma_u_ws,
        chroma_v_ws,
        coeff_blocks,
        mb_row_data,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        top_border_y,
        left_border_y,
        top_border_u,
        left_border_u,
        top_border_v,
        left_border_v,
        mby,
        mbwidth,
    );
    filter_row_scalar(
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

// =============================================================================
// x86_64: AVX2+FMA pipeline — single #[arcane] for predict+IDCT
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn process_mb_row_v3(
    _token: archmage::X64V3Token,
    luma_ws: &mut [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_row_data: &[MbRowEntry],
    mb_filter_params: &[MbFilterParams],
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
    mby: usize,
    mbwidth: usize,
    filter_type: bool,
) {
    // Phase B: predict+IDCT for all MBs in this row
    for mbx in 0..mbwidth {
        let mb_entry = &mb_row_data[mbx];

        // Luma
        update_border_luma(luma_ws, mbx, mby, mbwidth, top_border_y, left_border_y);
        luma_predict_inner(luma_ws, mb_entry.luma_mode, &mb_entry.bpred, mbx, mby);
        luma_idct_x86(
            _token,
            luma_ws,
            coeff_blocks,
            mb_entry.luma_mode,
            &mb_entry.bpred,
            mb_entry.non_zero_blocks,
        );
        extract_luma_borders_and_copy_to_cache(
            luma_ws,
            cache_y,
            cache_y_stride,
            extra_y_rows,
            mbx,
            top_border_y,
            left_border_y,
        );

        // Chroma
        update_border_chroma(chroma_u_ws, mbx, mby, top_border_u, left_border_u);
        update_border_chroma(chroma_v_ws, mbx, mby, top_border_v, left_border_v);
        chroma_predict_inner(chroma_u_ws, chroma_v_ws, mb_entry.chroma_mode, mbx, mby);
        chroma_idct_x86(
            _token,
            chroma_u_ws,
            chroma_v_ws,
            coeff_blocks,
            mb_entry.non_zero_blocks,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_u_ws,
            cache_u,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_u,
            left_border_u,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_v_ws,
            cache_v,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_v,
            left_border_v,
        );
    }

    // Phase C: loop filter (calls the existing #[arcane] filter_row_simd)
    loop_filter::filter_row_simd(
        _token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

// =============================================================================
// aarch64: NEON pipeline
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn process_mb_row_neon(
    _token: archmage::NeonToken,
    luma_ws: &mut [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_row_data: &[MbRowEntry],
    mb_filter_params: &[MbFilterParams],
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
    mby: usize,
    mbwidth: usize,
    filter_type: bool,
) {
    for mbx in 0..mbwidth {
        let mb_entry = &mb_row_data[mbx];

        update_border_luma(luma_ws, mbx, mby, mbwidth, top_border_y, left_border_y);
        luma_predict_inner(luma_ws, mb_entry.luma_mode, &mb_entry.bpred, mbx, mby);
        luma_idct_neon(
            _token,
            luma_ws,
            coeff_blocks,
            mb_entry.luma_mode,
            &mb_entry.bpred,
            mb_entry.non_zero_blocks,
        );
        extract_luma_borders_and_copy_to_cache(
            luma_ws,
            cache_y,
            cache_y_stride,
            extra_y_rows,
            mbx,
            top_border_y,
            left_border_y,
        );

        update_border_chroma(chroma_u_ws, mbx, mby, top_border_u, left_border_u);
        update_border_chroma(chroma_v_ws, mbx, mby, top_border_v, left_border_v);
        chroma_predict_inner(chroma_u_ws, chroma_v_ws, mb_entry.chroma_mode, mbx, mby);
        chroma_idct_neon(
            _token,
            chroma_u_ws,
            chroma_v_ws,
            coeff_blocks,
            mb_entry.non_zero_blocks,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_u_ws,
            cache_u,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_u,
            left_border_u,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_v_ws,
            cache_v,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_v,
            left_border_v,
        );
    }

    loop_filter::filter_row_simd(
        _token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

// =============================================================================
// wasm32: WASM SIMD128 pipeline
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn process_mb_row_wasm(
    _token: archmage::Wasm128Token,
    luma_ws: &mut [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_row_data: &[MbRowEntry],
    mb_filter_params: &[MbFilterParams],
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
    mby: usize,
    mbwidth: usize,
    filter_type: bool,
) {
    for mbx in 0..mbwidth {
        let mb_entry = &mb_row_data[mbx];

        update_border_luma(luma_ws, mbx, mby, mbwidth, top_border_y, left_border_y);
        luma_predict_inner(luma_ws, mb_entry.luma_mode, &mb_entry.bpred, mbx, mby);
        luma_idct_scalar(
            luma_ws,
            coeff_blocks,
            mb_entry.luma_mode,
            &mb_entry.bpred,
            mb_entry.non_zero_blocks,
        );
        extract_luma_borders_and_copy_to_cache(
            luma_ws,
            cache_y,
            cache_y_stride,
            extra_y_rows,
            mbx,
            top_border_y,
            left_border_y,
        );

        update_border_chroma(chroma_u_ws, mbx, mby, top_border_u, left_border_u);
        update_border_chroma(chroma_v_ws, mbx, mby, top_border_v, left_border_v);
        chroma_predict_inner(chroma_u_ws, chroma_v_ws, mb_entry.chroma_mode, mbx, mby);
        chroma_idct_scalar(
            chroma_u_ws,
            chroma_v_ws,
            coeff_blocks,
            mb_entry.non_zero_blocks,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_u_ws,
            cache_u,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_u,
            left_border_u,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_v_ws,
            cache_v,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_v,
            left_border_v,
        );
    }

    loop_filter::filter_row_simd(
        _token,
        cache_y,
        cache_u,
        cache_v,
        cache_y_stride,
        cache_uv_stride,
        extra_y_rows,
        filter_type,
        mby,
        mb_filter_params,
    );
}

// =============================================================================
// Scalar fallback (predict+IDCT)
// =============================================================================

#[cold]
#[inline(never)]
fn predict_idct_all_mbs_scalar(
    luma_ws: &mut [u8; LUMA_BLOCK_SIZE],
    chroma_u_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    chroma_v_ws: &mut [u8; CHROMA_BLOCK_SIZE],
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    mb_row_data: &[MbRowEntry],
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    top_border_y: &mut [u8],
    left_border_y: &mut [u8; 17],
    top_border_u: &mut [u8],
    left_border_u: &mut [u8; 9],
    top_border_v: &mut [u8],
    left_border_v: &mut [u8; 9],
    mby: usize,
    mbwidth: usize,
) {
    for mbx in 0..mbwidth {
        let mb_entry = &mb_row_data[mbx];

        update_border_luma(luma_ws, mbx, mby, mbwidth, top_border_y, left_border_y);
        luma_predict_inner(luma_ws, mb_entry.luma_mode, &mb_entry.bpred, mbx, mby);
        luma_idct_scalar(
            luma_ws,
            coeff_blocks,
            mb_entry.luma_mode,
            &mb_entry.bpred,
            mb_entry.non_zero_blocks,
        );
        extract_luma_borders_and_copy_to_cache(
            luma_ws,
            cache_y,
            cache_y_stride,
            extra_y_rows,
            mbx,
            top_border_y,
            left_border_y,
        );

        update_border_chroma(chroma_u_ws, mbx, mby, top_border_u, left_border_u);
        update_border_chroma(chroma_v_ws, mbx, mby, top_border_v, left_border_v);
        chroma_predict_inner(chroma_u_ws, chroma_v_ws, mb_entry.chroma_mode, mbx, mby);
        chroma_idct_scalar(
            chroma_u_ws,
            chroma_v_ws,
            coeff_blocks,
            mb_entry.non_zero_blocks,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_u_ws,
            cache_u,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_u,
            left_border_u,
        );
        extract_chroma_borders_and_copy_to_cache(
            chroma_v_ws,
            cache_v,
            cache_uv_stride,
            extra_y_rows,
            mbx,
            top_border_v,
            left_border_v,
        );
    }
}

// =============================================================================
// Prediction functions (shared across all architectures)
// =============================================================================

#[inline(always)]
fn luma_predict_inner(
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
        LumaMode::B => { /* interleaved with IDCT */ }
    }
}

#[inline(always)]
fn chroma_predict_inner(
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

#[inline(always)]
fn bpred_dispatch_inner(
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
// IDCT functions (architecture-specific)
// =============================================================================

#[cfg(target_arch = "x86_64")]
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
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sbx + sby * 4;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;
                bpred_dispatch_inner(ws, bpred[i], x0, y0, stride);
                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_sse2_inner(
                        _token, rb, ws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    } else if nz & 0xFFFF != 0 {
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

#[cfg(target_arch = "x86_64")]
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
                bpred_dispatch_inner(ws, bpred[i], x0, y0, stride);
                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    let dc_only = rb[1..].iter().all(|&c| c == 0);
                    crate::common::transform::idct_add_residue_inplace_neon_inner(
                        _token, rb, ws, y0, x0, stride, dc_only,
                    );
                }
            }
        }
    } else if nz & 0xFFFF != 0 {
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

/// Scalar IDCT for luma.
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
                bpred_dispatch_inner(ws, bpred[i], x0, y0, stride);
                if nz & (1u32 << i) != 0 {
                    let rb = coeff_block(coeff_blocks, i);
                    idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                }
            }
        }
    } else if nz & 0xFFFF != 0 {
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

/// Scalar IDCT for chroma.
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
// Scalar loop filter fallback
// =============================================================================

fn filter_row_scalar(
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    use crate::decoder::loop_filter::*;

    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = &mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        if mbx > 0 {
            if filter_type {
                simple_filter_horizontal_16_rows(
                    cache_y,
                    extra_y_rows,
                    mbx * 16,
                    cache_y_stride,
                    p.mbedge_limit,
                );
            } else {
                normal_filter_horizontal_mb_16_rows(
                    cache_y,
                    extra_y_rows,
                    mbx * 16,
                    cache_y_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.mbedge_limit,
                );
                normal_filter_horizontal_uv_mb(
                    cache_u,
                    cache_v,
                    extra_uv_rows,
                    mbx * 8,
                    cache_uv_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.mbedge_limit,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    simple_filter_horizontal_16_rows(
                        cache_y,
                        extra_y_rows,
                        mbx * 16 + x,
                        cache_y_stride,
                        p.sub_bedge_limit,
                    );
                }
            } else {
                for x in (4usize..16 - 3).step_by(4) {
                    normal_filter_horizontal_sub_16_rows(
                        cache_y,
                        extra_y_rows,
                        mbx * 16 + x,
                        cache_y_stride,
                        p.hev_threshold,
                        p.interior_limit,
                        p.sub_bedge_limit,
                    );
                }
                normal_filter_horizontal_uv_sub(
                    cache_u,
                    cache_v,
                    extra_uv_rows,
                    mbx * 8 + 4,
                    cache_uv_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.sub_bedge_limit,
                );
            }
        }

        if mby > 0 {
            if filter_type {
                simple_filter_vertical_16_cols(
                    cache_y,
                    extra_y_rows,
                    mbx * 16,
                    cache_y_stride,
                    p.mbedge_limit,
                );
            } else {
                normal_filter_vertical_mb_16_cols(
                    cache_y,
                    extra_y_rows,
                    mbx * 16,
                    cache_y_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.mbedge_limit,
                );
                normal_filter_vertical_uv_mb(
                    cache_u,
                    cache_v,
                    extra_uv_rows,
                    mbx * 8,
                    cache_uv_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.mbedge_limit,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    simple_filter_vertical_16_cols(
                        cache_y,
                        extra_y_rows + y,
                        mbx * 16,
                        cache_y_stride,
                        p.sub_bedge_limit,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    normal_filter_vertical_sub_16_cols(
                        cache_y,
                        extra_y_rows + y,
                        mbx * 16,
                        cache_y_stride,
                        p.hev_threshold,
                        p.interior_limit,
                        p.sub_bedge_limit,
                    );
                }
                normal_filter_vertical_uv_sub(
                    cache_u,
                    cache_v,
                    extra_uv_rows + 4,
                    mbx * 8,
                    cache_uv_stride,
                    p.hev_threshold,
                    p.interior_limit,
                    p.sub_bedge_limit,
                );
            }
        }
    }
}

// =============================================================================
// Border extraction + cache copy (shared across all paths)
// =============================================================================

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

    left_border_y[0] = ws[16];
    for i in 0usize..16 {
        left_border_y[1 + i] = ws[(i + 1) * stride + 16];
    }

    top_border_y[mbx * 16..][..16].copy_from_slice(&ws[16 * stride + 1..][..16]);

    let cache_y_offset = extra_y_rows * cache_y_stride;
    let region_start = cache_y_offset + mbx * 16;
    let region_len = 15 * cache_y_stride + 16;
    let cache_region = &mut cache_y[region_start..region_start + region_len];
    for y in 0usize..16 {
        let src_start = (1 + y) * stride + 1;
        cache_region[y * cache_y_stride..][..16].copy_from_slice(&ws[src_start..][..16]);
    }
}

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

    left_border[0] = chroma_ws[8];
    for i in 0usize..8 {
        left_border[1 + i] = chroma_ws[(i + 1) * stride + 8];
    }

    top_border[mbx * 8..][..8].copy_from_slice(&chroma_ws[8 * stride + 1..][..8]);

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
