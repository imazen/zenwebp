//! Intra prediction and border update functions for VP8 decoder.
//!
//! Contains luma/chroma prediction dispatch, scalar fallbacks, and border updates.

use super::*;

impl<'a> Vp8Decoder<'a> {
    pub(super) fn intra_predict_luma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        simd_token: SimdTokenType,
    ) {
        let stride = LUMA_STRIDE;
        let mw = self.mbwidth as usize;
        // Reuse persistent workspace — only borders are updated; interior is overwritten
        // by prediction + IDCT before any reads occur (raster-order processing invariant).
        let ws = &mut self.luma_ws;
        update_border_luma(ws, mbx, mby, mw, &self.top_border_y, &self.left_border_y);

        let nz = mb.non_zero_blocks;

        // Dispatch prediction + IDCT to multi-tier pipeline when SIMD available.
        // This puts all prediction loops and IDCT in a single target_feature region,
        // enabling AVX2 autovectorization of prediction and eliminating per-block
        // dispatch overhead (24 token checks per MB).
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        ))]
        if let Some(token) = simd_token {
            super::super::predict_simd::process_luma_mb(
                token,
                ws,
                &mut self.coeff_blocks,
                mb.luma_mode,
                &mb.bpred,
                nz,
                mbx,
                mby,
            );
        } else {
            Self::intra_predict_luma_scalar(ws, &mut self.coeff_blocks, mb, nz, mbx, mby);
        }

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        )))]
        {
            let _ = simd_token;
            Self::intra_predict_luma_scalar(ws, &mut self.coeff_blocks, mb, nz, mbx, mby);
        }

        self.left_border_y[0] = ws[16];

        for (i, left) in self.left_border_y[1..][..16].iter_mut().enumerate() {
            *left = ws[(i + 1) * stride + 16];
        }

        self.top_border_y[mbx * 16..][..16].copy_from_slice(&ws[16 * stride + 1..][..16]);

        // Write to row cache instead of final buffer
        // Cache layout: [extra_y_rows rows][16 rows for current MB row]
        // Pre-slice the destination region once to eliminate per-iteration bounds checks.
        let cache_y_offset = self.extra_y_rows * self.cache_y_stride;
        let cache_y_stride = self.cache_y_stride;
        let region_start = cache_y_offset + mbx * 16;
        let region_len = 15 * cache_y_stride + 16;
        let cache_region = &mut self.cache_y[region_start..region_start + region_len];
        for y in 0usize..16 {
            let src_start = (1 + y) * stride + 1;
            cache_region[y * cache_y_stride..][..16].copy_from_slice(&ws[src_start..][..16]);
        }
    }

    pub(super) fn intra_predict_chroma(
        &mut self,
        mbx: usize,
        mby: usize,
        mb: &MacroBlock,
        simd_token: SimdTokenType,
    ) {
        let stride = CHROMA_STRIDE;

        // Reuse persistent workspaces — avoids re-zeroing 288 bytes x 2 per macroblock.
        let uws = &mut self.chroma_u_ws;
        let vws = &mut self.chroma_v_ws;
        update_border_chroma(uws, mbx, mby, &self.top_border_u, &self.left_border_u);
        update_border_chroma(vws, mbx, mby, &self.top_border_v, &self.left_border_v);

        let nz = mb.non_zero_blocks;

        // Dispatch prediction + IDCT to multi-tier pipeline when SIMD available.
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        ))]
        if let Some(token) = simd_token {
            super::super::predict_simd::process_chroma_mb(
                token,
                uws,
                vws,
                &mut self.coeff_blocks,
                mb.chroma_mode,
                nz,
                mbx,
                mby,
            );
        } else {
            Self::intra_predict_chroma_scalar(uws, vws, &mut self.coeff_blocks, mb, nz, mbx, mby);
        }

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        )))]
        {
            let _ = simd_token;
            Self::intra_predict_chroma_scalar(uws, vws, &mut self.coeff_blocks, mb, nz, mbx, mby);
        }

        set_chroma_border(&mut self.left_border_u, &mut self.top_border_u, uws, mbx);
        set_chroma_border(&mut self.left_border_v, &mut self.top_border_v, vws, mbx);

        // Write to row cache instead of final buffer
        // Pre-slice destination regions once to eliminate per-iteration bounds checks.
        let extra_uv_rows = self.extra_y_rows / 2;
        let cache_uv_stride = self.cache_uv_stride;
        let cache_uv_offset = extra_uv_rows * cache_uv_stride;
        let uv_region_start = cache_uv_offset + mbx * 8;
        let uv_region_len = 7 * cache_uv_stride + 8;
        let cache_u_region = &mut self.cache_u[uv_region_start..uv_region_start + uv_region_len];
        let cache_v_region = &mut self.cache_v[uv_region_start..uv_region_start + uv_region_len];
        for y in 0usize..8 {
            let ws_index = (1 + y) * stride + 1;
            cache_u_region[y * cache_uv_stride..][..8].copy_from_slice(&uws[ws_index..][..8]);
            cache_v_region[y * cache_uv_stride..][..8].copy_from_slice(&vws[ws_index..][..8]);
        }
    }

    /// Scalar fallback for luma prediction + IDCT (no SIMD token available).
    #[cold]
    #[inline(never)]
    fn intra_predict_luma_scalar(
        ws: &mut [u8; LUMA_BLOCK_SIZE],
        coeff_blocks: &mut [i32],
        mb: &MacroBlock,
        nz: u32,
        mbx: usize,
        mby: usize,
    ) {
        let stride = LUMA_STRIDE;

        match mb.luma_mode {
            LumaMode::V => predict_vpred(ws, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(ws, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(ws, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(ws, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => {
                for sby in 0usize..4 {
                    for sbx in 0usize..4 {
                        let i = sbx + sby * 4;
                        let y0 = sby * 4 + 1;
                        let x0 = sbx * 4 + 1;

                        match mb.bpred[i] {
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

                        let rb: &mut [i32; 16] =
                            (&mut coeff_blocks[i * 16..][..16]).try_into().unwrap();
                        if nz & (1u32 << i) != 0 {
                            idct_add_residue_and_clear(ws, rb, y0, x0, stride);
                        }
                    }
                }
            }
        }

        if mb.luma_mode != LumaMode::B && nz & 0xFFFF != 0 {
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

    /// Scalar fallback for chroma prediction + IDCT (no SIMD token available).
    #[cold]
    #[inline(never)]
    fn intra_predict_chroma_scalar(
        uws: &mut [u8; CHROMA_BLOCK_SIZE],
        vws: &mut [u8; CHROMA_BLOCK_SIZE],
        coeff_blocks: &mut [i32],
        mb: &MacroBlock,
        nz: u32,
        mbx: usize,
        mby: usize,
    ) {
        let stride = CHROMA_STRIDE;

        match mb.chroma_mode {
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
}

// set border
fn set_chroma_border(
    left_border: &mut [u8],
    top_border: &mut [u8],
    chroma_block: &[u8; CHROMA_BLOCK_SIZE],
    mbx: usize,
) {
    let stride = CHROMA_STRIDE;
    // top left is top right of previous chroma block
    left_border[0] = chroma_block[8];

    // left border — chroma_block is fixed-size so stride offsets are provably in-bounds
    for (i, left) in left_border[1..][..8].iter_mut().enumerate() {
        *left = chroma_block[(i + 1) * stride + 8];
    }

    // Pre-slice top_border to eliminate repeated mbx*8 bounds check
    let top_region = &mut top_border[mbx * 8..][..8];
    top_region.copy_from_slice(&chroma_block[8 * stride + 1..8 * stride + 9]);
}
