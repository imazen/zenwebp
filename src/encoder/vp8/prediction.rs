//! Block prediction and transform for encoding.
//!
//! Contains methods for generating predicted blocks, computing residuals,
//! forward/inverse DCT transform, quantization, and reconstruction for
//! both luma (16x16 and 4x4) and chroma (8x8) blocks.

#[cfg(target_arch = "x86_64")]
use archmage::{X64V3Token, prelude::*};

use crate::common::prediction::*;
use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::trellis_quantize_block;

use super::MacroblockInfo;
use super::residuals::get_coeffs0_from_block;

/// Result from luma block transform, containing quantized coefficients
/// and the prediction/reconstruction buffer for SSE computation.
pub(super) struct LumaBlockResult {
    /// Quantized coefficients for all 16 4x4 blocks.
    pub coeffs: [i32; 16 * 16],
    /// The bordered prediction/reconstruction buffer.
    /// After transform, contains reconstructed pixels (prediction + IDCT).
    pub pred_block: [u8; LUMA_BLOCK_SIZE],
    /// Quantized Y1 levels in ZIGZAG order for each of the 16 sub-blocks,
    /// captured during reconstruction — trellis levels on the trellis
    /// tiers (#35-#8), simple-quantizer levels otherwise. The recorder
    /// consumes these directly instead of re-quantizing the same DCT
    /// input a second time (levels ARE what reconstruction dequantized,
    /// so reuse is definitionally byte-identical).
    pub y1_zigzag: [[i32; 16]; 16],
    /// Whether `y1_zigzag` came from the trellis (enables the recorder's
    /// one-shot debug cross-check of the cached trellis output).
    pub y1_from_trellis: bool,
    /// Quantized Y2 (WHT) levels in ZIGZAG order — I16 only, zeros for I4.
    pub y2_zigzag: [i32; 16],
}

/// Result from chroma block transform, containing quantized coefficients
/// and the reconstruction buffer for SSE computation.
pub(super) struct ChromaBlockResult {
    /// Quantized levels in ZIGZAG order per sub-block (what the recorder
    /// emits — captured from the reconstruction quantize).
    pub zigzag: [[i32; 16]; 4],
    /// The bordered prediction/reconstruction buffer.
    pub pred_block: [u8; CHROMA_BLOCK_SIZE],
}

/// One-region SSE2 body of `fill_luma_blocks_from_predicted_16x16`: the
/// eight block-pair forward DCTs and the i16→i32 widen run inside a single
/// `target_feature` region (the dispatching version paid an arcane boundary
/// per pair).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn fill_luma_blocks_16x16_arcane(
    token: archmage::X64V3Token,
    ybuf: &[u8],
    predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
    mbx: usize,
    mby: usize,
    src_stride: usize,
    luma_blocks: &mut [i32; 16 * 16],
) {
    let pred_stride = LUMA_STRIDE;
    for block_y in 0..4 {
        for block_x_pair in 0..2 {
            let block_x = block_x_pair * 2;
            let pred_start = (block_y * 4 + 1) * pred_stride + block_x * 4 + 1;
            let src_row = mby * 16 + block_y * 4;
            let src_col = mbx * 16 + block_x * 4;
            let src_start = src_row * src_stride + src_col;

            let mut out16 = [0i16; 32];
            crate::common::transform::ftransform2_sse2(
                token,
                &ybuf[src_start..],
                &predicted_y_block[pred_start..],
                src_stride,
                pred_stride,
                &mut out16,
            );
            let out_base0 = (block_y * 4 + block_x) * 16;
            let out_base1 = out_base0 + 16;
            for i in 0..16 {
                luma_blocks[out_base0 + i] = i32::from(out16[i]);
                luma_blocks[out_base1 + i] = i32::from(out16[16 + i]);
            }
        }
    }
}

impl<'a> super::Vp8Encoder<'a> {
    // this is for all the luma modes except B
    pub(super) fn get_predicted_luma_block_16x16(
        &self,
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
    ) -> [u8; LUMA_BLOCK_SIZE] {
        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            self.macroblock_width.into(),
            &self.top_border_y,
            &self.left_border_y,
        );
        Self::predict_luma_16x16_into(&mut y_with_border, luma_mode, mbx, mby);
        y_with_border
    }

    /// Run a 16×16 luma prediction into an already-bordered workspace.
    ///
    /// Every mode writes the FULL interior and reads only the border, so a
    /// single `create_border_luma` workspace can be reused across the four
    /// I16 modes of a selection loop instead of being rebuilt per mode.
    #[inline]
    pub(super) fn predict_luma_16x16_into(
        y_with_border: &mut [u8; LUMA_BLOCK_SIZE],
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
    ) {
        let stride = LUMA_STRIDE;
        match luma_mode {
            LumaMode::V => predict_vpred(y_with_border, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(y_with_border, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(y_with_border, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(y_with_border, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => unreachable!(),
        }
    }

    // Uses fused residual+DCT for pairs of blocks
    pub(super) fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let mut luma_blocks = [0i32; 16 * 16];
        self.fill_luma_blocks_from_predicted_16x16(predicted_y_block, mbx, mby, &mut luma_blocks);
        luma_blocks
    }

    // SIMD version with output parameter: avoids redundant zero-init when caller reuses buffer
    pub(super) fn fill_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
        luma_blocks: &mut [i32; 16 * 16],
    ) {
        let pred_stride = LUMA_STRIDE;
        let src_stride = usize::from(self.macroblock_width * 16);

        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            fill_luma_blocks_16x16_arcane(
                token,
                &self.frame.ybuf,
                predicted_y_block,
                mbx,
                mby,
                src_stride,
                luma_blocks,
            );
            return;
        }

        // Process pairs of horizontally adjacent blocks using fused residual+DCT
        for block_y in 0..4 {
            for block_x_pair in 0..2 {
                let block_x = block_x_pair * 2;

                // Starting position for this pair of blocks
                let pred_start = (block_y * 4 + 1) * pred_stride + block_x * 4 + 1;
                let src_row = mby * 16 + block_y * 4;
                let src_col = mbx * 16 + block_x * 4;
                let src_start = src_row * src_stride + src_col;

                // ftransform2 outputs i16, convert to i32
                let mut out16 = [0i16; 32];
                crate::common::transform::ftransform2_from_u8(
                    &self.frame.ybuf[src_start..],
                    &predicted_y_block[pred_start..],
                    src_stride,
                    pred_stride,
                    &mut out16,
                );

                // Copy to output with i16 -> i32 conversion
                let block_idx0 = block_y * 4 + block_x;
                let block_idx1 = block_y * 4 + block_x + 1;
                let out_base0 = block_idx0 * 16;
                let out_base1 = block_idx1 * 16;

                for i in 0..16 {
                    luma_blocks[out_base0 + i] = out16[i] as i32;
                    luma_blocks[out_base1 + i] = out16[16 + i] as i32;
                }
            }
        }
    }

    // Scalar fallback
    #[allow(dead_code)]
    fn get_luma_blocks_from_predicted_16x16_scalar(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let stride = LUMA_STRIDE;
        let width = usize::from(self.macroblock_width * 16);
        let mut luma_blocks = [0i32; 16 * 16];

        for block_y in 0..4 {
            for block_x in 0..4 {
                // the index on the luma block
                let block_index = block_y * 16 * 4 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let y_data_block_index = (mby * 16 + block_y * 4) * width + mbx * 16 + block_x * 4;

                let mut block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_y_block[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        block[y * 4 + x] = i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                // transform block before copying it into main block
                transform::dct4x4(&mut block);

                luma_blocks[block_index..][..16].copy_from_slice(&block);
            }
        }

        luma_blocks
    }

    // Transforms the luma macroblock in the following ways
    // 1. Does the luma prediction and subtracts from the block
    // 2. Converts the block so each 4x4 subblock is contiguous within the block
    // 3. Does the DCT on each subblock
    // 4. Quantizes the block and dequantizes each subblock
    // 5. Calculates the quantized block - this can be used to calculate how accurate the
    // result is and is used to populate the borders for the next macroblock
    #[allow(clippy::needless_range_loop)] // x,y indices used for multiple arrays and coordinate computation
    pub(super) fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> LumaBlockResult {
        if macroblock_info.luma_mode == LumaMode::B {
            if let Some(bpred_modes) = macroblock_info.luma_bpred {
                return self.transform_luma_blocks_4x4(bpred_modes, mbx, mby);
            } else {
                panic!("Invalid, need bpred modes for luma mode B");
            }
        }

        let mut y_with_border =
            self.get_predicted_luma_block_16x16(macroblock_info.luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);

        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();

        // Y2 (DC) block: no trellis, simple quantization
        let mut coeffs0 = get_coeffs0_from_block(&luma_blocks);
        transform::wht4x4(&mut coeffs0);
        for (index, value) in coeffs0.iter_mut().enumerate() {
            *value = y2_matrix.quantize_coeff(*value, index);
        }
        // Levels in zigzag order for the recorder (avoids its re-WHT +
        // re-quantize of the same input).
        let mut y2_zigzag = [0i32; 16];
        for i in 0..16 {
            y2_zigzag[i] = coeffs0[usize::from(ZIGZAG[i])];
        }

        // Y1 blocks (AC only): use trellis if enabled
        let trellis_lambda = if self.do_trellis {
            Some(segment.lambda_trellis_i16)
        } else {
            None
        };

        // Track non-zero context for trellis (I16_AC uses ctype=0)
        // IMPORTANT: Initialize from global state to match encode_residual
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

        // Dequantize Y2 for reconstruction
        let mut y2_dequant = coeffs0;
        for (k, y2_coeff) in y2_dequant.iter_mut().enumerate() {
            *y2_coeff = y2_matrix.dequantize(*y2_coeff, k);
        }
        transform::iwht4x4(&mut y2_dequant);

        // Process each Y1 block.
        // When trellis is enabled, capture the zigzag levels per sub-block so
        // record_residual_tokens_storing can reuse them instead of re-running
        // trellis on the exact same DCT input. (#35-#8)
        let mut y1_zigzag = [[0i32; 16]; 16];
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = y * 4 + x;
                let mut block: [i32; 16] = luma_blocks[i * 16..][..16].try_into().unwrap();

                // Apply DCT was already done, now quantize
                let dequant_block = if let Some(lambda) = trellis_lambda {
                    // Trellis quantization for better RD trade-off
                    let ctx0 = (u8::from(left_nz[y]) + u8::from(top_nz[x])).min(2) as usize;
                    let zigzag_slot = &mut y1_zigzag[i];
                    // Per-block trellis dump, format-matched to TRELDBG in the
                    // instrumented libwebp's ReconstructIntra16. (#38)
                    #[cfg(feature = "mode_debug")]
                    let treldbg = std::env::var("TRELDBG").is_ok()
                        && std::env::var("TARGX").is_ok_and(|v| v == mbx.to_string())
                        && std::env::var("TARGY").is_ok_and(|v| v == mby.to_string());
                    #[cfg(feature = "mode_debug")]
                    if treldbg {
                        let dbg_in: &[i32] = &block;
                        eprintln!("TRELI16 n={i} ctx={ctx0} lam={lambda} in={dbg_in:?}");
                    }
                    let has_nz = trellis_quantize_block(
                        &mut block,
                        zigzag_slot,
                        y1_matrix,
                        lambda,
                        1, // first=1 for I16_AC (AC only)
                        &self.level_costs,
                        0, // ctype=0 for I16_AC
                        ctx0,
                        &segment.psy_config,
                    );
                    #[cfg(feature = "mode_debug")]
                    if treldbg {
                        let dbg_out: &[i32] = &zigzag_slot[..];
                        eprintln!("TRELI16 n={i} out={dbg_out:?} nz={}", u8::from(has_nz));
                    }
                    top_nz[x] = has_nz;
                    left_nz[y] = has_nz;
                    // block now contains dequantized values at natural indices
                    block
                } else {
                    // Fused SIMD quantize+dequantize, AC only (bit-identical:
                    // the sharpen boost is baked into the Y1 matrix, and the
                    // nz flags below feed only the trellis branch's context).
                    let mut quantized = [0i32; 16];
                    let mut dq = [0i32; 16];
                    crate::encoder::quantize::quantize_dequantize_ac_only_simd(
                        &block,
                        y1_matrix,
                        true,
                        &mut quantized,
                        &mut dq,
                    );
                    // Zigzag levels for the recorder; slot 0 stays 0 (I16-AC).
                    for k in 0..16 {
                        y1_zigzag[i][k] = quantized[usize::from(ZIGZAG[k])];
                    }
                    dq
                };

                // Add Y2 DC component, then fused IDCT + add-residue straight
                // into the prediction (replaces the scalar idct4x4 + staging
                // array + separate add_residue pass).
                let mut full_block = dequant_block;
                full_block[0] = y2_dequant[i];
                let dc_only = full_block[1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut full_block,
                    &mut y_with_border,
                    1 + y * 4,
                    1 + x * 4,
                    LUMA_STRIDE,
                    dc_only,
                );
            }
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * LUMA_STRIDE + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * LUMA_STRIDE + x + 1];
        }

        LumaBlockResult {
            coeffs: luma_blocks,
            pred_block: y_with_border,
            y1_zigzag,
            y1_from_trellis: trellis_lambda.is_some(),
            y2_zigzag,
        }
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    #[allow(clippy::needless_range_loop)] // sbx,sby indices used for multiple arrays and coordinate computation
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> LumaBlockResult {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = LUMA_STRIDE;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        let segment = self.get_segment_for_mb(mbx, mby);
        let trellis_lambda = if self.do_trellis {
            Some(segment.lambda_trellis_i4)
        } else {
            None
        };

        // Capture per-sub-block zigzag levels so the recorder can skip
        // re-quantizing (or re-trellising, #35-#8) the same DCT input.
        let mut y1_zigzag = [[0i32; 16]; 16];

        // Track non-zero context for trellis (I4 uses ctype=3)
        // IMPORTANT: Initialize from global state to match encode_residual
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

        // libwebp's m5 (RD_OPT_TRELLIS) I4 trellis runs in `SimpleQuantize` ->
        // `ReconstructIntra4`, which reads the trellis rate context from the
        // inter-MB neighbour nz but NEVER updates `it->top_nz`/`left_nz` between
        // sub-blocks (`quant_enc.c:844`) — so all 16 sub-blocks share one static
        // per-MB context. Its m6 (RD_OPT_TRELLIS_ALL) I4 trellis instead runs in
        // `PickBestIntra4`, which DOES update the context per sub-block. zenwebp
        // updates the context per sub-block in this final pass, which matches m6
        // but diverges from m5's static context. Under parity, freeze the context
        // at m5 (`do_trellis && !do_trellis_i4_mode`) to match `SimpleQuantize`;
        // the tuned default keeps the (more accurate) intra-MB-updated context. (#38)
        let static_i4_ctx = self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity
            && !self.do_trellis_i4_mode;
        let top_nz_frozen = top_nz;
        let left_nz_frozen = left_nz;

        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                match bpred_modes[i] {
                    IntraMode::TM => predict_tmpred(&mut y_with_border, 4, x0, y0, stride),
                    IntraMode::VE => predict_bvepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HE => predict_bhepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::DC => predict_bdcpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::LD => predict_bldpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::RD => predict_brdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VR => predict_bvrpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VL => predict_bvlpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HD => predict_bhdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HU => predict_bhupred(&mut y_with_border, x0, y0, stride),
                }

                let block_index = sby * 16 * 4 + sbx * 16;
                let mut current_subblock = [0i32; 16];

                // subtract actual values here
                let border_subblock_index = y0 * stride + x0;
                let y_data_block_index = (mby * 16 + sby * 4) * width + mbx * 16 + sbx * 4;
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_subblock_index + y * stride + x;
                        let predicted_value = y_with_border[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        current_subblock[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut current_subblock);

                luma_blocks[block_index..][..16].copy_from_slice(&current_subblock);

                // quantize and de-quantize the subblock
                // IMPORTANT: Must use same quantization method as encode_coefficients
                // to avoid prediction mismatch between encoder and decoder
                let y1_matrix = segment.y1_matrix.as_ref().unwrap();
                let has_nz = if let Some(lambda) = trellis_lambda {
                    // Trellis quantization for better RD trade-off
                    // trellis_quantize_block modifies current_subblock to contain
                    // the dequantized values (level * q)
                    let ctx0 = if static_i4_ctx {
                        (u8::from(left_nz_frozen[sby]) + u8::from(top_nz_frozen[sbx])).min(2)
                            as usize
                    } else {
                        (u8::from(left_nz[sby]) + u8::from(top_nz[sbx])).min(2) as usize
                    };
                    // Capture zigzag levels into the LumaBlockResult slot so the
                    // recorder doesn't re-run trellis on the same input. (#35-#8)
                    let zigzag_slot = &mut y1_zigzag[i];
                    trellis_quantize_block(
                        &mut current_subblock,
                        zigzag_slot,
                        y1_matrix,
                        lambda,
                        0, // first=0 for I4 (DC+AC)
                        &self.level_costs,
                        3, // ctype=3 for I4
                        ctx0,
                        &segment.psy_config,
                    )
                } else {
                    // Fused SIMD quantize+dequantize (bit-identical: sharpen
                    // is baked into the Y1 matrix; `true` applies it exactly
                    // like `quantize_coeff` does).
                    let mut quantized = [0i32; 16];
                    let mut dq = [0i32; 16];
                    let has_nz = crate::encoder::quantize::quantize_dequantize_block_simd(
                        &current_subblock,
                        y1_matrix,
                        true,
                        &mut quantized,
                        &mut dq,
                    );
                    for k in 0..16 {
                        y1_zigzag[i][k] = quantized[usize::from(ZIGZAG[k])];
                    }
                    current_subblock = dq;
                    has_nz
                };

                // Update context for next block
                top_nz[sbx] = has_nz;
                left_nz[sby] = has_nz;

                let dc_only = current_subblock[1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut current_subblock,
                    &mut y_with_border,
                    y0,
                    x0,
                    stride,
                    dc_only,
                );
            }
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * stride + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * stride + x + 1];
        }

        LumaBlockResult {
            coeffs: luma_blocks,
            pred_block: y_with_border,
            y1_zigzag,
            y1_from_trellis: trellis_lambda.is_some(),
            y2_zigzag: [0i32; 16],
        }
    }

    pub(super) fn get_predicted_chroma_block(
        &self,
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
        top_border: &[u8],
        left_border: &[u8],
    ) -> [u8; CHROMA_BLOCK_SIZE] {
        let mut chroma_with_border = create_border_chroma(mbx, mby, top_border, left_border);
        Self::predict_chroma_8x8_into(&mut chroma_with_border, chroma_mode, mbx, mby);
        chroma_with_border
    }

    /// Run an 8×8 chroma prediction into an already-bordered workspace (same
    /// reuse contract as [`Self::predict_luma_16x16_into`]).
    #[inline]
    pub(super) fn predict_chroma_8x8_into(
        chroma_with_border: &mut [u8; CHROMA_BLOCK_SIZE],
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
    ) {
        match chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(chroma_with_border, 8, CHROMA_STRIDE, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::H => {
                predict_hpred(chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::TM => {
                predict_tmpred(chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
        }
    }

    /// SIMD version: uses fused residual+DCT for pairs of blocks
    pub(super) fn get_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 4] {
        let mut chroma_blocks = [0i32; 16 * 4];
        self.fill_chroma_blocks_from_predicted(
            predicted_chroma,
            chroma_data,
            mbx,
            mby,
            &mut chroma_blocks,
        );
        chroma_blocks
    }

    /// SIMD version with output parameter: avoids redundant zero-init when caller reuses buffer
    pub(super) fn fill_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
        chroma_blocks: &mut [i32; 16 * 4],
    ) {
        let pred_stride = CHROMA_STRIDE;
        let src_stride = usize::from(self.macroblock_width * 8);

        // Process pairs of horizontally adjacent blocks using fused residual+DCT
        for block_y in 0..2 {
            // Starting position for this pair of blocks (2 blocks side by side)
            let pred_start = (block_y * 4 + 1) * pred_stride + 1;
            let src_row = mby * 8 + block_y * 4;
            let src_col = mbx * 8;
            let src_start = src_row * src_stride + src_col;

            // ftransform2 outputs i16, convert to i32
            let mut out16 = [0i16; 32];
            crate::common::transform::ftransform2_from_u8(
                &chroma_data[src_start..],
                &predicted_chroma[pred_start..],
                src_stride,
                pred_stride,
                &mut out16,
            );

            // Copy to output with i16 -> i32 conversion
            let out_base0 = block_y * 32;
            let out_base1 = out_base0 + 16;

            for i in 0..16 {
                chroma_blocks[out_base0 + i] = out16[i] as i32;
                chroma_blocks[out_base1 + i] = out16[16 + i] as i32;
            }
        }
    }

    pub(super) fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        chroma_mode: ChromaMode,
    ) -> (ChromaBlockResult, ChromaBlockResult) {
        let stride = CHROMA_STRIDE;

        let mut predicted_u = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_u,
            &self.left_border_u,
        );
        let mut predicted_v = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_v,
            &self.left_border_v,
        );

        let mut u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let mut v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        // Apply error diffusion to chroma BEFORE quantization for reconstruction.
        // This must be done here so the same quantized values are used for both
        // reconstruction (border updates) and encoding (bitstream).
        let segment = self.get_segment_for_mb(mbx, mby);
        let uv_matrix = segment.uv_matrix.clone().unwrap();
        if self.do_error_diffusion {
            self.apply_chroma_error_diffusion(&mut u_blocks, &mut v_blocks, mbx, &uv_matrix);
        }

        // Fused quantize+dequantize (one SIMD pass per block) followed by
        // fused IDCT+add-residue into the prediction — the same primitives
        // the RD path (`pick_best_uv`) uses, replacing the historical
        // per-coefficient scalar quantize → scalar dequant → scalar idct →
        // add_residue chain. Bit-identical math (the scalar tier of the
        // fused pair IS `quantize_coeff`/`* q`), ~4× fewer instructions.
        let segment = self.get_segment_for_mb(mbx, mby);
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();
        let mut u_zigzag = [[0i32; 16]; 4];
        let mut v_zigzag = [[0i32; 16]; 4];
        for (blocks, zigzag, predicted) in [
            (&u_blocks, &mut u_zigzag, &mut predicted_u),
            (&v_blocks, &mut v_zigzag, &mut predicted_v),
        ] {
            for i in 0usize..4 {
                let block: &[i32; 16] = blocks[i * 16..][..16].try_into().unwrap();
                let mut quantized = [0i32; 16];
                let mut dequantized = [0i32; 16];
                crate::encoder::quantize::quantize_dequantize_block_simd(
                    block,
                    uv_matrix,
                    false,
                    &mut quantized,
                    &mut dequantized,
                );
                for k in 0..16 {
                    zigzag[i][k] = quantized[usize::from(ZIGZAG[k])];
                }
                let (x, y) = (i % 2, i / 2);
                let dc_only = dequantized[1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut dequantized,
                    predicted,
                    1 + y * 4,
                    1 + x * 4,
                    stride,
                    dc_only,
                );
            }
        }

        // set borders
        for ((y, u_border_value), v_border_value) in self
            .left_border_u
            .iter_mut()
            .enumerate()
            .zip(self.left_border_v.iter_mut())
        {
            *u_border_value = predicted_u[y * stride + 8];
            *v_border_value = predicted_v[y * stride + 8];
        }

        for ((x, u_border_value), v_border_value) in self.top_border_u[mbx * 8..][..8]
            .iter_mut()
            .enumerate()
            .zip(self.top_border_v[mbx * 8..][..8].iter_mut())
        {
            *u_border_value = predicted_u[8 * stride + x + 1];
            *v_border_value = predicted_v[8 * stride + x + 1];
        }

        (
            ChromaBlockResult {
                zigzag: u_zigzag,
                pred_block: predicted_u,
            },
            ChromaBlockResult {
                zigzag: v_zigzag,
                pred_block: predicted_v,
            },
        )
    }
}
