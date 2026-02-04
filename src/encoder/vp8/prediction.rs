//! Block prediction and transform for encoding.
//!
//! Contains methods for generating predicted blocks, computing residuals,
//! forward/inverse DCT transform, quantization, and reconstruction for
//! both luma (16x16 and 4x4) and chroma (8x8) blocks.

use crate::common::prediction::*;
use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::trellis_quantize_block;

use super::residuals::get_coeffs0_from_block;
use super::{ChromaCoeffs, MacroblockInfo};

/// Result from luma block transform, containing quantized coefficients
/// and the prediction/reconstruction buffer for SSE computation.
pub(super) struct LumaBlockResult {
    /// Quantized coefficients for all 16 4x4 blocks.
    pub coeffs: [i32; 16 * 16],
    /// The bordered prediction/reconstruction buffer.
    /// After transform, contains reconstructed pixels (prediction + IDCT).
    pub pred_block: [u8; LUMA_BLOCK_SIZE],
}

/// Result from chroma block transform, containing quantized coefficients
/// and the reconstruction buffer for SSE computation.
pub(super) struct ChromaBlockResult {
    /// Quantized coefficients for 4 sub-blocks (U or V).
    pub coeffs: [i32; 16 * 4],
    /// The bordered prediction/reconstruction buffer.
    pub pred_block: [u8; CHROMA_BLOCK_SIZE],
}

impl<'a> super::Vp8Encoder<'a> {
    // this is for all the luma modes except B
    pub(super) fn get_predicted_luma_block_16x16(
        &self,
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
    ) -> [u8; LUMA_BLOCK_SIZE] {
        let stride = LUMA_STRIDE;

        let mbw = self.macroblock_width;

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        // do the prediction
        match luma_mode {
            LumaMode::V => predict_vpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut y_with_border, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => unreachable!(),
        }

        y_with_border
    }

    // gets the luma blocks with the DCT applied to them
    #[cfg(not(feature = "simd"))]
    pub(super) fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        self.get_luma_blocks_from_predicted_16x16_scalar(predicted_y_block, mbx, mby)
    }

    // SIMD version: uses fused residual+DCT for pairs of blocks
    #[cfg(feature = "simd")]
    pub(super) fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let pred_stride = LUMA_STRIDE;
        let src_stride = usize::from(self.macroblock_width * 16);
        let mut luma_blocks = [0i32; 16 * 16];

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
                crate::common::transform_simd_intrinsics::ftransform2_from_u8(
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

        luma_blocks
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

        // Process each Y1 block
        let mut dequantized_blocks = [0i32; 16 * 16];
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = y * 4 + x;
                let mut block = luma_blocks[i * 16..][..16].try_into().unwrap();

                // Apply DCT was already done, now quantize
                let dequant_block = if let Some(lambda) = trellis_lambda {
                    // Trellis quantization for better RD trade-off
                    let ctx0 = (u8::from(left_nz[y]) + u8::from(top_nz[x])).min(2) as usize;
                    let mut zigzag_out = [0i32; 16];
                    let has_nz = trellis_quantize_block(
                        &mut block,
                        &mut zigzag_out,
                        y1_matrix,
                        lambda,
                        1, // first=1 for I16_AC (AC only)
                        &self.level_costs,
                        0, // ctype=0 for I16_AC
                        ctx0,
                        &segment.psy_config,
                    );
                    top_nz[x] = has_nz;
                    left_nz[y] = has_nz;
                    // block now contains dequantized values at natural indices
                    block
                } else {
                    // Simple quantization
                    let mut has_nz = false;
                    for (idx, y_value) in block.iter_mut().enumerate() {
                        if idx == 0 {
                            *y_value = 0; // DC goes to Y2
                        } else {
                            let level = y1_matrix.quantize_coeff(*y_value, idx);
                            has_nz |= level != 0;
                            *y_value = y1_matrix.dequantize(level, idx);
                        }
                    }
                    top_nz[x] = has_nz;
                    left_nz[y] = has_nz;
                    block
                };

                // Add Y2 DC component
                let mut full_block = dequant_block;
                full_block[0] = y2_dequant[i];

                // IDCT
                transform::idct4x4(&mut full_block);
                dequantized_blocks[i * 16..][..16].copy_from_slice(&full_block);
            }
        }

        // Apply residue to each block
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = x + y * 4;
                let rb: &[i32; 16] = dequantized_blocks[i * 16..][..16].try_into().unwrap();
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                add_residue(&mut y_with_border, rb, y0, x0, LUMA_STRIDE);
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
                    let ctx0 = (u8::from(left_nz[sby]) + u8::from(top_nz[sbx])).min(2) as usize;
                    let mut zigzag_out = [0i32; 16];
                    trellis_quantize_block(
                        &mut current_subblock,
                        &mut zigzag_out,
                        y1_matrix,
                        lambda,
                        0, // first=0 for I4 (DC+AC)
                        &self.level_costs,
                        3, // ctype=3 for I4
                        ctx0,
                        &segment.psy_config,
                    )
                } else {
                    // Simple quantization
                    let mut has_nz = false;
                    for (index, y_value) in current_subblock.iter_mut().enumerate() {
                        let level = y1_matrix.quantize_coeff(*y_value, index);
                        has_nz |= level != 0;
                        *y_value = y1_matrix.dequantize(level, index);
                    }
                    has_nz
                };

                // Update context for next block
                top_nz[sbx] = has_nz;
                left_nz[sby] = has_nz;

                transform::idct4x4(&mut current_subblock);
                add_residue(&mut y_with_border, &current_subblock, y0, x0, stride);
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

        match chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(
                    &mut chroma_with_border,
                    8,
                    CHROMA_STRIDE,
                    mby != 0,
                    mbx != 0,
                );
            }
            ChromaMode::V => {
                predict_vpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::H => {
                predict_hpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
        }

        chroma_with_border
    }

    /// SIMD version: uses fused residual+DCT for pairs of blocks
    #[cfg(feature = "simd")]
    pub(super) fn get_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 4] {
        let pred_stride = CHROMA_STRIDE;
        let src_stride = usize::from(self.macroblock_width * 8);
        let mut chroma_blocks = [0i32; 16 * 4];

        // Process pairs of horizontally adjacent blocks using fused residual+DCT
        for block_y in 0..2 {
            // Starting position for this pair of blocks (2 blocks side by side)
            let pred_start = (block_y * 4 + 1) * pred_stride + 1;
            let src_row = mby * 8 + block_y * 4;
            let src_col = mbx * 8;
            let src_start = src_row * src_stride + src_col;

            // ftransform2 outputs i16, convert to i32
            let mut out16 = [0i16; 32];
            crate::common::transform_simd_intrinsics::ftransform2_from_u8(
                &chroma_data[src_start..],
                &predicted_chroma[pred_start..],
                src_stride,
                pred_stride,
                &mut out16,
            );

            // Copy to output with i16 -> i32 conversion
            // Block layout: [block0, block1, block2, block3] where blocks are 16 elements each
            // block_y=0: blocks 0,1 (indices 0-31)
            // block_y=1: blocks 2,3 (indices 32-63)
            let out_base0 = block_y * 32; // 0 or 32
            let out_base1 = out_base0 + 16; // 16 or 48

            for i in 0..16 {
                chroma_blocks[out_base0 + i] = out16[i] as i32;
                chroma_blocks[out_base1 + i] = out16[16 + i] as i32;
            }
        }

        chroma_blocks
    }

    /// Scalar fallback
    #[cfg(not(feature = "simd"))]
    pub(super) fn get_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 4] {
        let mut chroma_blocks = [0i32; 16 * 4];
        let stride = CHROMA_STRIDE;

        let chroma_width = usize::from(self.macroblock_width * 8);

        for block_y in 0..2 {
            for block_x in 0..2 {
                // the index on the chroma block
                let block_index = block_y * 16 * 2 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let chroma_data_block_index =
                    (mby * 8 + block_y * 4) * chroma_width + mbx * 8 + block_x * 4;

                let mut chroma_block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_chroma[predicted_index];
                        let actual_index = chroma_data_block_index + y * chroma_width + x;
                        let actual_value = chroma_data[actual_index];
                        chroma_block[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut chroma_block);

                chroma_blocks[block_index..][..16].copy_from_slice(&chroma_block);
            }
        }

        chroma_blocks
    }

    fn get_chroma_block_coeffs(
        &self,
        chroma_blocks: [i32; 16 * 4],
        segment: &Segment,
    ) -> ChromaCoeffs {
        let mut chroma_coeffs: ChromaCoeffs = [0i32; 16 * 4];
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();

        for (block, coeff_block) in chroma_blocks
            .chunks_exact(16)
            .zip(chroma_coeffs.chunks_exact_mut(16))
        {
            for ((index, &value), coeff) in block.iter().enumerate().zip(coeff_block.iter_mut()) {
                *coeff = uv_matrix.quantize_coeff(value, index);
            }
        }

        chroma_coeffs
    }

    fn get_dequantized_blocks_from_coeffs_chroma(
        &self,
        chroma_coeffs: &ChromaCoeffs,
        segment: &Segment,
    ) -> [i32; 16 * 4] {
        let mut dequantized_blocks = [0i32; 16 * 4];
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();

        for (coeffs_block, dequant_block) in chroma_coeffs
            .chunks_exact(16)
            .zip(dequantized_blocks.chunks_exact_mut(16))
        {
            for ((index, &coeff), dequant_value) in coeffs_block
                .iter()
                .enumerate()
                .zip(dequant_block.iter_mut())
            {
                *dequant_value = uv_matrix.dequantize(coeff, index);
            }

            transform::idct4x4(dequant_block);
        }

        dequantized_blocks
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

        let segment = self.get_segment_for_mb(mbx, mby);
        let u_coeffs = self.get_chroma_block_coeffs(u_blocks, segment);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks, segment);

        let quantized_u_residue =
            self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs, segment);
        let quantized_v_residue =
            self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs, segment);

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = quantized_u_residue[i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut predicted_u, urb, y0, x0, stride);

                let vrb: &[i32; 16] = quantized_v_residue[i * 16..][..16].try_into().unwrap();

                add_residue(&mut predicted_v, vrb, y0, x0, stride);
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
                coeffs: u_blocks,
                pred_block: predicted_u,
            },
            ChromaBlockResult {
                coeffs: v_blocks,
                pred_block: predicted_v,
            },
        )
    }
}
