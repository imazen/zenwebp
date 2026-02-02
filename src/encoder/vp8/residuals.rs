//! Residual coefficient encoding and statistics recording.
//!
//! Contains methods for encoding DCT coefficients into the bitstream,
//! trellis-optimized quantization dispatch, error diffusion, and
//! probability statistics recording.

use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::{record_coeffs, trellis_quantize_block, TokenType};
use super::MacroblockInfo;

impl<'a> super::Vp8Encoder<'a> {
    /// Apply Floyd-Steinberg-like error diffusion to chroma DC coefficients.
    ///
    /// This reduces banding artifacts in smooth gradients by spreading quantization
    /// error to neighboring blocks. The pattern for a 2x2 grid of chroma blocks:
    /// ```text
    /// | c[0] | c[1] |    errors: err0, err1
    /// | c[2] | c[3] |            err2, err3
    /// ```
    ///
    /// Modifies the DC coefficients in place and stores errors for next macroblock.
    pub(super) fn apply_chroma_error_diffusion(
        &mut self,
        u_blocks: &mut [i32; 16 * 4],
        v_blocks: &mut [i32; 16 * 4],
        mbx: usize,
        uv_matrix: &crate::encoder::cost::VP8Matrix,
    ) {
        // Diffusion constants from libwebp
        const C1: i32 = 7; // fraction from top
        const C2: i32 = 8; // fraction from left
        const DSHIFT: i32 = 4;
        const DSCALE: i32 = 1;

        let q = uv_matrix.q[0] as i32;
        let iq = uv_matrix.iq[0];
        let bias = uv_matrix.bias[0];

        // Helper: add diffused error to DC, predict quantization, return error
        // Does NOT overwrite the coefficient - leaves it adjusted for encoding
        let diffuse_dc = |dc: &mut i32, top_err: i8, left_err: i8| -> i8 {
            // Add diffused error from neighbors
            let adjustment = (C1 * top_err as i32 + C2 * left_err as i32) >> (DSHIFT - DSCALE);
            *dc += adjustment;

            // Predict what quantization will produce (to compute error)
            let sign = *dc < 0;
            let abs_dc = dc.unsigned_abs();

            let zthresh = ((1u32 << 17) - 1 - bias) / iq;
            let level = if abs_dc > zthresh {
                ((abs_dc * iq + bias) >> 17) as i32
            } else {
                0
            };

            // Error = |adjusted_input| - |reconstruction|
            // This is what we'll diffuse to neighbors
            let err = abs_dc as i32 - level * q;
            let signed_err = if sign { -err } else { err };
            (signed_err >> DSCALE).clamp(-127, 127) as i8
        };

        // Process each channel's 4 blocks in scan order
        let process_channel =
            |blocks: &mut [i32; 16 * 4], top: [i8; 2], left: [i8; 2]| -> [i8; 4] {
                // Block 0 (position 0): top-left, uses top[0] and left[0]
                let err0 = diffuse_dc(&mut blocks[0], top[0], left[0]);

                // Block 1 (position 16): top-right, uses top[1] and err0
                let err1 = diffuse_dc(&mut blocks[16], top[1], err0);

                // Block 2 (position 32): bottom-left, uses err0 and left[1]
                let err2 = diffuse_dc(&mut blocks[32], err0, left[1]);

                // Block 3 (position 48): bottom-right, uses err1 and err2
                let err3 = diffuse_dc(&mut blocks[48], err1, err2);

                [err0, err1, err2, err3]
            };

        // Process U channel
        let u_errs = process_channel(u_blocks, self.top_derr[mbx][0], self.left_derr[0]);
        // Process V channel
        let v_errs = process_channel(v_blocks, self.top_derr[mbx][1], self.left_derr[1]);

        // Store errors for next macroblock
        for (ch, errs) in [(0usize, u_errs), (1, v_errs)] {
            let [_err0, err1, err2, err3] = errs;
            // left[0] = err1, left[1] = 3/4 of err3
            self.left_derr[ch][0] = err1;
            self.left_derr[ch][1] = ((3 * err3 as i32) >> 2) as i8;
            // top[0] = err2, top[1] = 1/4 of err3
            self.top_derr[mbx][ch][0] = err2;
            self.top_derr[mbx][ch][1] = (err3 as i32 - self.left_derr[ch][1] as i32) as i8;
        }
    }

    // 13 in specification, matches read_residual_data in the decoder
    pub(super) fn encode_residual_data(
        &mut self,
        macroblock_info: &MacroblockInfo,
        partition_index: usize,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        // Extract VP8 matrices and trellis lambdas upfront to avoid borrow conflicts
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();

        // Trellis lambda for Y1 blocks (I4 vs I16 mode)
        let is_i4 = macroblock_info.luma_mode == LumaMode::B;
        let y1_trellis_lambda = if self.do_trellis {
            Some(if is_i4 {
                segment.lambda_trellis_i4
            } else {
                segment.lambda_trellis_i16
            })
        } else {
            None
        };
        // Note: libwebp disables trellis for UV (DO_TRELLIS_UV = 0)
        // and for Y2 DC coefficients, so we use None for those

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);

            // wht here on the 0th coeffs
            transform::wht4x4(&mut coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;

            let has_coeffs = self.encode_coefficients(
                &coeffs0,
                partition_index,
                plane,
                complexity.into(),
                &y2_matrix,
                None, // No trellis for Y2 DC
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            // next encode luma coefficients without the 0th coeffs
            plane = Plane::YCoeff1;
        }

        // now encode the 16 luma 4x4 subblocks in the macroblock
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &y1_matrix,
                    y1_trellis_lambda,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            // set for the next macroblock
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

        // Note: Error diffusion is already applied in transform_chroma_blocks
        // so u_block_data and v_block_data contain the error-diffused values

        // encode the 4 u 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &uv_matrix,
                    None, // No trellis for UV (libwebp: DO_TRELLIS_UV = 0)
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // encode the 4 v 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &uv_matrix,
                    None, // No trellis for UV (libwebp: DO_TRELLIS_UV = 0)
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    // encodes the coefficients which is the reverse procedure of read_coefficients in the decoder
    // returns whether there was any non-zero data in the block for the complexity
    pub(super) fn encode_coefficients(
        &mut self,
        block: &[i32; 16],
        partition_index: usize,
        plane: Plane,
        complexity: usize,
        matrix: &crate::encoder::cost::VP8Matrix,
        trellis_lambda: Option<u32>,
    ) -> bool {
        // transform block
        // dc is used for the 0th coefficient, ac for the others

        let encoder = &mut self.partitions[partition_index];

        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];

        assert!(complexity <= 2);
        let mut complexity = complexity;

        // convert to zigzag and quantize using VP8Matrix biased quantization
        // this is the only lossy part of the encoding
        let mut zigzag_block = [0i32; 16];

        if let Some(lambda) = trellis_lambda {
            // Trellis quantization for better RD optimization
            let mut coeffs = *block;
            let ctype = plane as usize;
            trellis_quantize_block(
                &mut coeffs,
                &mut zigzag_block,
                matrix,
                lambda,
                first_coeff,
                &self.level_costs,
                ctype,
                complexity,
            );
        } else {
            // Simple quantization
            for i in first_coeff..16 {
                let zigzag_index = usize::from(ZIGZAG[i]);
                zigzag_block[i] = matrix.quantize_coeff(block[zigzag_index], zigzag_index);
            }
        }

        // get index of last coefficient that isn't 0
        let end_of_block_index =
            if let Some(last_non_zero_index) = zigzag_block.iter().rev().position(|x| *x != 0) {
                (15 - last_non_zero_index) + 1
            } else {
                // if it's all 0s then the first block is end of block
                0
            };

        let mut skip_eob = false;

        for index in first_coeff..end_of_block_index {
            let coeff = zigzag_block[index];

            let band = usize::from(COEFF_BANDS[index]);
            let probabilities = &probs[band][complexity];
            let start_index_token_tree = if skip_eob { 2 } else { 0 };
            let token_tree = &DCT_TOKEN_TREE;
            let token_probs = probabilities;

            let token = match coeff.abs() {
                0 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        DCT_0,
                        start_index_token_tree,
                    );

                    // never going to have an end of block after a 0, so skip checking next coeff
                    skip_eob = true;
                    DCT_0
                }

                // just encode as literal
                literal @ 1..=4 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        literal as i8,
                        start_index_token_tree,
                    );

                    skip_eob = false;
                    literal as i8
                }

                // encode the category
                value => {
                    let category = match value {
                        5..=6 => DCT_CAT1,
                        7..=10 => DCT_CAT2,
                        11..=18 => DCT_CAT3,
                        19..=34 => DCT_CAT4,
                        35..=66 => DCT_CAT5,
                        67..=2048 => DCT_CAT6,
                        _ => unreachable!(),
                    };

                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        category,
                        start_index_token_tree,
                    );

                    let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];

                    let extra = value - i32::from(DCT_CAT_BASE[(category - DCT_CAT1) as usize]);

                    let mut mask = if category == DCT_CAT6 {
                        1 << (11 - 1)
                    } else {
                        1 << (category - DCT_CAT1)
                    };

                    for &prob in category_probs.iter() {
                        if prob == 0 {
                            break;
                        }
                        let extra_bool = extra & mask > 0;
                        encoder.write_bool(extra_bool, prob);
                        mask >>= 1;
                    }

                    skip_eob = false;

                    category
                }
            };

            // encode sign if token is not zero
            if token != DCT_0 {
                // note flag means coeff is negative
                encoder.write_flag(!coeff.is_positive());
            }

            complexity = match token {
                DCT_0 => 0,
                DCT_1 => 1,
                _ => 2,
            };
        }

        // encode end of block
        if end_of_block_index < 16 {
            let band_index = usize::max(first_coeff, end_of_block_index);
            let band = usize::from(COEFF_BANDS[band_index]);
            let probabilities = &probs[band][complexity];
            encoder.write_with_tree(&DCT_TOKEN_TREE, probabilities, DCT_EOB);
        }

        // whether the block has a non zero coefficient
        end_of_block_index > 0
    }

    /// Check if all coefficients in a macroblock would quantize to zero.
    /// Used for skip detection.
    pub(super) fn check_all_coeffs_zero(
        &self,
        macroblock_info: &MacroblockInfo,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) -> bool {
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();

        // For Intra16 mode, check Y2 block (DC coefficients after WHT)
        if macroblock_info.luma_mode != LumaMode::B {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            for (idx, &val) in coeffs0.iter().enumerate() {
                if y2_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }

            // Check Y blocks (AC only for Intra16)
            for block in y_block_data.chunks_exact(16) {
                for (idx, &val) in block.iter().enumerate().skip(1) {
                    if y1_matrix.quantize_coeff(val, idx) != 0 {
                        return false;
                    }
                }
            }
        } else {
            // For Intra4 mode, check all Y coefficients (DC + AC)
            for block in y_block_data.chunks_exact(16) {
                for (idx, &val) in block.iter().enumerate() {
                    if y1_matrix.quantize_coeff(val, idx) != 0 {
                        return false;
                    }
                }
            }
        }

        // Check U blocks
        for block in u_block_data.chunks_exact(16) {
            for (idx, &val) in block.iter().enumerate() {
                if uv_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }
        }

        // Check V blocks
        for block in v_block_data.chunks_exact(16) {
            for (idx, &val) in block.iter().enumerate() {
                if uv_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }
        }

        true
    }

    /// Record token statistics for a macroblock (used in first pass of two-pass encoding).
    /// This mirrors the structure of encode_residual_data but only records stats.
    pub(super) fn record_residual_stats(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        // Extract VP8 matrices
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();

        let is_i4 = macroblock_info.luma_mode == LumaMode::B;

        // Y2 (DC transform) - only for I16 mode
        if !is_i4 {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            // Quantize to zigzag order
            let mut zigzag = [0i32; 16];
            for i in 0..16 {
                let zigzag_index = usize::from(ZIGZAG[i]);
                zigzag[i] = y2_matrix.quantize_coeff(coeffs0[zigzag_index], zigzag_index);
            }

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            record_coeffs(
                &zigzag,
                TokenType::I16DC,
                0,
                complexity.min(2) as usize,
                &mut self.proba_stats,
            );

            let has_coeffs = zigzag.iter().any(|&c| c != 0);
            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };
        }

        // Y1 blocks (AC only for I16, DC+AC for I4)
        let token_type = if is_i4 {
            TokenType::I4
        } else {
            TokenType::I16AC
        };
        let first_coeff = if is_i4 { 0 } else { 1 };

        // Get trellis lambda based on block type
        let trellis_lambda = if is_i4 {
            segment.lambda_trellis_i4
        } else {
            segment.lambda_trellis_i16
        };

        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block: &[i32; 16] = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                // Quantize to zigzag order
                let mut zigzag = [0i32; 16];

                let top = self.top_complexity[mbx].y[x];
                let ctx0 = (left + top).min(2) as usize;

                if self.do_trellis {
                    // Trellis quantization: optimizes coefficient levels for RD
                    let mut coeffs = *block;
                    // Use same ctype as record_coeffs: TokenType I4=3 or I16AC=0
                    let ctype = token_type as usize;
                    trellis_quantize_block(
                        &mut coeffs,
                        &mut zigzag,
                        &y1_matrix,
                        trellis_lambda,
                        first_coeff,
                        &self.level_costs,
                        ctype,
                        ctx0,
                    );
                } else {
                    // Simple quantization
                    for i in first_coeff..16 {
                        let zigzag_index = usize::from(ZIGZAG[i]);
                        zigzag[i] = y1_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                    }
                }

                record_coeffs(
                    &zigzag,
                    token_type,
                    first_coeff,
                    ctx0,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag[first_coeff..].iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        // U blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zigzag_index = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                }

                let top = self.top_complexity[mbx].u[x];
                let complexity = (left + top).min(2);

                record_coeffs(
                    &zigzag,
                    TokenType::Chroma,
                    0,
                    complexity as usize,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag.iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // V blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zigzag_index = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                }

                let top = self.top_complexity[mbx].v[x];
                let complexity = (left + top).min(2);

                record_coeffs(
                    &zigzag,
                    TokenType::Chroma,
                    0,
                    complexity as usize,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag.iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }
}

pub(super) fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
}
