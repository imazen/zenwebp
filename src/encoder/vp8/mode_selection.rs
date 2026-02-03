//! Intra prediction mode selection for macroblocks.
//!
//! Contains methods for selecting optimal I16, I4, and UV prediction modes
//! using rate-distortion cost evaluation.

use crate::common::prediction::*;
use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::{
    estimate_dc16_cost, estimate_residual_cost, get_cost_luma16, get_cost_luma4, get_cost_uv,
    get_i4_mode_cost, is_flat_coeffs, is_flat_source_16, tdisto_16x16, trellis_quantize_block,
    FIXED_COSTS_I16, FIXED_COSTS_UV, FLATNESS_LIMIT_I16, FLATNESS_LIMIT_UV, FLATNESS_PENALTY,
    RD_DISTO_MULT, VP8_WEIGHT_Y,
};

use super::{sse_16x16_luma, sse_8x8_chroma, MacroblockInfo};

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
    /// Returns (best_mode, distortion_score) for comparison against Intra4x4.
    fn pick_best_intra16(&self, mbx: usize, mby: usize) -> (LumaMode, u64) {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

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

        let mut best_mode = LumaMode::DC;
        let mut best_rd_score = i64::MAX;
        // Store best mode's cost components for final score recalculation with lambda_mode
        let mut best_coeff_cost = 0u32;
        let mut best_mode_cost = 0u16;
        let mut best_sse = 0u32;
        let mut best_spectral_disto = 0i32;

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // Skip V mode if no top row available (first row of macroblocks)
            if mode == LumaMode::V && mby == 0 {
                continue;
            }
            // Skip H mode if no left column available (first column of macroblocks)
            if mode == LumaMode::H && mbx == 0 {
                continue;
            }
            // Skip TM mode if at top-left corner (needs both top and left)
            if mode == LumaMode::TM && (mbx == 0 || mby == 0) {
                continue;
            }

            // Generate prediction for this mode
            let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);

            // === Full reconstruction for RD evaluation ===
            // 1. Compute residuals and forward DCT
            let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&pred, mbx, mby);

            // 2. Extract DC coefficients and do WHT
            let mut dc_coeffs = [0i32; 16];
            for (i, dc) in dc_coeffs.iter_mut().enumerate() {
                *dc = luma_blocks[i * 16];
            }
            let mut y2_coeffs = dc_coeffs;
            transform::wht4x4(&mut y2_coeffs);

            // 3. Quantize Y2 (DC) coefficients
            let mut y2_quant = [0i32; 16];
            for (idx, &coeff) in y2_coeffs.iter().enumerate() {
                y2_quant[idx] = y2_matrix.quantize_coeff(coeff, idx);
            }

            // 4. Quantize Y1 (AC) coefficients and collect for cost estimation
            let mut y1_quant = [[0i32; 16]; 16];
            for block_idx in 0..16 {
                for i in 1..16 {
                    // AC coefficients only (DC=0 is handled by Y2)
                    y1_quant[block_idx][i] =
                        y1_matrix.quantize_coeff(luma_blocks[block_idx * 16 + i], i);
                }
            }

            // 5. Compute coefficient cost using probability-dependent tables
            // This matches libwebp's VP8GetCostLuma16 which uses proper token probabilities
            let coeff_cost = get_cost_luma16(&y2_quant, &y1_quant, &self.level_costs, probs);

            // 6. Dequantize Y2 and do inverse WHT
            let mut y2_dequant = [0i32; 16];
            for (idx, &level) in y2_quant.iter().enumerate() {
                y2_dequant[idx] = y2_matrix.dequantize(level, idx);
            }
            transform::iwht4x4(&mut y2_dequant);

            // 7. Dequantize Y1, add DC from Y2, and do inverse DCT
            let mut reconstructed = pred;
            for block_idx in 0..16 {
                let bx = block_idx % 4;
                let by = block_idx / 4;

                let mut block = [0i32; 16];
                // AC from Y1
                for i in 1..16 {
                    block[i] = y1_matrix.dequantize(y1_quant[block_idx][i], i);
                }
                // DC from Y2
                block[0] = y2_dequant[block_idx];

                // Inverse DCT
                transform::idct4x4(&mut block);

                // Add residue to prediction
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                add_residue(&mut reconstructed, &block, y0, x0, LUMA_STRIDE);
            }

            // 8. Compute SSE between source and reconstructed (NOT prediction!)
            let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &reconstructed);

            // 9. Compute spectral distortion (TDisto) if enabled
            let spectral_disto = if tlambda > 0 {
                // Need to extract 16x16 source and reconstructed blocks for TDisto
                let mut src_block = [0u8; 256];
                let mut rec_block = [0u8; 256];
                for y in 0..16 {
                    for x in 0..16 {
                        let src_idx = (mby * 16 + y) * src_width + mbx * 16 + x;
                        src_block[y * 16 + x] = self.frame.ybuf[src_idx];
                        rec_block[y * 16 + x] = reconstructed[(y + 1) * LUMA_STRIDE + x + 1];
                    }
                }
                let td = tdisto_16x16(&src_block, &rec_block, 16, &VP8_WEIGHT_Y);
                // Scale by tlambda and round: (tlambda * td + 128) >> 8
                (tlambda as i32 * td + 128) >> 8
            } else {
                0
            };

            // 10. Apply flat source penalty if applicable
            let (d_final, sd_final) = if is_flat {
                // Check if coefficients are also flat
                let mut all_levels = [0i16; 256];
                for block_idx in 0..16 {
                    for i in 1..16 {
                        all_levels[block_idx * 16 + i] = y1_quant[block_idx][i] as i16;
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
            // score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
            let mode_cost = FIXED_COSTS_I16[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost)) * i64::from(lambda);
            let distortion = i64::from(RD_DISTO_MULT) * (i64::from(d_final) + i64::from(sd_final));
            let rd_score = rate + distortion;

            if rd_score < best_rd_score {
                best_rd_score = rd_score;
                best_mode = mode;
                // Store components for final score recalculation
                best_coeff_cost = coeff_cost;
                best_mode_cost = mode_cost;
                best_sse = d_final;
                best_spectral_disto = sd_final;
            }
        }

        // Recalculate final score using lambda_mode for I4 vs I16 comparison
        // This matches libwebp's: SetRDScore(dqm->lambda_mode, rd);
        let lambda_mode = segment.lambda_mode;
        let final_rate =
            (i64::from(best_mode_cost) + i64::from(best_coeff_cost)) * i64::from(lambda_mode);
        let final_distortion =
            i64::from(RD_DISTO_MULT) * (i64::from(best_sse) + i64::from(best_spectral_disto));
        let final_score = final_rate + final_distortion;

        // Convert to u64 for interface compatibility (score should be positive)
        (best_mode, final_score.max(0) as u64)
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
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // All 10 intra4 modes
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
        // (This matches libwebp's SetRDScore flow in PickBestIntra4)
        let lambda_i4 = segment.lambda_i4;
        let lambda_mode = segment.lambda_mode;

        // Initialize I4 running score with the BMODE_COST penalty (211 in 1/256 bits)
        // matching libwebp's PickBestIntra4: rd_best.H = 211; SetRDScore(lambda_mode, &rd_best);
        // This is the fixed overhead cost for signaling that we're using I4 mode.
        const BMODE_COST: u64 = 211;
        let initial_penalty = BMODE_COST * u64::from(lambda_mode);
        let mut running_score = initial_penalty;

        // Track total mode cost for header bit limiting
        let mut total_mode_cost = 0u32;
        // Maximum header bits for I4 modes (from libwebp)
        let max_header_bits: u32 = 256 * 16 * 16 / 4;

        // Track non-zero context for accurate coefficient cost estimation
        // top_nz[x] = whether block above has non-zero coefficients
        // left_nz[y] = whether block to left has non-zero coefficients
        let mut top_nz = [false; 4];
        let mut left_nz = [false; 4];

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
                let nz_top = if sby == 0 { false } else { top_nz[sbx] };
                let nz_left = if sbx == 0 { false } else { left_nz[sby] };

                let mut best_mode = IntraMode::DC;
                let mut best_mode_idx = 0usize;
                let mut best_block_score = u64::MAX;
                let mut best_has_nz = false;
                let mut best_quantized = [0i32; 16];
                // Track best block's SSE and coeff cost for recalculating with lambda_mode
                let mut best_sse = 0u32;
                let mut best_coeff_cost = 0u32;

                // Pre-compute all 10 I4 prediction modes at once
                let preds = I4Predictions::compute(&y_with_border, x0, y0, LUMA_STRIDE);

                // Compute source block once
                let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;
                let mut src_block = [0u8; 16];
                for y in 0..4 {
                    let src_row = src_base + y * src_width;
                    for x in 0..4 {
                        src_block[y * 4 + x] = self.frame.ybuf[src_row + x];
                    }
                }

                let y1_matrix = segment.y1_matrix.as_ref().unwrap();

                // Fast mode filtering: compute prediction SSE for all modes
                // and only try the top candidates with lowest SSE
                // This reduces DCT/IDCT calls while maintaining quality
                // Number of modes to try depends on method:
                // - method 0-3: 3 modes (fast)
                // - method 4+: 10 modes (full search, ~0.7% smaller files)
                let max_modes_to_try = match self.method {
                    0..=3 => 3,
                    _ => 10, // method 4+: try all modes
                };
                let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
                for (mode_idx, _) in MODES.iter().enumerate() {
                    let pred = preds.get(mode_idx);
                    let mut sse = 0u32;
                    for k in 0..16 {
                        let diff = i32::from(src_block[k]) - i32::from(pred[k]);
                        sse += (diff * diff) as u32;
                    }
                    mode_sse[mode_idx] = (sse, mode_idx);
                }
                // Sort by SSE (ascending) - try modes with best predictions first
                mode_sse.sort_unstable_by_key(|&(sse, _)| sse);

                // Get trellis lambda if enabled for mode selection (method >= 6)
                let trellis_lambda_i4 = if self.do_trellis_i4_mode {
                    Some(segment.lambda_trellis_i4)
                } else {
                    None
                };

                // Try only the best candidate modes (ordered by prediction SSE)
                for &(_, mode_idx) in mode_sse[..max_modes_to_try].iter() {
                    let mode = MODES[mode_idx];
                    // Get pre-computed prediction for this mode
                    let pred = preds.get(mode_idx);

                    // Compute residual using pre-computed prediction
                    let mut residual = [0i32; 16];
                    for i in 0..16 {
                        residual[i] = i32::from(src_block[i]) - i32::from(pred[i]);
                    }

                    // Transform
                    transform::dct4x4(&mut residual);

                    // Quantize - use trellis if enabled for mode selection (RD_OPT_TRELLIS_ALL)
                    // quantized_zigzag: coefficients in zigzag order (for cost estimation)
                    // quantized_natural: coefficients in natural order (for distortion calc)
                    let mut quantized_zigzag = [0i32; 16];
                    let mut quantized_natural = [0i32; 16];
                    let has_nz = if let Some(lambda) = trellis_lambda_i4 {
                        // Trellis quantization for better RD during mode selection
                        // Context: 0=neither neighbor has nz, 1=one has nz, 2=both have nz
                        let ctx0 = usize::from(nz_top) + usize::from(nz_left);
                        // ctype=3 for I4_AC (TokenType::I4AC)
                        const CTYPE_I4_AC: usize = 3;
                        let nz = trellis_quantize_block(
                            &mut residual,
                            &mut quantized_zigzag,
                            y1_matrix,
                            lambda,
                            0, // first=0 for I4 (includes DC)
                            &self.level_costs,
                            CTYPE_I4_AC,
                            ctx0,
                        );
                        // Convert zigzag to natural order for distortion calculation
                        for n in 0..16 {
                            let j = ZIGZAG[n] as usize;
                            quantized_natural[j] = quantized_zigzag[n];
                        }
                        nz
                    } else {
                        // Simple quantization (already in natural order)
                        let mut any_nz = false;
                        for (idx, &val) in residual.iter().enumerate() {
                            let q = y1_matrix.quantize_coeff(val, idx);
                            quantized_natural[idx] = q;
                            if q != 0 {
                                any_nz = true;
                            }
                        }
                        // Convert natural to zigzag for cost estimation
                        for n in 0..16 {
                            let j = ZIGZAG[n] as usize;
                            quantized_zigzag[n] = quantized_natural[j];
                        }
                        any_nz
                    };

                    // Get accurate coefficient cost using probability tables
                    // Cost functions expect zigzag order
                    let (coeff_cost, _) = get_cost_luma4(
                        &quantized_zigzag,
                        nz_top,
                        nz_left,
                        &self.level_costs,
                        probs,
                    );

                    // Compute distortion (SSE of quantized vs original)
                    // Use natural order for dequantization and IDCT
                    let mut dequantized = quantized_natural;
                    for (idx, val) in dequantized.iter_mut().enumerate() {
                        *val = y1_matrix.dequantize(*val, idx);
                    }
                    transform::idct4x4(&mut dequantized);

                    // Compute SSE between source and reconstructed using SIMD
                    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
                    let sse = crate::common::simd_sse::sse4x4_with_residual(
                        &src_block,
                        pred,
                        &dequantized,
                    );
                    #[cfg(not(all(
                        feature = "simd",
                        any(target_arch = "x86_64", target_arch = "x86")
                    )))]
                    let sse = {
                        let mut sum = 0u32;
                        for i in 0..16 {
                            let reconstructed =
                                (i32::from(pred[i]) + dequantized[i]).clamp(0, 255) as u8;
                            let diff = i32::from(src_block[i]) - i32::from(reconstructed);
                            sum += (diff * diff) as u32;
                        }
                        sum
                    };

                    // Compute RD score: distortion + lambda * (mode_cost + coeff_cost)
                    // Use rd_score_with_coeffs to avoid u16 overflow when total_rate > 65535
                    let mode_cost = get_i4_mode_cost(top_ctx, left_ctx, mode_idx);
                    let rd_score = crate::encoder::cost::rd_score_with_coeffs(
                        sse, mode_cost, coeff_cost, lambda_i4,
                    );

                    // Block-level debug output (enabled with BLOCK_DEBUG=mbx,mby,block_idx)
                    #[cfg(feature = "mode_debug")]
                    if let Ok(s) = std::env::var("BLOCK_DEBUG") {
                        let parts: Vec<_> = s.split(',').collect();
                        if parts.len() == 3 {
                            if let (Ok(dx), Ok(dy), Ok(db)) = (
                                parts[0].parse::<usize>(),
                                parts[1].parse::<usize>(),
                                parts[2].parse::<usize>(),
                            ) {
                                if dx == mbx && dy == mby && db == i {
                                    // Print context once at start
                                    static PRINTED_CTX: std::sync::atomic::AtomicBool =
                                        std::sync::atomic::AtomicBool::new(false);
                                    if !PRINTED_CTX.swap(true, std::sync::atomic::Ordering::Relaxed)
                                    {
                                        eprintln!(
                                            "Context: top_ctx={}, left_ctx={}, lambda_i4={}",
                                            top_ctx, left_ctx, lambda_i4
                                        );
                                    }
                                    eprintln!(
                                        "  Mode {:2} ({:?}): sse={:5}, mode_cost={:4}, coeff_cost={:5}, rd_score={:8}",
                                        mode_idx, mode, sse, mode_cost, coeff_cost, rd_score
                                    );
                                }
                            }
                        }
                    }

                    if rd_score < best_block_score {
                        best_block_score = rd_score;
                        best_mode = mode;
                        best_mode_idx = mode_idx;
                        best_has_nz = has_nz;
                        best_quantized = quantized_natural;
                        best_sse = sse;
                        best_coeff_cost = coeff_cost;
                    }
                }

                best_modes[i] = best_mode;
                best_mode_indices[i] = best_mode_idx;

                // Update non-zero context for subsequent blocks
                top_nz[sbx] = best_has_nz;
                left_nz[sby] = best_has_nz;

                let best_mode_cost = get_i4_mode_cost(top_ctx, left_ctx, best_mode_idx);
                total_mode_cost += u32::from(best_mode_cost);

                // Recalculate the block score with lambda_mode for accumulation
                // (matching libwebp's SetRDScore(lambda_mode, &rd_i4) before AddScore)
                // Use rd_score_with_coeffs to avoid u16 overflow
                let block_score_for_comparison = crate::encoder::cost::rd_score_with_coeffs(
                    best_sse,
                    best_mode_cost,
                    best_coeff_cost,
                    lambda_mode,
                );

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

                // Dequantize and add back for reconstruction
                let y1_matrix = segment.y1_matrix.as_ref().unwrap();
                let mut dequantized = best_quantized;
                for (idx, val) in dequantized.iter_mut().enumerate() {
                    *val = y1_matrix.dequantize(*val, idx);
                }
                transform::idct4x4(&mut dequantized);
                add_residue(&mut y_with_border, &dequantized, y0, x0, LUMA_STRIDE);
            }
        }

        // I4 wins! Return the modes and final score
        Some((best_modes, running_score))
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
    /// RD formula: score = (R + H) * lambda + RD_DISTO_MULT * D
    /// Note: UV does not use spectral distortion (TDisto).
    #[allow(clippy::needless_range_loop)] // block_idx used for both indexing and coordinate computation
    fn pick_best_uv(&self, mbx: usize, mby: usize) -> ChromaMode {
        let mbw = usize::from(self.macroblock_width);
        let chroma_width = mbw * 8;

        // Order matches FIXED_COSTS_UV
        const MODES: [ChromaMode; 4] =
            [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

        let segment = self.get_segment_for_mb(mbx, mby);
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();
        let lambda = segment.lambda_uv;

        // Use updated probabilities if available (for consistent mode selection)
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        let mut best_mode = ChromaMode::DC;
        let mut best_rd_score = i64::MAX;

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // Skip modes that need unavailable reference pixels
            if mode == ChromaMode::V && mby == 0 {
                continue;
            }
            if mode == ChromaMode::H && mbx == 0 {
                continue;
            }
            if mode == ChromaMode::TM && (mbx == 0 || mby == 0) {
                continue;
            }

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
            // 1. Compute residuals and forward DCT
            let u_blocks =
                self.get_chroma_blocks_from_predicted(&pred_u, &self.frame.ubuf, mbx, mby);
            let v_blocks =
                self.get_chroma_blocks_from_predicted(&pred_v, &self.frame.vbuf, mbx, mby);

            // 2. Quantize coefficients
            let mut uv_quant = [[0i32; 16]; 8]; // 4 U blocks + 4 V blocks

            // Process U blocks (indices 0-3)
            for block_idx in 0..4 {
                for i in 0..16 {
                    uv_quant[block_idx][i] =
                        uv_matrix.quantize_coeff(u_blocks[block_idx * 16 + i], i);
                }
            }

            // Process V blocks (indices 4-7)
            for block_idx in 0..4 {
                for i in 0..16 {
                    uv_quant[4 + block_idx][i] =
                        uv_matrix.quantize_coeff(v_blocks[block_idx * 16 + i], i);
                }
            }

            // 3. Compute coefficient cost using probability-dependent tables
            let coeff_cost = get_cost_uv(&uv_quant, &self.level_costs, probs);

            // 4. Dequantize and inverse DCT for reconstruction
            let mut reconstructed_u = pred_u;
            let mut reconstructed_v = pred_v;

            // Reconstruct U blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;

                let mut block = [0i32; 16];
                for i in 0..16 {
                    block[i] = uv_matrix.dequantize(uv_quant[block_idx][i], i);
                }
                transform::idct4x4(&mut block);

                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                add_residue(&mut reconstructed_u, &block, y0, x0, CHROMA_STRIDE);
            }

            // Reconstruct V blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;

                let mut block = [0i32; 16];
                for i in 0..16 {
                    block[i] = uv_matrix.dequantize(uv_quant[4 + block_idx][i], i);
                }
                transform::idct4x4(&mut block);

                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                add_residue(&mut reconstructed_v, &block, y0, x0, CHROMA_STRIDE);
            }

            // 4. Compute SSE between source and reconstructed (NOT prediction!)
            let sse_u = sse_8x8_chroma(&self.frame.ubuf, chroma_width, mbx, mby, &reconstructed_u);
            let sse_v = sse_8x8_chroma(&self.frame.vbuf, chroma_width, mbx, mby, &reconstructed_v);
            let sse = sse_u + sse_v;

            // 5. Apply flatness penalty for non-DC modes
            let rate_penalty = if mode_idx > 0 {
                // Check if coefficients are flat
                let mut all_levels = [0i16; 128]; // 8 blocks * 16 coeffs
                for block_idx in 0..8 {
                    for i in 0..16 {
                        all_levels[block_idx * 16 + i] = uv_quant[block_idx][i] as i16;
                    }
                }
                if is_flat_coeffs(&all_levels, 8, FLATNESS_LIMIT_UV) {
                    // Add flatness penalty: FLATNESS_PENALTY * num_blocks
                    FLATNESS_PENALTY * 8
                } else {
                    0
                }
            } else {
                0
            };

            // 6. Compute full RD score
            // score = (R + H) * lambda + RD_DISTO_MULT * D
            let mode_cost = FIXED_COSTS_UV[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost) + i64::from(rate_penalty))
                * i64::from(lambda);
            let distortion = i64::from(RD_DISTO_MULT) * i64::from(sse);
            let rd_score = rate + distortion;

            if rd_score < best_rd_score {
                best_rd_score = rd_score;
                best_mode = mode;
            }
        }

        best_mode
    }

    pub(super) fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        // Pick the best 16x16 luma mode using RD cost selection
        let (luma_mode, i16_score) = self.pick_best_intra16(mbx, mby);

        // Debug output for specific macroblock (check MB_DEBUG env var)
        // Set MB_DEBUG=x,y to debug mode selection for that macroblock
        let debug_mb = if cfg!(feature = "mode_debug") {
            std::env::var("MB_DEBUG")
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
                .is_some_and(|(dx, dy)| dx == mbx && dy == mby)
        } else {
            false
        };

        #[allow(unused_variables)]
        let debug_i16_details = if debug_mb {
            let segment = self.get_segment_for_mb(mbx, mby);
            eprintln!("=== MB({},{}) Mode Selection Debug ===", mbx, mby);
            eprintln!("I16: mode={:?}, score={}", luma_mode, i16_score);
            eprintln!(
                "  lambda_mode={}, lambda_i4={}, lambda_i16={}",
                segment.lambda_mode, segment.lambda_i4, segment.lambda_i16
            );
            true
        } else {
            false
        };

        // Method-based I4 mode selection:
        // - method 0-1: Skip I4 entirely (fastest)
        // - method 2-4: Try I4 with fast filtering
        // - method 5-6: Full I4 search
        let (luma_mode, luma_bpred) = if self.method <= 1 {
            // Fastest: I16 only, no I4 evaluation
            (luma_mode, None)
        } else {
            // For method >= 2, try I4 with early exit optimizations
            let segment = self.get_segment_for_mb(mbx, mby);
            let skip_i4_threshold = 211 * u64::from(segment.lambda_mode);

            // Skip I4 for very flat DC blocks (method 2-4)
            // For method 5-6, always try I4 for best quality
            let should_try_i4 =
                self.method >= 5 || i16_score > skip_i4_threshold || luma_mode != LumaMode::DC;

            if should_try_i4 {
                match self.pick_best_intra4(mbx, mby, i16_score) {
                    Some((modes, i4_score)) => {
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
                        if debug_mb {
                            eprintln!("I4: score >= {} (I16 wins)", i16_score);
                            eprintln!("  RESULT: I16 wins");
                        }
                        (luma_mode, None)
                    }
                }
            } else {
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

        MacroblockInfo {
            luma_mode,
            luma_bpred,
            chroma_mode,
            segment_id,
            coeffs_skipped: false,
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
