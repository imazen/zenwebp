use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use super::vec_writer::VecWriter;

use super::api::ColorType;
use super::api::EncodingError;
use super::arithmetic::ArithmeticEncoder;
use super::cost::{
    analyze_image, assign_segments_kmeans, classify_image_type, compute_segment_quant,
    content_type_to_tuning, LevelCosts, ProbaStats,
};
use crate::common::prediction::*;
use crate::common::types::Frame;
use crate::common::types::*;
use crate::decoder::yuv::convert_image_y;
use crate::decoder::yuv::convert_image_yuv;

mod header;
mod mode_selection;
mod prediction;
mod residuals;

//------------------------------------------------------------------------------
// Quality to quantization index mapping
//
// Ported from libwebp src/enc/quant_enc.c

/// Convert user-facing quality (0-100) to compression factor.
/// Emulates jpeg-like behaviour where Q75 is "good quality".
/// Ported from libwebp's QualityToCompression().
fn quality_to_compression(quality: u8) -> f64 {
    let c = f64::from(quality) / 100.0;
    // Piecewise linear mapping to get jpeg-like behavior at Q75
    let linear_c = if c < 0.75 {
        c * (2.0 / 3.0)
    } else {
        2.0 * c - 1.0
    };
    // File size roughly scales as pow(quantizer, 3), so we use inverse
    super::fast_math::cbrt(linear_c)
}

/// Convert user-facing quality (0-100) to internal quant index (0-127)
/// Ported from libwebp's VP8SetSegmentParams().
fn quality_to_quant_index(quality: u8) -> u8 {
    let c = quality_to_compression(quality);
    let q = super::fast_math::round(127.0 * (1.0 - c)) as i32;
    q.clamp(0, 127) as u8
}

//------------------------------------------------------------------------------
// SSE (Sum of Squared Errors) distortion functions
//
// These measure the distortion between source and predicted blocks.
// Lower SSE = better prediction = less data to encode.

/// Compute SSE for a 16x16 luma block within bordered prediction buffer
/// Compares source YUV data against predicted block with border
#[inline]
pub(super) fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        crate::common::simd_sse::sse_16x16_luma(src_y, src_width, mbx, mby, pred)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        let mut sse = 0u32;
        let src_base = mby * 16 * src_width + mbx * 16;

        for y in 0..16 {
            let src_row = src_base + y * src_width;
            let pred_row = (y + 1) * LUMA_STRIDE + 1; // +1 for border offset

            for x in 0..16 {
                let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
    }
}

/// Compute SSE for an 8x8 chroma block within bordered prediction buffer
#[inline]
pub(super) fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        crate::common::simd_sse::sse_8x8_chroma(src_uv, src_width, mbx, mby, pred)
    }
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
    {
        let mut sse = 0u32;
        let src_base = mby * 8 * src_width + mbx * 8;

        for y in 0..8 {
            let src_row = src_base + y * src_width;
            let pred_row = (y + 1) * CHROMA_STRIDE + 1; // +1 for border offset

            for x in 0..8 {
                let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
    }
}

// currently in decoder it actually stores this information on the macroblock but that's confusing
// because it doesn't update the macroblock, just the complexity values as we decode
// this is used as the complexity per 13.3 in the decoder
#[derive(Clone, Copy, Default)]
struct Complexity {
    y2: u8,
    y: [u8; 4],
    u: [u8; 2],
    v: [u8; 2],
}

impl Complexity {
    fn clear(&mut self, include_y2: bool) {
        self.y = [0; 4];
        self.u = [0; 2];
        self.v = [0; 2];
        if include_y2 {
            self.y2 = 0;
        }
    }
}

#[derive(Default)]
struct QuantizationIndices {
    yac_abs: u8,
    ydc_delta: Option<i8>,
    y2dc_delta: Option<i8>,
    y2ac_delta: Option<i8>,
    uvdc_delta: Option<i8>,
    uvac_delta: Option<i8>,
}

/// TODO: Consider merging this with the MacroBlock from the decoder
#[derive(Clone, Copy, Default)]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,

    coeffs_skipped: bool,
}

pub(super) type ChromaCoeffs = [i32; 16 * 4];

/// Quantized zigzag coefficients for a macroblock, stored for multi-pass encoding.
/// libwebp stores quantized coefficients from pass 1 and reuses them in pass 2+.
/// These are the final quantized values (post-trellis if applicable), ready for
/// direct token recording without re-quantization.
#[derive(Clone)]
struct QuantizedMbCoeffs {
    /// Y2 DC transform coefficients (16 values), only used for I16 mode
    y2_zigzag: [i32; 16],
    /// Y1 block coefficients (16 blocks × 16 values), zigzag order
    y1_zigzag: [[i32; 16]; 16],
    /// U block coefficients (4 blocks × 16 values), zigzag order
    u_zigzag: [[i32; 16]; 4],
    /// V block coefficients (4 blocks × 16 values), zigzag order
    v_zigzag: [[i32; 16]; 4],
}

struct Vp8Encoder<'a> {
    writer: &'a mut Vec<u8>,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,
    segments_update_map: bool,
    segment_tree_probs: [Prob; 3],
    /// Segment ID for each macroblock (mb_width * mb_height)
    segment_map: Vec<u8>,

    loop_filter_adjustments: bool,
    macroblock_no_skip_coeff: Option<u8>,
    quantization_indices: QuantizationIndices,

    token_probs: TokenProbTables,
    /// Token statistics for adaptive probability updates
    proba_stats: ProbaStats,
    /// Updated probabilities computed from statistics
    updated_probs: Option<TokenProbTables>,
    /// Precomputed level costs for coefficient cost estimation
    level_costs: LevelCosts,
    /// Whether to use trellis quantization for better RD optimization
    do_trellis: bool,
    /// Whether to use chroma error diffusion to reduce banding
    do_error_diffusion: bool,
    /// Encoding method (0-6): 0=fastest, 6=best quality
    method: u8,
    /// Spatial noise shaping strength (0-100)
    sns_strength: u8,
    /// Loop filter strength (0-100)
    filter_strength: u8,
    /// Loop filter sharpness (0-7)
    filter_sharpness: u8,
    /// Number of segments (1-4)
    num_segments: u8,
    /// Selected preset (used for Auto detection)
    preset: super::api::Preset,

    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    /// Partitions of encoders for the macroblock coefficient data
    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,

    // Error diffusion state for chroma DC coefficients
    // This implements Floyd-Steinberg-like error spreading to reduce banding
    // top_derr[mbx][channel][0..2] = errors from block above
    // left_derr[channel][0..2] = errors from block to the left
    top_derr: Vec<[[i8; 2]; 2]>,
    left_derr: [[i8; 2]; 2],

    /// Token buffer for deferred coefficient encoding (method >= 2).
    /// Stores bit-level tokens during the recording pass for later emission
    /// with optimized probability tables.
    token_buffer: Option<residuals::TokenBuffer>,
    /// Stored macroblock info from token recording pass, used to write
    /// headers without redoing mode selection.
    stored_mb_info: Vec<MacroblockInfo>,
    /// Stored quantized coefficients for multi-pass encoding (method >= 5).
    /// Pass 1 stores mode decisions + quantized zigzag coefficients; pass 2+ reuses them.
    stored_mb_coeffs: Vec<QuantizedMbCoeffs>,
}

impl<'a> Vp8Encoder<'a> {
    fn new(writer: &'a mut Vec<u8>) -> Self {
        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: core::array::from_fn(|_| Segment::default()),
            segments_enabled: false,
            segments_update_map: false,
            segment_tree_probs: [255, 255, 255], // Default probs
            segment_map: Vec::new(),

            loop_filter_adjustments: false,
            macroblock_no_skip_coeff: None,
            quantization_indices: QuantizationIndices::default(),

            token_probs: Default::default(),
            proba_stats: ProbaStats::new(),
            updated_probs: None,
            level_costs: LevelCosts::new(),
            // Trellis quantization for RD-optimized coefficient selection.
            // Uses proper probability-dependent init/EOB/skip costs from LevelCosts.
            do_trellis: true,
            // Error diffusion improves quality in smooth gradients
            do_error_diffusion: true,
            // Default to balanced method
            method: 4,
            sns_strength: 50,
            filter_strength: 60,
            filter_sharpness: 0,
            num_segments: 4,
            preset: super::api::Preset::Default,

            top_complexity: Vec::new(),
            left_complexity: Complexity::default(),

            top_b_pred: Vec::new(),
            left_b_pred: [IntraMode::default(); 4],

            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],

            left_border_y: [0u8; 16 + 1],
            left_border_u: [0u8; 8 + 1],
            left_border_v: [0u8; 8 + 1],
            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),

            // Error diffusion starts with zero
            top_derr: Vec::new(),
            left_derr: [[0; 2]; 2],

            token_buffer: None,
            stored_mb_info: Vec::new(),
            stored_mb_coeffs: Vec::new(),
        }
    }

    /// Get the segment for a macroblock at (mbx, mby).
    ///
    /// When segments are enabled, looks up the segment ID from the segment map.
    /// Otherwise, returns segment 0.
    #[inline]
    fn get_segment_for_mb(&self, mbx: usize, mby: usize) -> &Segment {
        let segment_id = if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            self.segment_map[mb_idx] as usize
        } else {
            0
        };
        &self.segments[segment_id]
    }

    /// Get segment ID for a macroblock at (mbx, mby).
    #[inline]
    fn get_segment_id_for_mb(&self, mbx: usize, mby: usize) -> Option<usize> {
        if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            Some(self.segment_map[mb_idx] as usize)
        } else {
            None
        }
    }

    /// Compute updated probabilities from recorded statistics.
    ///
    /// IMPORTANT: For multi-pass encoding, this computes the optimal probabilities
    /// to use for emission. The header encoder compares against COEFF_PROBS (decoder
    /// defaults) to decide which updates to signal.
    fn compute_updated_probabilities(&mut self) {
        // Always start from COEFF_PROBS (decoder defaults) for computing what to update.
        // This ensures the header signaling matches what the decoder expects.
        let mut updated = COEFF_PROBS;

        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for p in 0..11 {
                        // Compare against COEFF_PROBS (decoder defaults), not token_probs
                        let default_prob = COEFF_PROBS[t][b][c][p];
                        let update_prob = COEFF_UPDATE_PROBS[t][b][c][p];

                        let (should_update, new_p, _savings) =
                            self.proba_stats
                                .should_update(t, b, c, p, default_prob, update_prob);

                        // Update if savings are positive, matching libwebp's approach.
                        // The signaling cost (8 bits for value + 1 bit flag) is already
                        // included in the should_update calculation.
                        if should_update {
                            updated[t][b][c][p] = new_p;
                        }
                    }
                }
            }
        }

        // Always set updated_probs with the computed values.
        // For multi-pass, this ensures the header has the final probabilities to signal.
        // If no updates are beneficial, updated will equal COEFF_PROBS.
        self.updated_probs = Some(updated);
    }

    /// Reset encoder state for a new recording pass.
    /// Used by multi-pass token recording (method >= 6) to reset borders,
    /// complexity tracking, and partitions before re-recording.
    fn reset_for_second_pass(&mut self) {
        // Reset complexity tracking
        for complexity in self.top_complexity.iter_mut() {
            *complexity = Complexity::default();
        }
        self.left_complexity = Complexity::default();

        // Reset B-pred tracking
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        // Reset border pixels to initial state
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        for val in self.top_border_y.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_u.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_v.iter_mut() {
            *val = 127;
        }

        // Estimate output size: ~0.3 bytes per pixel is conservative for most quality levels
        let num_pixels =
            usize::from(self.macroblock_width) * usize::from(self.macroblock_height) * 256; // 16x16 per macroblock
        let estimated_partition_size = num_pixels / 4; // ~0.25 bytes per pixel for coefficients

        // Reset partitions with pre-allocated capacity
        self.partitions = vec![ArithmeticEncoder::with_capacity(estimated_partition_size)];

        // Reset encoder (header is small, ~1KB is plenty)
        self.encoder = ArithmeticEncoder::with_capacity(1024);
    }

    fn encode_image(
        &mut self,
        data: &[u8],
        color: ColorType,
        width: u16,
        height: u16,
        params: &super::api::EncoderParams,
    ) -> Result<(), EncodingError> {
        // Store method and configure features based on it
        self.method = params.method.min(6); // Clamp to 0-6
                                            // Trellis quantization only for method >= 4 (like libwebp)
        self.do_trellis = self.method >= 4;
        // Store tuning parameters
        self.sns_strength = params.sns_strength.min(100);
        self.filter_strength = params.filter_strength.min(100);
        self.filter_sharpness = params.filter_sharpness.min(7);
        self.num_segments = params.num_segments.clamp(1, 4);
        self.preset = params.preset;
        let (y_bytes, u_bytes, v_bytes) = match color {
            ColorType::Rgb8 => convert_image_yuv::<3>(data, width, height),
            ColorType::Rgba8 => convert_image_yuv::<4>(data, width, height),
            ColorType::L8 => convert_image_y::<1>(data, width, height),
            ColorType::La8 => convert_image_y::<2>(data, width, height),
        };

        let bytes_per_pixel = match color {
            ColorType::L8 => 1,
            ColorType::La8 => 2,
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
        };
        assert_eq!(
            (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
            data.len() as u64,
            "width/height doesn't match data length of {} for the color type {:?}",
            data.len(),
            color
        );

        self.setup_encoding(
            params.lossy_quality,
            width,
            height,
            y_bytes,
            u_bytes,
            v_bytes,
        );

        // Calculate initial level costs for mode selection and trellis
        if self.level_costs.is_dirty() {
            self.level_costs.calculate(&self.token_probs);
        }

        // Token buffer encoding (matching libwebp's VP8EncTokenLoop).
        //
        // Records coefficient decisions as compact tokens while collecting
        // probability statistics with mid-stream refresh. For method >= 6,
        // performs multiple passes: each pass re-records with probabilities
        // refined from the previous pass, matching libwebp's multi-pass behavior.
        //
        // Benefits over the old two-pass approach:
        // - Statistics reflect actual trellis decisions (no trellis mismatch)
        // - Mid-stream probability refresh improves later coefficient decisions
        // - Multi-pass (method 6) iteratively refines probabilities

        let num_mb = usize::from(self.macroblock_width) * usize::from(self.macroblock_height);

        // Method 6 gets multiple passes for iterative probability refinement.
        // Like libwebp, each pass re-records all macroblocks with improved
        // probabilities from the previous pass's statistics.
        // Method 5-6: 2-pass token recording for iterative probability refinement.
        // Pass 1 records with initial/mid-stream-refreshed probabilities.
        // Pass 2 re-records with fully refined probabilities from pass 1.
        // Third pass was tested but gains <0.05% — not worth the time cost.
        let num_passes = if self.method >= 5 { 2 } else { 1 };

        for pass in 0..num_passes {
            self.token_buffer = Some(residuals::TokenBuffer::with_estimated_capacity(num_mb));
            self.proba_stats.reset();

            // For passes after the first, apply the updated probabilities
            // from the previous pass as the new baseline for cost estimation.
            // IMPORTANT: In pass 2+, we reuse stored mode decisions and coefficients
            // from pass 1 rather than re-doing mode selection. This matches libwebp's
            // behavior where quantization happens once and tokens are re-recorded.
            if pass > 0 {
                if let Some(ref updated) = self.updated_probs {
                    self.token_probs = *updated;
                }
                // Note: level_costs don't need recalculating for pass 1+ since we're
                // not re-quantizing - we're just re-recording the same tokens.
                self.reset_for_second_pass();
            } else {
                // Pass 0: prepare storage for mode decisions and coefficients
                self.stored_mb_info.clear();
                self.stored_mb_info.reserve(num_mb);
                self.stored_mb_coeffs.clear();
                self.stored_mb_coeffs.reserve(num_mb);
            }

            // Mid-stream refresh interval: roughly every total_mb/8 macroblocks
            let max_count = (num_mb / 8).max(96) as i32; // MIN_COUNT = 96 (matches libwebp)
            let mut refresh_countdown = max_count;

            let mut total_mb: u32 = 0;
            let mut skip_mb: u32 = 0;

            // ===== RECORDING PASS =====
            // Pass 0: mode selection + transform + token recording (store coefficients)
            // Pass 1+: re-record tokens with updated probs (reuse stored coefficients)
            for mby in 0..self.macroblock_height {
                // Reset left state for start of row
                self.left_complexity = Complexity::default();
                self.left_b_pred = [IntraMode::default(); 4];

                if pass == 0 {
                    self.left_derr = [[0; 2]; 2]; // reset chroma error diffusion for row start
                    self.left_border_y = [129u8; 16 + 1];
                    self.left_border_u = [129u8; 8 + 1];
                    self.left_border_v = [129u8; 8 + 1];
                }

                for mbx in 0..self.macroblock_width {
                    let mb_idx = (mby as usize) * (self.macroblock_width as usize) + (mbx as usize);

                    if pass == 0 {
                        // Pass 0: Full mode selection + transform + quantize + store coefficients
                        // Mid-stream probability refresh (like libwebp's VP8EncTokenLoop)
                        refresh_countdown -= 1;
                        if refresh_countdown < 0 {
                            self.compute_updated_probabilities();
                            let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
                            self.level_costs.calculate(probs);
                            refresh_countdown = max_count;
                        }

                        let macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                        // Transform blocks (updates border state for next macroblock)
                        let y_block_data =
                            self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                        let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                            mbx.into(),
                            mby.into(),
                            macroblock_info.chroma_mode,
                        );

                        // Check if all coefficients are zero (skip detection)
                        total_mb += 1;
                        let all_zero = self.check_all_coeffs_zero(
                            &macroblock_info,
                            &y_block_data,
                            &u_block_data,
                            &v_block_data,
                        );

                        let mut mb_info = macroblock_info;
                        if all_zero {
                            skip_mb += 1;
                            mb_info.coeffs_skipped = true;
                            // Reset complexity for skipped blocks
                            self.left_complexity
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                            self.top_complexity[usize::from(mbx)]
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                            // Store empty coefficients for skipped blocks
                            self.stored_mb_coeffs.push(QuantizedMbCoeffs {
                                y2_zigzag: [0; 16],
                                y1_zigzag: [[0; 16]; 16],
                                u_zigzag: [[0; 16]; 4],
                                v_zigzag: [[0; 16]; 4],
                            });
                        } else {
                            // Record tokens, quantize, and store quantized coefficients
                            let stored_coeffs = self.record_residual_tokens_storing(
                                &macroblock_info,
                                mbx as usize,
                                &y_block_data,
                                &u_block_data,
                                &v_block_data,
                            );
                            self.stored_mb_coeffs.push(stored_coeffs);
                        }

                        // Store macroblock info for header writing in emit pass
                        self.stored_mb_info.push(mb_info);
                    } else {
                        // Pass 1+: Re-record tokens with updated probability tables.
                        //
                        // NOTE: We tried true re-quantization (re-running trellis with updated
                        // level_costs derived from pass 0 statistics), but it made files LARGER.
                        // The problem is circular: pass 0's trellis decisions are optimal for
                        // COEFF_PROBS, so training costs on those outputs creates overfitting.
                        //
                        // libwebp's multi-pass also doesn't re-quantize - it re-records the same
                        // tokens with better probability tables. The benefit comes from:
                        // 1. More accurate probability signaling in headers (fewer update bits)
                        // 2. Better entropy coding (probabilities match actual distribution)
                        //
                        // Our token buffer with mid-stream probability refresh already provides
                        // most of this benefit. Additional passes mainly help the final probability
                        // table estimation for very large images.
                        let mb_info = self.stored_mb_info[mb_idx];

                        total_mb += 1;
                        if mb_info.coeffs_skipped {
                            skip_mb += 1;
                            // Reset complexity for skipped blocks
                            self.left_complexity.clear(mb_info.luma_mode != LumaMode::B);
                            self.top_complexity[usize::from(mbx)]
                                .clear(mb_info.luma_mode != LumaMode::B);
                        } else {
                            // Re-record tokens from stored quantized coefficients
                            // Clone to avoid borrow conflict
                            let stored_coeffs = self.stored_mb_coeffs[mb_idx].clone();
                            self.record_from_stored_coeffs(
                                &mb_info,
                                mbx as usize,
                                &stored_coeffs,
                            );
                        }
                    }
                }
            }

            // Compute skip probability from actual data
            if total_mb > 0 {
                let non_skip_mb = total_mb - skip_mb;
                let prob = ((255 * non_skip_mb + total_mb / 2) / total_mb).min(255) as u8;
                self.macroblock_no_skip_coeff = Some(prob.clamp(1, 254));
            }

            // Finalize probabilities from this pass
            self.compute_updated_probabilities();

        }

        // ===== FINALIZE: write bitstream =====

        // Write compressed frame header (includes probability updates)
        self.encode_compressed_frame_header();

        // Write macroblock headers from stored info.
        // Take the vec out to avoid borrow conflict with self.write_macroblock_header.
        let stored_mb_info = mem::take(&mut self.stored_mb_info);

        // Reset b-pred tracking state for header writing
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        let mb_w = usize::from(self.macroblock_width);
        for (idx, mb_info) in stored_mb_info.iter().enumerate() {
            let mbx = idx % mb_w;
            if mbx == 0 {
                self.left_b_pred = [IntraMode::default(); 4];
            }
            self.write_macroblock_header(mb_info, mbx);
        }

        // Emit tokens to partition using final probabilities
        let final_probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
        let token_buf = self.token_buffer.take().unwrap();
        token_buf.emit_tokens(&mut self.partitions[0], final_probs);

        // Assemble output
        let compressed_header_encoder = mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32);

        self.writer.write_all(&compressed_header_bytes);

        self.write_partitions();

        // Clean up
        self.stored_mb_info.clear();

        Ok(())
    }

    /// Merge segments with identical quantizer and filter settings.
    ///
    /// This reduces the number of effective segments when different alpha regions
    /// end up with the same quantization and filter parameters. Reducing segment
    /// count can save bits in the bitstream.
    ///
    /// Ported from libwebp's SimplifySegments.
    #[allow(clippy::needless_range_loop)] // s1 indexes both seg_map and self.segments
    fn simplify_segments(&mut self) {
        // Map from old segment ID to new segment ID
        let mut seg_map = [0u8, 1, 2, 3];
        let num_segments = self.num_segments as usize;
        let mut num_final_segments = 1usize;

        // Check each segment starting from 1 to see if it matches an earlier segment
        for s1 in 1..num_segments {
            let seg1 = &self.segments[s1];
            let mut found = false;

            // Check if we already have a segment with same quant_index and loopfilter_level
            for s2 in 0..num_final_segments {
                let seg2 = &self.segments[s2];
                if seg1.quant_index == seg2.quant_index
                    && seg1.loopfilter_level == seg2.loopfilter_level
                {
                    seg_map[s1] = s2 as u8;
                    found = true;
                    break;
                }
            }

            if !found {
                // This is a new unique segment
                seg_map[s1] = num_final_segments as u8;
                if num_final_segments != s1 {
                    // Move segment data to its new position
                    self.segments[num_final_segments] = self.segments[s1].clone();
                }
                num_final_segments += 1;
            }
        }

        // If we reduced segments, remap the segment map
        if num_final_segments < num_segments {
            for seg_id in &mut self.segment_map {
                *seg_id = seg_map[*seg_id as usize];
            }
            self.num_segments = num_final_segments as u8;

            // Replicate trailing segment infos (cosmetic, required by bitstream syntax)
            for i in num_final_segments..num_segments {
                self.segments[i] = self.segments[num_final_segments - 1].clone();
            }
        }
    }

    /// Analyze image complexity and assign macroblocks to segments.
    ///
    /// This performs a DCT-based analysis pass (ported from libwebp) to:
    /// 1. Compute "alpha" (compressibility) for each macroblock using DCT histogram
    /// 2. Build a histogram of alpha values
    /// 3. Use k-means clustering to assign macroblocks to 4 segments
    /// 4. Configure per-segment quantization based on alpha
    ///
    /// Segments allow different quantization for different image regions:
    /// - Flat areas (high alpha from segment perspective) can use more aggressive quantization
    /// - Textured areas (low alpha) need finer quantization to preserve detail
    ///
    /// Ported from libwebp's VP8EncAnalyze / MBAnalyze.
    fn analyze_and_assign_segments(&mut self, base_quant_index: u8) {
        let y_stride = usize::from(self.macroblock_width * 16);
        let uv_stride = usize::from(self.macroblock_width * 8);
        let width = usize::from(self.frame.width);
        let height = usize::from(self.frame.height);

        // Run full DCT-based analysis pass using libwebp-compatible algorithm
        // This tests DC and TM modes for I16 and UV, computes per-MB alpha,
        // and builds the alpha histogram
        let analysis = analyze_image(
            &self.frame.ybuf,
            &self.frame.ubuf,
            &self.frame.vbuf,
            width,
            height,
            y_stride,
            uv_stride,
        );

        // Auto-detect content type when Preset::Auto is selected.
        // Runs after analyze_image (reuses alpha histogram, nearly free).
        if self.preset == super::api::Preset::Auto {
            let content_type = classify_image_type(
                &self.frame.ybuf,
                width,
                height,
                y_stride,
                &analysis.alpha_histogram,
            );
            let (sns, filter, sharp, segs) = content_type_to_tuning(content_type);
            self.sns_strength = sns;
            self.filter_strength = filter;
            self.filter_sharpness = sharp;
            self.num_segments = segs;
        }

        // Use k-means to assign segments
        // weighted_average is computed from final cluster centers, matching libwebp
        let (centers, alpha_to_segment, mid_alpha) =
            assign_segments_kmeans(&analysis.alpha_histogram, usize::from(self.num_segments));

        // Find min and max of centers for alpha transformation
        // This matches libwebp's SetSegmentAlphas
        let min_center = centers.iter().copied().min().unwrap_or(0) as i32;
        let max_center = centers.iter().copied().max().unwrap_or(255) as i32;
        let range = if max_center == min_center {
            1 // Avoid division by zero
        } else {
            max_center - min_center
        };

        // Assign segment IDs to macroblocks
        self.segment_map = analysis
            .mb_alphas
            .iter()
            .map(|&alpha| alpha_to_segment[alpha as usize])
            .collect();

        // Smooth segment map to reduce noisy boundaries
        // Only smooth if we have multiple segments
        if self.num_segments > 1 {
            super::cost::smooth_segment_map(
                &mut self.segment_map,
                usize::from(self.macroblock_width),
                usize::from(self.macroblock_height),
            );
        }

        // Configure per-segment quantization using preset's SNS strength
        let sns_strength = self.sns_strength;

        // Compute UV quant deltas from uv_alpha average (from libwebp's VP8SetSegmentParams)
        // uv_alpha is typically ~30 (bad) to ~100 (ok to decimate UV more), centered ~60
        // Constants from libwebp quant_enc.c
        const MID_UV_ALPHA: i32 = 64;
        const MIN_UV_ALPHA: i32 = 30;
        const MAX_UV_ALPHA: i32 = 100;
        const MAX_DQ_UV: i32 = 6;
        const MIN_DQ_UV: i32 = -4;

        // Map uv_alpha to the safe maximal range of MAX/MIN_DQ_UV
        let dq_uv_ac = (analysis.uv_alpha_avg - MID_UV_ALPHA) * (MAX_DQ_UV - MIN_DQ_UV)
            / (MAX_UV_ALPHA - MIN_UV_ALPHA);
        // Rescale by user-defined SNS strength
        let dq_uv_ac = (dq_uv_ac * i32::from(sns_strength) / 100).clamp(MIN_DQ_UV, MAX_DQ_UV);

        // Boost dc-uv-quant based on sns-strength (UV is more reactive to high quants)
        let dq_uv_dc = (-4 * i32::from(sns_strength) / 100).clamp(-15, 15);

        // Compute base filter level for delta computation (using beta=0 for base)
        let base_filter = super::cost::compute_filter_level(
            base_quant_index,
            self.filter_sharpness,
            self.filter_strength,
        );

        for (seg_idx, &center) in centers.iter().enumerate() {
            let center = center as i32;

            // Transform center to libwebp's alpha scale [-127, 127]
            // Formula from SetSegmentAlphas: alpha = 255 * (center - mid) / (max - min)
            let transformed_alpha = (255 * (center - mid_alpha) / range).clamp(-127, 127);

            // Compute beta for per-segment filter modulation
            // Formula from libwebp: beta = 255 * (center - min) / (max - min)
            // Beta indicates segment complexity: 0 = simplest (closer to min), 255 = most complex
            let beta = (255 * (center - min_center) / range).clamp(0, 255) as u8;

            // Compute adjusted quantizer for this segment
            let seg_quant_index =
                compute_segment_quant(base_quant_index, transformed_alpha, sns_strength);
            let seg_quant_usize = seg_quant_index as usize;

            // Compute the delta from base quantizer
            let delta = seg_quant_index as i8 - base_quant_index as i8;

            // Compute per-segment loop filter with beta modulation
            // Simpler segments (low beta) get less filtering
            let seg_filter = super::cost::compute_filter_level_with_beta(
                seg_quant_index,
                self.filter_sharpness,
                self.filter_strength,
                beta,
            );
            let filter_delta = (seg_filter as i8) - (base_filter as i8);

            // Apply UV quant deltas (from libwebp's SetupMatrices)
            // UV DC quant uses dq_uv_dc offset, clamped to [0, 117]
            // UV AC quant uses dq_uv_ac offset, clamped to [0, 127]
            let uv_dc_idx = (seg_quant_usize as i32 + dq_uv_dc).clamp(0, 117) as usize;
            let uv_ac_idx = (seg_quant_usize as i32 + dq_uv_ac).clamp(0, 127) as usize;

            let mut segment = Segment {
                ydc: DC_QUANT[seg_quant_usize],
                yac: AC_QUANT[seg_quant_usize],
                y2dc: DC_QUANT[seg_quant_usize] * 2,
                y2ac: ((i32::from(AC_QUANT[seg_quant_usize]) * 155 / 100) as i16).max(8),
                uvdc: DC_QUANT[uv_dc_idx],
                uvac: AC_QUANT[uv_ac_idx],
                quantizer_level: delta,
                loopfilter_level: filter_delta,
                quant_index: seg_quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength);
            self.segments[seg_idx] = segment;
        }

        // Simplify segments by merging those with identical quant and filter settings
        // This can reduce the number of effective segments and save bits in the bitstream
        if self.num_segments > 1 {
            self.simplify_segments();
        }

        // Compute segment tree probabilities from actual distribution
        // This matches libwebp's SetSegmentProbas
        let mut seg_counts = [0u32; 4];
        for &seg_id in &self.segment_map {
            seg_counts[seg_id as usize] += 1;
        }

        // Segment tree uses 3 probabilities for binary splits:
        // prob[0] = P(segment < 2), prob[1] = P(segment == 0 | segment < 2)
        // prob[2] = P(segment == 2 | segment >= 2)
        let get_proba = |a: u32, b: u32| -> u8 {
            let total = a + b;
            if total == 0 {
                255 // default
            } else {
                ((255 * a + total / 2) / total) as u8
            }
        };

        self.segment_tree_probs[0] =
            get_proba(seg_counts[0] + seg_counts[1], seg_counts[2] + seg_counts[3]);
        self.segment_tree_probs[1] = get_proba(seg_counts[0], seg_counts[1]);
        self.segment_tree_probs[2] = get_proba(seg_counts[2], seg_counts[3]);

        // Only enable update_map if probabilities differ from default (255)
        let should_update_map = self.segment_tree_probs[0] != 255
            || self.segment_tree_probs[1] != 255
            || self.segment_tree_probs[2] != 255;

        // Enable segment-based encoding
        self.segments_enabled = true;
        self.segments_update_map = should_update_map;

        // Reset borders for actual encoding pass
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];
    }

    // sets up the encoding of the encoder by setting all the encoder params based on the width and height
    fn setup_encoding(
        &mut self,
        lossy_quality: u8,
        width: u16,
        height: u16,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
    ) {
        // choosing the quantization quality based on the quality passed in
        if lossy_quality > 100 {
            panic!("lossy quality must be between 0 and 100");
        }

        // Use libwebp-style quality curve to match expected behavior at Q75
        // This emulates jpeg-like behavior where Q75 is "good quality"
        let quant_index: u8 = quality_to_quant_index(lossy_quality);
        let quant_index_usize: usize = quant_index as usize;

        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        self.macroblock_width = mb_width;
        self.macroblock_height = mb_height;

        // Compute optimal filter level based on quantization and preset tuning params
        let filter_level = super::cost::compute_filter_level(
            quant_index,
            self.filter_sharpness,
            self.filter_strength,
        );

        self.frame = Frame {
            width,
            height,

            ybuf: y_buf,
            ubuf: u_buf,
            vbuf: v_buf,

            version: 0,

            for_display: true,
            pixel_type: 0,

            filter_type: false,
            filter_level,
            sharpness_level: self.filter_sharpness,
        };

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        self.token_probs = COEFF_PROBS;

        // Enable skip mode for zero macroblocks
        // The probability is P(not skip) - 200 means ~78% expected to have coefficients
        self.macroblock_no_skip_coeff = Some(200);

        let quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };
        self.quantization_indices = quantization_indices;

        // Initialize all 4 segments with base quantization first
        // This provides fallback values before segment analysis
        for seg_idx in 0..4 {
            let mut segment = Segment {
                ydc: DC_QUANT[quant_index_usize],
                yac: AC_QUANT[quant_index_usize],
                y2dc: DC_QUANT[quant_index_usize] * 2,
                y2ac: ((i32::from(AC_QUANT[quant_index_usize]) * 155 / 100) as i16).max(8),
                uvdc: DC_QUANT[quant_index_usize],
                uvac: AC_QUANT[quant_index_usize],
                quantizer_level: 0, // No delta for base segment
                quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength);
            self.segments[seg_idx] = segment;
        }

        // Segment-based quantization using DCT histogram analysis (ported from libwebp).
        // This allows different quantization for different image regions:
        // - Flat areas get more aggressive quantization
        // - Textured areas get finer quantization
        //
        // Only enable for images large enough to benefit (overhead vs gain tradeoff).
        // libwebp uses segments for images with method > 0 and multiple segments configured.
        let total_mbs = usize::from(mb_width) * usize::from(mb_height);
        let use_segments = self.num_segments > 1 && total_mbs >= 256;

        if use_segments {
            // DCT-based segment analysis and assignment.
            // For Preset::Auto, this also runs content detection and may override
            // sns_strength, filter_strength, filter_sharpness, and num_segments.
            self.analyze_and_assign_segments(quant_index);

            // If Auto detection changed filter params, recompute frame filter level
            if self.preset == super::api::Preset::Auto {
                let new_filter = super::cost::compute_filter_level(
                    quant_index,
                    self.filter_sharpness,
                    self.filter_strength,
                );
                self.frame.filter_level = new_filter;
                self.frame.sharpness_level = self.filter_sharpness;
            }
        } else {
            // Disable segments for small images (overhead not worth it)
            self.segments_enabled = false;
            self.segments_update_map = false;
            self.segment_map = Vec::new();
        }

        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        self.top_border_y = vec![127u8; usize::from(self.macroblock_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(self.macroblock_width) * 8];
        self.top_border_v = vec![127u8; usize::from(self.macroblock_width) * 8];

        // Initialize error diffusion arrays (one entry per macroblock column)
        // [channel][position], channels are U=0, V=1
        self.top_derr = vec![[[0i8; 2]; 2]; usize::from(self.macroblock_width)];
        self.left_derr = [[0; 2]; 2];
    }
}

pub(crate) fn encode_frame_lossy(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    params: &super::api::EncoderParams,
) -> Result<(), EncodingError> {
    let mut vp8_encoder = Vp8Encoder::new(writer);

    let width = width
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;
    let height = height
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;

    vp8_encoder.encode_image(data, color, width, height, params)?;

    Ok(())
}
