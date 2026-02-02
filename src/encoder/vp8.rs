use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use super::vec_writer::VecWriter;

use super::arithmetic::ArithmeticEncoder;
use super::cost::{
    analyze_image, assign_segments_kmeans, classify_image_type, compute_segment_quant,
    content_type_to_tuning, estimate_dc16_cost, estimate_residual_cost, get_cost_luma4,
    record_coeffs, trellis_quantize_block, LevelCosts, ProbaStats, TokenType, FIXED_COSTS_I16,
    FIXED_COSTS_UV,
};
use crate::common::transform;
use crate::common::types::Frame;
use crate::common::types::*;
// Intra4 imports for coefficient-level cost estimation (used in pick_best_intra4)
use super::cost::get_i4_mode_cost;
// Full RD imports for spectral distortion and flat source detection
use super::api::ColorType;
use super::api::EncodingError;
use super::cost::{
    get_cost_luma16, get_cost_uv, is_flat_coeffs, is_flat_source_16, tdisto_16x16,
    FLATNESS_LIMIT_I16, FLATNESS_LIMIT_UV, FLATNESS_PENALTY, RD_DISTO_MULT, VP8_WEIGHT_Y,
};
use crate::common::prediction::*;
use crate::decoder::yuv::convert_image_y;
use crate::decoder::yuv::convert_image_yuv;

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
fn sse_16x16_luma(
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
fn sse_8x8_chroma(
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

type ChromaCoeffs = [i32; 16 * 4];

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

    /// Writes the uncompressed part of the frame header (9.1)
    fn write_uncompressed_frame_header(&mut self, partition_size: u32) {
        let version = u32::from(self.frame.version);
        let for_display = if self.frame.for_display { 1 } else { 0 };

        let keyframe_bit = 0;
        let tag = (partition_size << 5) | (for_display << 4) | (version << 1) | (keyframe_bit);
        self.writer.write_u24_le(tag);

        let magic_bytes_buffer: [u8; 3] = [0x9d, 0x01, 0x2a];
        self.writer.write_all(&magic_bytes_buffer);

        let width = self.frame.width & 0x3FFF;
        let height = self.frame.height & 0x3FFF;
        self.writer.write_u16_le(width);
        self.writer.write_u16_le(height);
    }

    fn encode_compressed_frame_header(&mut self) {
        // if keyframe, color space must be 0
        self.encoder.write_literal(1, 0);
        // pixel type
        self.encoder.write_literal(1, 0);

        self.encoder.write_flag(self.segments_enabled);
        if self.segments_enabled {
            self.encode_segment_updates();
        }

        self.encoder.write_flag(self.frame.filter_type);
        self.encoder.write_literal(6, self.frame.filter_level);
        self.encoder.write_literal(3, self.frame.sharpness_level);

        self.encoder.write_flag(self.loop_filter_adjustments);
        if self.loop_filter_adjustments {
            self.encode_loop_filter_adjustments();
        }

        // partitions length must be 1, 2, 4 or 8, so value will be 0, 1, 2 or 3
        let partitions_value: u8 = self.partitions.len().ilog2().try_into().unwrap();
        self.encoder.write_literal(2, partitions_value);

        self.encode_quantization_indices();

        // refresh entropy probs
        self.encoder.write_literal(1, 0);

        self.encode_updated_token_probabilities();

        let mb_no_skip_coeff = if self.macroblock_no_skip_coeff.is_some() {
            1
        } else {
            0
        };
        self.encoder.write_literal(1, mb_no_skip_coeff);
        if let Some(prob_skip_false) = self.macroblock_no_skip_coeff {
            self.encoder.write_literal(8, prob_skip_false);
        }
    }

    fn write_partitions(&mut self) {
        let partitions = mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions
            .into_iter()
            .map(|x| x.flush_and_get_buffer())
            .collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer.write_u24_le(partition.len() as u32);
                self.writer.write_all(partition);
            }
        }

        // write the final partition
        self.writer
            .write_all(&partitions_bytes[partitions_bytes.len() - 1]);
    }

    fn encode_segment_updates(&mut self) {
        // Section 9.3 - Segment-based adjustments
        // update_mb_segmentation_map - whether we're updating the map
        self.encoder.write_flag(self.segments_update_map);

        // update_segment_feature_data - whether we're updating segment feature data
        // We always update when segments are enabled to set quantizer deltas
        let update_data = self.segments_enabled;
        self.encoder.write_flag(update_data);

        if update_data {
            // segment_feature_mode: 0 = delta, 1 = absolute
            // We use delta mode (relative to base quantizer)
            self.encoder.write_flag(false); // delta mode

            // Write quantizer deltas for each segment (4 segments)
            for seg in &self.segments {
                let has_delta = seg.quantizer_level != 0;
                self.encoder.write_flag(has_delta);
                if has_delta {
                    // Quantizer delta is signed 7-bit value
                    let abs_val = seg.quantizer_level.unsigned_abs();
                    self.encoder.write_literal(7, abs_val);
                    self.encoder.write_flag(seg.quantizer_level < 0);
                }
            }

            // Write loop filter deltas for each segment
            for seg in &self.segments {
                let has_delta = seg.loopfilter_level != 0;
                self.encoder.write_flag(has_delta);
                if has_delta {
                    // Loop filter delta is signed 6-bit value
                    self.encoder
                        .write_literal(6, seg.loopfilter_level.unsigned_abs());
                    self.encoder.write_flag(seg.loopfilter_level < 0);
                }
            }
        }

        // Write segment ID tree probabilities if updating the map
        if self.segments_update_map {
            // Write the 3 probabilities for the segment ID tree
            for &prob in &self.segment_tree_probs {
                let has_prob = prob != 255; // 255 means no update
                self.encoder.write_flag(has_prob);
                if has_prob {
                    self.encoder.write_literal(8, prob);
                }
            }
        }
    }

    fn encode_loop_filter_adjustments(&mut self) {
        // Whether the deltas are being updated this frame
        self.encoder.write_flag(false);
        // If false, no more data needed - use defaults or previous values
    }

    fn encode_quantization_indices(&mut self) {
        self.encoder
            .write_literal(7, self.quantization_indices.yac_abs);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.ydc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2dc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2ac_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvdc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvac_delta);
    }

    /// Encode token probability updates to the bitstream.
    /// Uses accumulated statistics to decide which probabilities to update.
    fn encode_updated_token_probabilities(&mut self) {
        // Get the updated probabilities if available
        let updated_probs = self.updated_probs.take();

        for (t, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (b, js) in is.iter().enumerate() {
                for (c, ks) in js.iter().enumerate() {
                    for (p, &update_prob) in ks.iter().enumerate() {
                        let old_prob = self.token_probs[t][b][c][p];

                        // Check if we have updated probabilities that differ from the default
                        let (should_update, new_prob) = if let Some(ref probs) = updated_probs {
                            let new_p = probs[t][b][c][p];
                            // Only update if the probability actually changed
                            (new_p != old_prob, new_p)
                        } else {
                            (false, old_prob)
                        };

                        if should_update {
                            // Signal that we're updating this probability
                            self.encoder.write_bool(true, update_prob);
                            // Write the new probability value (8 bits)
                            self.encoder.write_literal(8, new_prob);
                            // Update our local copy for future encoding
                            self.token_probs[t][b][c][p] = new_prob;
                        } else {
                            // Signal no update
                            self.encoder.write_bool(false, update_prob);
                        }
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        if self.segments_enabled && self.segments_update_map {
            // Write segment ID using the segment tree
            let segment_id = macroblock_info.segment_id.unwrap_or(0) as i8;
            self.encoder
                .write_with_tree(&SEGMENT_ID_TREE, &self.segment_tree_probs, segment_id);
        }

        if let Some(prob) = self.macroblock_no_skip_coeff {
            self.encoder
                .write_bool(macroblock_info.coeffs_skipped, prob);
        }

        // encode macroblock info y mode using KEYFRAME_YMODE_TREE
        self.encoder.write_with_tree(
            &KEYFRAME_YMODE_TREE,
            &KEYFRAME_YMODE_PROBS,
            macroblock_info.luma_mode as i8,
        );

        match macroblock_info.luma_mode.into_intra() {
            None => {
                // 11.3 code each of the subblocks
                if let Some(bpred) = macroblock_info.luma_bpred {
                    for y in 0usize..4 {
                        let mut left = self.left_b_pred[y];
                        for x in 0usize..4 {
                            let top = self.top_b_pred[mbx * 4 + x];
                            let probs = &KEYFRAME_BPRED_MODE_PROBS[top as usize][left as usize];
                            let intra_mode = bpred[y * 4 + x];
                            self.encoder.write_with_tree(
                                &KEYFRAME_BPRED_MODE_TREE,
                                probs,
                                intra_mode as i8,
                            );
                            left = intra_mode;
                            self.top_b_pred[mbx * 4 + x] = intra_mode;
                        }
                        self.left_b_pred[y] = left;
                    }
                } else {
                    panic!("Invalid, can't set luma mode to B without setting preds");
                }
            }
            Some(intra_mode) => {
                for (left, top) in self
                    .left_b_pred
                    .iter_mut()
                    .zip(self.top_b_pred[4 * mbx..][..4].iter_mut())
                {
                    *left = intra_mode;
                    *top = intra_mode;
                }
            }
        }

        // encode macroblock info chroma mode
        self.encoder.write_with_tree(
            &KEYFRAME_UV_MODE_TREE,
            &KEYFRAME_UV_MODE_PROBS,
            macroblock_info.chroma_mode as i8,
        );
    }

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
    fn apply_chroma_error_diffusion(
        &mut self,
        u_blocks: &mut [i32; 16 * 4],
        v_blocks: &mut [i32; 16 * 4],
        mbx: usize,
        uv_matrix: &super::cost::VP8Matrix,
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
    fn encode_residual_data(
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
    fn encode_coefficients(
        &mut self,
        block: &[i32; 16],
        partition_index: usize,
        plane: Plane,
        complexity: usize,
        matrix: &super::cost::VP8Matrix,
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
    fn check_all_coeffs_zero(
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
    fn record_residual_stats(
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

    /// Compute updated probabilities from recorded statistics.
    /// Only includes updates where the savings exceed a minimum threshold.
    fn compute_updated_probabilities(&mut self) {
        let mut updated = self.token_probs;
        let mut total_savings = 0i32;
        let mut num_updates = 0u32;

        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for p in 0..11 {
                        let old_prob = self.token_probs[t][b][c][p];
                        let update_prob = COEFF_UPDATE_PROBS[t][b][c][p];

                        let (should_update, new_p, savings) =
                            self.proba_stats
                                .should_update(t, b, c, p, old_prob, update_prob);

                        // Update if savings are positive, matching libwebp's approach.
                        // The signaling cost (8 bits for value + 1 bit flag) is already
                        // included in the should_update calculation.
                        if should_update && savings > 0 {
                            updated[t][b][c][p] = new_p;
                            total_savings += savings;
                            num_updates += 1;
                        }
                    }
                }
            }
        }

        // Only use updated probabilities if we have net positive savings
        if total_savings > 0 && num_updates > 0 {
            self.updated_probs = Some(updated);
        } else {
            // Keep default probabilities - no updates
            self.updated_probs = None;
        }
    }

    /// Reset encoder state for second pass encoding.
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

        // Two-pass encoding with adaptive probabilities.
        // Pass 1: Collect token statistics (without trellis) to estimate probabilities.
        // Pass 2: Encode with updated probabilities (with trellis) for better compression.
        // This is essential for good compression, especially at high quality where
        // coefficients have more varied magnitudes and accurate probability estimates
        // are crucial for efficient entropy coding.
        let use_two_pass = true;

        if use_two_pass {
            // ===== PASS 1: Collect token statistics for adaptive probabilities =====
            // Disable trellis during pass 1 to ensure consistent statistics collection.
            // The statistics should reflect simple quantization, then trellis optimizes
            // the actual encoding in pass 2 using those probabilities.
            let trellis_during_pass2 = self.do_trellis;
            self.do_trellis = false;

            self.proba_stats.reset();
            let mut total_mb: u32 = 0;
            let mut skip_mb: u32 = 0;

            for mby in 0..self.macroblock_height {
                // reset left complexity / bpreds for left of image
                self.left_complexity = Complexity::default();
                self.left_b_pred = [IntraMode::default(); 4];

                self.left_border_y = [129u8; 16 + 1];
                self.left_border_u = [129u8; 8 + 1];
                self.left_border_v = [129u8; 8 + 1];

                for mbx in 0..self.macroblock_width {
                    let macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                    // Transform blocks (updates border state for next macroblock)
                    let y_block_data =
                        self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                    let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                        mbx.into(),
                        mby.into(),
                        macroblock_info.chroma_mode,
                    );

                    // Track skip statistics
                    total_mb += 1;
                    let all_zero = self.check_all_coeffs_zero(
                        &macroblock_info,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                    if all_zero {
                        skip_mb += 1;
                        // Reset complexity for skipped blocks
                        self.left_complexity
                            .clear(macroblock_info.luma_mode != LumaMode::B);
                        self.top_complexity[usize::from(mbx)]
                            .clear(macroblock_info.luma_mode != LumaMode::B);
                    } else {
                        // Record token statistics for this macroblock
                        self.record_residual_stats(
                            &macroblock_info,
                            mbx as usize,
                            &y_block_data,
                            &u_block_data,
                            &v_block_data,
                        );
                    }
                }
            }

            // Compute skip probability from actual data
            // prob_skip_false = P(macroblock has non-zero coefficients)
            if total_mb > 0 {
                let non_skip_mb = total_mb - skip_mb;
                // Round to nearest: (255 * non_skip + total/2) / total
                let prob = ((255 * non_skip_mb + total_mb / 2) / total_mb).min(255) as u8;
                // Clamp to valid range [1, 254] - 0 means always skip, 255 means never skip
                self.macroblock_no_skip_coeff = Some(prob.clamp(1, 254));
            }

            // Compute optimal probabilities from statistics
            self.compute_updated_probabilities();

            // Calculate level costs based on final probabilities
            // This enables accurate mode cost estimation and trellis decisions in pass 2
            let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
            self.level_costs.calculate(probs);

            // Restore trellis setting for pass 2
            self.do_trellis = trellis_during_pass2;

            // Reset state for actual encoding pass
            self.reset_for_second_pass();
        }

        // Calculate level costs for initial probabilities if no two-pass
        if self.level_costs.is_dirty() {
            self.level_costs.calculate(&self.token_probs);
        }

        self.encode_compressed_frame_header();

        // encode residual partitions
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            // reset left complexity / bpreds for left of image
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];
            self.left_derr = [[0; 2]; 2]; // reset chroma error diffusion for row start

            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let mut macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                // Transform first to check for skip
                let y_block_data =
                    self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                    mbx.into(),
                    mby.into(),
                    macroblock_info.chroma_mode,
                );

                // Check if all coefficients are zero (can skip)
                if self.macroblock_no_skip_coeff.is_some() {
                    let all_zero = self.check_all_coeffs_zero(
                        &macroblock_info,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                    macroblock_info.coeffs_skipped = all_zero;
                }

                // write macroblock headers (now with correct skip flag)
                self.write_macroblock_header(&macroblock_info, mbx.into());

                if !macroblock_info.coeffs_skipped {
                    self.encode_residual_data(
                        &macroblock_info,
                        partition_index,
                        mbx as usize,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                } else {
                    // since coeffs are all zero, need to set all complexities to 0
                    // except if the luma mode is B then won't set Y2
                    self.left_complexity
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                    self.top_complexity[usize::from(mbx)]
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                }
            }
        }

        let compressed_header_encoder = mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32);

        self.writer.write_all(&compressed_header_bytes);

        self.write_partitions();

        Ok(())
    }

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

                // Get mode context from neighboring blocks (DC=0 if at edge)
                let top_ctx = if sby == 0 {
                    0 // DC mode
                } else {
                    best_mode_indices[(sby - 1) * 4 + sbx]
                };
                let left_ctx = if sbx == 0 {
                    0 // DC mode
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
                // Track best block's SSE and rate for recalculating with lambda_mode
                let mut best_sse = 0u32;
                let mut best_rate = 0u32;

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
                // - method 2-3: 3 modes (fast)
                // - method 4: 4 modes (balanced)
                // - method 5-6: 10 modes (full search)
                let max_modes_to_try = match self.method {
                    0..=3 => 3,
                    4 => 4,
                    _ => 10, // method 5-6: try all modes
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

                    // Quantize
                    let mut quantized = [0i32; 16];
                    for (idx, &val) in residual.iter().enumerate() {
                        quantized[idx] = y1_matrix.quantize_coeff(val, idx);
                    }

                    // Get accurate coefficient cost using probability tables
                    let (coeff_cost, has_nz) =
                        get_cost_luma4(&quantized, nz_top, nz_left, &self.level_costs, probs);

                    // Compute distortion (SSE of quantized vs original)
                    let mut dequantized = quantized;
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
                    let mode_cost = get_i4_mode_cost(top_ctx, left_ctx, mode_idx);
                    let total_rate = u32::from(mode_cost) + coeff_cost;
                    let rd_score = super::cost::rd_score(sse, total_rate as u16, lambda_i4);

                    if rd_score < best_block_score {
                        best_block_score = rd_score;
                        best_mode = mode;
                        best_mode_idx = mode_idx;
                        best_has_nz = has_nz;
                        best_quantized = quantized;
                        best_sse = sse;
                        best_rate = total_rate;
                    }
                }

                best_modes[i] = best_mode;
                best_mode_indices[i] = best_mode_idx;

                // Update non-zero context for subsequent blocks
                top_nz[sbx] = best_has_nz;
                left_nz[sby] = best_has_nz;

                let mode_cost = get_i4_mode_cost(top_ctx, left_ctx, best_mode_idx);
                total_mode_cost += u32::from(mode_cost);

                // Recalculate the block score with lambda_mode for accumulation
                // (matching libwebp's SetRDScore(lambda_mode, &rd_i4) before AddScore)
                let block_score_for_comparison =
                    super::cost::rd_score(best_sse, best_rate as u16, lambda_mode);

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

    fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        // Pick the best 16x16 luma mode using RD cost selection
        let (luma_mode, i16_score) = self.pick_best_intra16(mbx, mby);

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
                    Some((modes, _)) => (LumaMode::B, Some(modes)),
                    None => (luma_mode, None),
                }
            } else {
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
        let (mb_alphas, alpha_histogram) = analyze_image(
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
            let content_type =
                classify_image_type(&self.frame.ybuf, width, height, y_stride, &alpha_histogram);
            let (sns, filter, sharp, segs) = content_type_to_tuning(content_type);
            self.sns_strength = sns;
            self.filter_strength = filter;
            self.filter_sharpness = sharp;
            self.num_segments = segs;
        }

        // Use k-means to assign segments
        // weighted_average is computed from final cluster centers, matching libwebp
        let (centers, alpha_to_segment, mid_alpha) =
            assign_segments_kmeans(&alpha_histogram, usize::from(self.num_segments));

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
        self.segment_map = mb_alphas
            .iter()
            .map(|&alpha| alpha_to_segment[alpha as usize])
            .collect();

        // Configure per-segment quantization using preset's SNS strength
        let sns_strength = self.sns_strength;

        // Compute base filter level for delta computation
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

            // Compute adjusted quantizer for this segment
            let seg_quant_index =
                compute_segment_quant(base_quant_index, transformed_alpha, sns_strength);
            let seg_quant_usize = seg_quant_index as usize;

            // Compute the delta from base quantizer
            let delta = seg_quant_index as i8 - base_quant_index as i8;

            // Compute per-segment loop filter delta
            let seg_filter = super::cost::compute_filter_level(
                seg_quant_index,
                self.filter_sharpness,
                self.filter_strength,
            );
            let filter_delta = (seg_filter as i8) - (base_filter as i8);

            let mut segment = Segment {
                ydc: DC_QUANT[seg_quant_usize],
                yac: AC_QUANT[seg_quant_usize],
                y2dc: DC_QUANT[seg_quant_usize] * 2,
                y2ac: ((i32::from(AC_QUANT[seg_quant_usize]) * 155 / 100) as i16).max(8),
                uvdc: DC_QUANT[seg_quant_usize],
                uvac: AC_QUANT[seg_quant_usize],
                quantizer_level: delta,
                loopfilter_level: filter_delta,
                quant_index: seg_quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength);
            self.segments[seg_idx] = segment;
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

    // this is for all the luma modes except B
    fn get_predicted_luma_block_16x16(
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
    fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        self.get_luma_blocks_from_predicted_16x16_scalar(predicted_y_block, mbx, mby)
    }

    // SIMD version: uses fused residual+DCT for pairs of blocks
    #[cfg(feature = "simd")]
    fn get_luma_blocks_from_predicted_16x16(
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
    fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> [i32; 16 * 16] {
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

        luma_blocks
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    #[allow(clippy::needless_range_loop)] // sbx,sby indices used for multiple arrays and coordinate computation
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
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

        luma_blocks
    }

    fn get_predicted_chroma_block(
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

    fn get_chroma_blocks_from_predicted(
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

    fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        chroma_mode: ChromaMode,
    ) -> ([i32; 16 * 4], [i32; 16 * 4]) {
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

        (u_blocks, v_blocks)
    }
}

fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
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
