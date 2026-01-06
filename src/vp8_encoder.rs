use std::io::Write;

use byteorder_lite::{LittleEndian, WriteBytesExt};

use crate::transform;
use crate::vp8::Frame;
use crate::vp8_arithmetic_encoder::ArithmeticEncoder;
use crate::vp8_common::*;
use crate::vp8_cost::{
    self, analyze_image, assign_segments_kmeans, compute_segment_quant, estimate_dc16_cost,
    estimate_residual_cost, get_cost_luma4, record_coeffs, LevelCosts, ProbaStats, TokenType,
    FIXED_COSTS_I16, FIXED_COSTS_UV, LAMBDA_I16, LAMBDA_UV,
};
// Intra4 imports for coefficient-level cost estimation (used in pick_best_intra4)
use crate::vp8_cost::{calc_i4_penalty, get_i4_mode_cost, LAMBDA_I4};
use crate::vp8_prediction::*;
use crate::yuv::convert_image_y;
use crate::yuv::convert_image_yuv;
use crate::ColorType;
use crate::EncodingError;

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
    linear_c.powf(1.0 / 3.0)
}

/// Convert user-facing quality (0-100) to internal quant index (0-127)
/// Ported from libwebp's VP8SetSegmentParams().
fn quality_to_quant_index(quality: u8) -> u8 {
    let c = quality_to_compression(quality);
    let q = (127.0 * (1.0 - c)).round() as i32;
    q.clamp(0, 127) as u8
}

//------------------------------------------------------------------------------
// SSE (Sum of Squared Errors) distortion functions
//
// These measure the distortion between source and predicted blocks.
// Lower SSE = better prediction = less data to encode.

/// Compute SSE for a 16x16 luma block within bordered prediction buffer
/// Compares source YUV data against predicted block with border
fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
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

/// Compute SSE for an 8x8 chroma block within bordered prediction buffer
fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
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

struct Luma16x16Coeffs {
    y2_coeffs: [i32; 16],
    y_coeffs: LumaYCoeffs,
}

type LumaYCoeffs = [i32; 16 * 16];

type ChromaCoeffs = [i32; 16 * 4];

struct Vp8Encoder<W> {
    writer: W,
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
}

impl<W: Write> Vp8Encoder<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: std::array::from_fn(|_| Segment::default()),
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
    fn write_uncompressed_frame_header(
        &mut self,
        partition_size: u32,
    ) -> Result<(), EncodingError> {
        let version = u32::from(self.frame.version);
        let for_display = if self.frame.for_display { 1 } else { 0 };

        let keyframe_bit = 0;
        let tag = (partition_size << 5) | (for_display << 4) | (version << 1) | (keyframe_bit);
        self.writer.write_u24::<LittleEndian>(tag)?;

        let magic_bytes_buffer: [u8; 3] = [0x9d, 0x01, 0x2a];
        self.writer.write_all(&magic_bytes_buffer)?;

        let width = self.frame.width & 0x3FFF;
        let height = self.frame.height & 0x3FFF;
        self.writer.write_u16::<LittleEndian>(width)?;
        self.writer.write_u16::<LittleEndian>(height)?;

        Ok(())
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

    fn write_partitions(&mut self) -> Result<(), EncodingError> {
        let partitions = std::mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions
            .into_iter()
            .map(|x| x.flush_and_get_buffer())
            .collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer
                    .write_u24::<LittleEndian>(partition.len() as u32)?;
                self.writer.write_all(partition)?;
            }
        }

        // write the final partition
        self.writer
            .write_all(&partitions_bytes[partitions_bytes.len() - 1])?;

        Ok(())
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

            // Write loop filter deltas for each segment (always 0 for now)
            for _ in 0..4 {
                self.encoder.write_flag(false); // no loop filter delta
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

        // Extract VP8 matrices upfront to avoid borrow conflicts
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();

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
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            // set for the next macroblock
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

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
        matrix: &vp8_cost::VP8Matrix,
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
        for i in first_coeff..16 {
            let zigzag_index = usize::from(ZIGZAG[i]);
            zigzag_block[i] = matrix.quantize_coeff(block[zigzag_index], zigzag_index);
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

        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block: &[i32; 16] = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                // Quantize to zigzag order
                let mut zigzag = [0i32; 16];
                for i in first_coeff..16 {
                    let zigzag_index = usize::from(ZIGZAG[i]);
                    zigzag[i] = y1_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                }

                let top = self.top_complexity[mbx].y[x];
                let complexity = (left + top).min(2);

                record_coeffs(
                    &zigzag,
                    token_type,
                    first_coeff,
                    complexity as usize,
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

        // Reset partitions
        self.partitions = vec![ArithmeticEncoder::new()];

        // Reset encoder
        self.encoder = ArithmeticEncoder::new();
    }

    fn encode_image(
        &mut self,
        data: &[u8],
        color: ColorType,
        width: u16,
        height: u16,
        lossy_quality: u8,
    ) -> Result<(), EncodingError> {
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

        self.setup_encoding(lossy_quality, width, height, y_bytes, u_bytes, v_bytes);

        // Two-pass encoding for adaptive probabilities is only beneficial at
        // lower quality settings where there are fewer coefficients and the
        // probability updates can provide meaningful savings.
        // At high quality (Q >= 85), the overhead of signaling updates exceeds savings.
        let use_two_pass = lossy_quality < 85;

        if use_two_pass {
            // ===== PASS 1: Collect token statistics for adaptive probabilities =====
            self.proba_stats.reset();

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

                    // Record token statistics for this macroblock
                    if !self.check_all_coeffs_zero(
                        &macroblock_info,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    ) {
                        self.record_residual_stats(
                            &macroblock_info,
                            mbx as usize,
                            &y_block_data,
                            &u_block_data,
                            &v_block_data,
                        );
                    } else {
                        // Reset complexity for skipped blocks
                        self.left_complexity
                            .clear(macroblock_info.luma_mode != LumaMode::B);
                        self.top_complexity[usize::from(mbx)]
                            .clear(macroblock_info.luma_mode != LumaMode::B);
                    }
                }
            }

            // Compute optimal probabilities from statistics
            self.compute_updated_probabilities();

            // Calculate level costs based on final probabilities
            // This enables accurate I4 coefficient cost estimation
            let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
            self.level_costs.calculate(probs);

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

        let compressed_header_encoder = std::mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32)?;

        self.writer.write_all(&compressed_header_bytes)?;

        self.write_partitions()?;

        Ok(())
    }

    /// Select the best 16x16 luma prediction mode using full RD (rate-distortion) cost.
    ///
    /// Tries all 4 modes (DC, V, H, TM) and picks the one with lowest RD cost:
    ///   RD_cost = SSE * 256 + (mode_cost + coeff_cost) * lambda
    ///
    /// This balances distortion against mode cost for comparison with Intra4.
    ///
    /// Uses libwebp's "RefineUsingDistortion" approach for fair I16 vs I4 comparison:
    /// score = SSE * RD_DISTO_MULT + mode_cost * LAMBDA_I16 (106)
    ///
    /// Returns (best_mode, distortion_score) for comparison against Intra4x4.
    fn pick_best_intra16(&self, mbx: usize, mby: usize) -> (LumaMode, u64) {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // The 4 modes to try for 16x16 luma prediction (order matches FIXED_COSTS_I16)
        const MODES: [LumaMode; 4] = [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM];

        let mut best_mode = LumaMode::DC;
        let mut best_rd_score = u64::MAX;

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

            // Compute SSE between source and prediction
            let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &pred);

            // Compute distortion-only RD score (matching libwebp's RefineUsingDistortion)
            // LAMBDA_I16 = 106 for I16 mode costs
            let mode_cost = FIXED_COSTS_I16[mode_idx];
            let rd_score = vp8_cost::rd_score(sse, mode_cost, LAMBDA_I16);

            if rd_score < best_rd_score {
                best_rd_score = rd_score;
                best_mode = mode;
            }
        }

        (best_mode, best_rd_score)
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

        // Get quant index for i4_penalty calculation
        let quant_index = segment.quant_index as u32;

        // Start with i4_penalty to account for typically higher I4 mode signaling cost.
        // This matches libwebp: i4_penalty = 1000 * q * q
        let mut running_score = calc_i4_penalty(quant_index);

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

                // Try each mode and pick the best
                for (mode_idx, &mode) in MODES.iter().enumerate() {
                    // Make a copy to test this mode
                    let mut test_buf = y_with_border;
                    Self::apply_intra4_prediction(&mut test_buf, mode, x0, y0);

                    // Compute residual
                    let mut residual = [0i32; 16];
                    let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;
                    for y in 0..4 {
                        let pred_row = (y0 + y) * LUMA_STRIDE + x0;
                        let src_row = src_base + y * src_width;
                        for x in 0..4 {
                            residual[y * 4 + x] = i32::from(self.frame.ybuf[src_row + x])
                                - i32::from(test_buf[pred_row + x]);
                        }
                    }

                    // Transform
                    transform::dct4x4(&mut residual);

                    // Quantize
                    let y1_matrix = segment.y1_matrix.as_ref().unwrap();
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

                    let mut sse = 0u32;
                    for y in 0..4 {
                        let pred_row = (y0 + y) * LUMA_STRIDE + x0;
                        let src_row = src_base + y * src_width;
                        for x in 0..4 {
                            let reconstructed = (i32::from(test_buf[pred_row + x])
                                + dequantized[y * 4 + x])
                            .clamp(0, 255) as u8;
                            let diff =
                                i32::from(self.frame.ybuf[src_row + x]) - i32::from(reconstructed);
                            sse += (diff * diff) as u32;
                        }
                    }

                    // Compute RD score: distortion + lambda * (mode_cost + coeff_cost)
                    let mode_cost = get_i4_mode_cost(top_ctx, left_ctx, mode_idx);
                    let total_rate = u32::from(mode_cost) + coeff_cost;
                    let rd_score = vp8_cost::rd_score(sse, total_rate as u16, LAMBDA_I4);

                    if rd_score < best_block_score {
                        best_block_score = rd_score;
                        best_mode = mode;
                        best_mode_idx = mode_idx;
                        best_has_nz = has_nz;
                        best_quantized = quantized;
                    }
                }

                best_modes[i] = best_mode;
                best_mode_indices[i] = best_mode_idx;

                // Update non-zero context for subsequent blocks
                top_nz[sbx] = best_has_nz;
                left_nz[sby] = best_has_nz;

                let mode_cost = get_i4_mode_cost(top_ctx, left_ctx, best_mode_idx);
                total_mode_cost += u32::from(mode_cost);

                // Add this block's score to running total
                running_score += best_block_score;

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

    /// Select the best chroma prediction mode using RD (rate-distortion) cost.
    ///
    /// Tries all 4 modes and picks the one with lowest RD cost for U+V combined.
    fn pick_best_uv(&self, mbx: usize, mby: usize) -> ChromaMode {
        let mbw = usize::from(self.macroblock_width);
        let chroma_width = mbw * 8;

        // Order matches FIXED_COSTS_UV
        const MODES: [ChromaMode; 4] =
            [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

        let mut best_mode = ChromaMode::DC;
        let mut best_rd_score = u64::MAX;

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

            // Compute combined SSE for U and V
            let sse_u = sse_8x8_chroma(&self.frame.ubuf, chroma_width, mbx, mby, &pred_u);
            let sse_v = sse_8x8_chroma(&self.frame.vbuf, chroma_width, mbx, mby, &pred_v);
            let sse = sse_u + sse_v;

            // Compute RD cost
            let mode_cost = FIXED_COSTS_UV[mode_idx];
            let rd_score = vp8_cost::rd_score(sse, mode_cost, LAMBDA_UV);

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

        // Try Intra4 mode and compare with Intra16 using accurate coefficient costs
        let (luma_mode, luma_bpred) = match self.pick_best_intra4(mbx, mby, i16_score) {
            Some((modes, _)) => (LumaMode::B, Some(modes)),
            None => (luma_mode, None),
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

        // Use k-means to assign segments (4 segments)
        // weighted_average is computed from final cluster centers, matching libwebp
        let (centers, alpha_to_segment, mid_alpha) = assign_segments_kmeans(&alpha_histogram, 4);

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

        // Configure per-segment quantization
        // SNS strength of 50 = moderate segment differentiation
        let sns_strength: u8 = 50;

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

            let mut segment = Segment {
                ydc: DC_QUANT[seg_quant_usize],
                yac: AC_QUANT[seg_quant_usize],
                y2dc: DC_QUANT[seg_quant_usize] * 2,
                y2ac: ((i32::from(AC_QUANT[seg_quant_usize]) * 155 / 100) as i16).max(8),
                uvdc: DC_QUANT[seg_quant_usize],
                uvac: AC_QUANT[seg_quant_usize],
                quantizer_level: delta,
                quant_index: seg_quant_index,
                ..Default::default()
            };
            segment.init_matrices();
            self.segments[seg_idx] = segment;
        }

        // Enable segment-based encoding
        self.segments_enabled = true;
        self.segments_update_map = true;

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

        // Compute optimal filter level based on quantization
        // Use sharpness=0 (no sharpening reduction) and default filter_strength=50
        let sharpness: u8 = 0;
        let filter_strength: u8 = 50;
        let filter_level = vp8_cost::compute_filter_level(quant_index, sharpness, filter_strength);

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
            sharpness_level: sharpness, // Matches sharpness used in filter level calculation
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
            segment.init_matrices();
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
        let use_segments = total_mbs >= 256; // Enable for images >= 256 macroblocks (~256x256)

        if use_segments {
            // DCT-based segment analysis and assignment
            self.analyze_and_assign_segments(quant_index);
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
    fn get_luma_blocks_from_predicted_16x16(
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

    // converts the predicted y block to the coeffs
    // Note: Quantization here must match encode_coefficients for correct reconstruction
    fn get_luma_block_coeffs_16x16(
        &self,
        mut luma_blocks: [i32; 16 * 16],
        segment: &Segment,
    ) -> Luma16x16Coeffs {
        let mut coeffs0 = get_coeffs0_from_block(&luma_blocks);
        // wht transform the y2 block and quantize it
        transform::wht4x4(&mut coeffs0);

        // Use VP8Matrix biased quantization (must match encode_coefficients)
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        for (index, value) in coeffs0.iter_mut().enumerate() {
            *value = y2_matrix.quantize_coeff(*value, index);
        }

        // quantize the y blocks - DC goes to Y2, only quantize AC
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        for y_block in luma_blocks.chunks_exact_mut(16) {
            for (index, y_value) in y_block.iter_mut().enumerate() {
                if index == 0 {
                    *y_value = 0;
                } else {
                    *y_value = y1_matrix.quantize_coeff(*y_value, index);
                }
            }
        }

        Luma16x16Coeffs {
            y2_coeffs: coeffs0,
            y_coeffs: luma_blocks,
        }
    }

    fn get_dequantized_blocks_from_coeffs_luma_16x16(
        &self,
        coeffs: &mut Luma16x16Coeffs,
        segment: &Segment,
    ) -> [i32; 16 * 16] {
        let mut dequantized_luma_residue = [0i32; 16 * 16];
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();

        // Dequantize Y2 block using VP8Matrix
        for (k, y2_coeff) in coeffs.y2_coeffs.iter_mut().enumerate() {
            *y2_coeff = y2_matrix.dequantize(*y2_coeff, k);
        }
        transform::iwht4x4(&mut coeffs.y2_coeffs);

        // de-quantize the y blocks as well as do the inverse transform
        for (k, luma_block) in coeffs.y_coeffs.chunks_exact_mut(16).enumerate() {
            for (idx, y_value) in luma_block[1..].iter_mut().enumerate() {
                *y_value = y1_matrix.dequantize(*y_value, idx + 1);
            }

            luma_block[0] = coeffs.y2_coeffs[k];

            transform::idct4x4(luma_block);

            dequantized_luma_residue[k * 16..][..16].copy_from_slice(luma_block);
        }

        dequantized_luma_residue
    }

    // Transforms the luma macroblock in the following ways
    // 1. Does the luma prediction and subtracts from the block
    // 2. Converts the block so each 4x4 subblock is contiguous within the block
    // 3. Does the DCT on each subblock
    // 4. Quantizes the block and dequantizes each subblock
    // 5. Calculates the quantized block - this can be used to calculate how accurate the
    // result is and is used to populate the borders for the next macroblock
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

        // get coeffs
        let mut coeffs = self.get_luma_block_coeffs_16x16(luma_blocks, &segment);

        // now we're essentially applying the same functions as the decoder in order to ensure
        // that the border is the same as the one used for the decoder in the same macroblock
        let dequantized_blocks =
            self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs, segment);

        // re-use the y_with_border from earlier since the prediction is still valid
        // applies the same thing as the decoder so that the border will line up
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = x + y * 4;
                // Create a reference to a [i32; 16] array for add_residue (slices of size 16 do not work).
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
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = 1usize + 16 + 4;
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

                // quantize and de-quantize the subblock using VP8Matrix
                let y1_matrix = segment.y1_matrix.as_ref().unwrap();
                for (index, y_value) in current_subblock.iter_mut().enumerate() {
                    let level = y1_matrix.quantize_coeff(*y_value, index);
                    *y_value = y1_matrix.dequantize(level, index);
                }
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

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

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

pub(crate) fn encode_frame_lossy<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    lossy_quality: u8,
) -> Result<(), EncodingError> {
    let mut vp8_encoder = Vp8Encoder::new(writer);

    let width = width
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;
    let height = height
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;

    vp8_encoder.encode_image(data, color, width, height, lossy_quality)?;

    Ok(())
}
