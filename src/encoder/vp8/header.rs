//! VP8 bitstream header encoding.
//!
//! Contains methods for writing the uncompressed frame header, compressed header,
//! partition layout, segment updates, and macroblock headers.

use alloc::vec::Vec;
use core::mem;

use crate::common::types::*;
use crate::encoder::vec_writer::VecWriter;

use super::MacroblockInfo;

impl<'a> super::Vp8Encoder<'a> {
    pub(super) fn write_uncompressed_frame_header(&mut self, partition_size: u32) {
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

    pub(super) fn encode_compressed_frame_header(&mut self) {
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

    pub(super) fn write_partitions(&mut self) {
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
    ///
    /// IMPORTANT: Compare against COEFF_PROBS (decoder defaults), not token_probs.
    /// For multi-pass encoding, token_probs may hold intermediate values from
    /// previous passes, but the decoder always starts from COEFF_PROBS.
    fn encode_updated_token_probabilities(&mut self) {
        // Get the updated probabilities if available
        let updated_probs = self.updated_probs.take();

        for (t, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (b, js) in is.iter().enumerate() {
                for (c, ks) in js.iter().enumerate() {
                    for (p, &update_prob) in ks.iter().enumerate() {
                        // IMPORTANT: Compare against COEFF_PROBS (decoder's initial state),
                        // not token_probs (which may have been modified by previous passes).
                        let default_prob = COEFF_PROBS[t][b][c][p];

                        // Check if we have updated probabilities that differ from the default
                        let (should_update, new_prob) = if let Some(ref probs) = updated_probs {
                            let new_p = probs[t][b][c][p];
                            // Only update if the probability differs from decoder's default
                            (new_p != default_prob, new_p)
                        } else {
                            (false, default_prob)
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
                            // Ensure token_probs matches what decoder will have
                            self.token_probs[t][b][c][p] = default_prob;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
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
}
