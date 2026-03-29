//! Frame header parsing for the v2 VP8 decoder.
//!
//! Reads the VP8 keyframe header from the bitstream and populates
//! `FrameTables` and `DecoderContext` buffers. Reuses the existing
//! `VP8HeaderBitReader` from `bit_reader.rs`.

use alloc::vec;
use alloc::vec::Vec;

use super::context::DecoderContext;
use super::tables::{DequantPair, PrecomputedFilterParams};
use crate::common::types::*;
use crate::decoder::api::DecodeError;
use crate::decoder::bit_reader::VP8HeaderBitReader;
use crate::slice_reader::SliceReader;

impl DecoderContext {
    /// Parse the VP8 keyframe header and prepare the decoder context.
    ///
    /// This reads the frame tag, dimensions, partition layout, segment
    /// parameters, quantization indices, filter settings, and token
    /// probabilities. On success, all buffers are sized for the frame
    /// and `self.tables` is fully populated.
    pub(crate) fn read_frame_header(&mut self, data: &[u8]) -> Result<(), DecodeError> {
        let mut r = SliceReader::new(data);
        let mut b = VP8HeaderBitReader::new();

        // ---- Frame tag (3 bytes) ----
        let tag = r.read_u24_le()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodeError::UnsupportedFeature(
                "Non-keyframe frames".into(),
            ));
        }

        // (tag >> 1) & 7 is 0..=7, always fits u8.
        self.tables.version = ((tag >> 1) & 7) as u8;
        self.tables.for_display = (tag >> 4) & 1 != 0;
        let first_partition_size = tag >> 5;

        // ---- Start code ----
        let mut start_code = [0u8; 3];
        r.read_exact(&mut start_code)?;
        if start_code != [0x9d, 0x01, 0x2a] {
            return Err(DecodeError::Vp8MagicInvalid(start_code));
        }

        // ---- Dimensions ----
        let w = r.read_u16_le()?;
        let h = r.read_u16_le()?;
        self.tables.width = w & 0x3FFF;
        self.tables.height = h & 0x3FFF;

        // Reject zero dimensions (no valid WebP has 0×0 or 0×N pixels)
        if self.tables.width == 0 || self.tables.height == 0 {
            return Err(DecodeError::ImageTooLarge);
        }

        // WebP spec limit: 16383×16383
        if self.tables.width > 16383 || self.tables.height > 16383 {
            return Err(DecodeError::ImageTooLarge);
        }

        self.tables.mbwidth = self.tables.width.div_ceil(16);
        self.tables.mbheight = self.tables.height.div_ceil(16);

        // Overflow-safe macroblock count check.
        // Max valid: ceil(16383/16) * ceil(16383/16) = 1024 * 1024 = 1_048_576
        let mb_count = u32::from(self.tables.mbwidth)
            .checked_mul(u32::from(self.tables.mbheight))
            .ok_or(DecodeError::ImageTooLarge)?;
        // Sanity bound: ~1M macroblocks for 16383×16383
        if mb_count > 1024 * 1024 {
            return Err(DecodeError::ImageTooLarge);
        }

        // ---- Read partition 0 data into the boolean decoder ----
        // first_partition_size is at most 19 bits (u24 >> 5), always fits usize.
        let size = first_partition_size as usize;
        let mut part0_data = vec![0u8; size];
        r.read_exact(&mut part0_data)?;
        b.init(part0_data)?;

        // ---- Color space & pixel type ----
        let color_space = b.read_literal(1);
        self.tables.pixel_type = b.read_literal(1);
        if color_space != 0 {
            return Err(DecodeError::ColorSpaceInvalid(color_space));
        }

        // ---- Segments ----
        self.tables.segments_enabled = b.read_flag();
        if self.tables.segments_enabled {
            self.read_segment_updates(&mut b)?;
        }

        // ---- Filter parameters ----
        self.tables.filter_type = b.read_flag();
        self.tables.filter_level = b.read_literal(6);
        self.tables.sharpness_level = b.read_literal(3);

        self.tables.loop_filter_adjustments_enabled = b.read_flag();
        if self.tables.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments(&mut b)?;
        }

        // ---- Partitions ----
        // read_literal(2) returns 0..=3, so num_partitions is 1, 2, 4, or 8.
        let num_partitions = 1usize << (b.read_literal(2) as usize);
        b.check(())?;
        debug_assert!(num_partitions <= 8);
        self.tables.num_partitions = num_partitions as u8;

        // Determine extra cache rows from filter settings
        self.tables.extra_y_rows = if self.tables.filter_level == 0 {
            0
        } else if self.tables.filter_type {
            2 // simple filter
        } else {
            8 // normal filter
        };

        // ---- Size buffers ----
        self.ensure_capacity(
            self.tables.mbwidth,
            self.tables.mbheight,
            self.tables.extra_y_rows,
        );

        // ---- Initialize token partitions ----
        self.init_partitions(&mut r, num_partitions)?;

        // ---- Quantization ----
        self.read_quantization_indices(&mut b)?;

        // ---- Refresh entropy probs ----
        let _ = b.read_literal(1);

        // ---- Token probabilities ----
        self.update_token_probabilities(&mut b)?;
        self.populate_probs_by_position();

        // ---- Skip coefficient flag ----
        let mb_no_skip_coeff = b.read_literal(1);
        self.tables.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(b.read_literal(8))
        } else {
            None
        };
        b.check(())?;

        // ---- Precompute filter parameters ----
        self.precompute_filter_params();

        // ---- Initialize chroma dithering ----
        if self.dither_strength > 0 {
            let (enabled, amps) = crate::decoder::dither::init_dither_amplitudes(
                &self.tables.uv_quant_indices,
                self.dither_strength,
            );
            self.dither_enabled = enabled;
            self.dither_amp = amps;
            self.dither_rg = crate::decoder::dither::VP8Random::new();
        } else {
            self.dither_enabled = false;
            self.dither_amp = [0; MAX_SEGMENTS];
        }

        // Store the header reader for per-MB mode parsing during decode
        self.header_reader = b;

        Ok(())
    }

    fn read_segment_updates(&mut self, b: &mut VP8HeaderBitReader) -> Result<(), DecodeError> {
        self.tables.segments_update_map = b.read_flag();
        let update_segment_feature_data = b.read_flag();

        if update_segment_feature_data {
            let segment_feature_mode = b.read_flag();

            for i in 0..MAX_SEGMENTS {
                self.tables.segment_delta_values[i] = !segment_feature_mode;
            }

            for i in 0..MAX_SEGMENTS {
                self.tables.segment_quantizer_level[i] = b.read_optional_signed_value(7) as i8;
            }

            for i in 0..MAX_SEGMENTS {
                self.tables.segment_loopfilter_level[i] = b.read_optional_signed_value(6) as i8;
            }
        }

        if self.tables.segments_update_map {
            for i in 0..3 {
                let update = b.read_flag();
                self.tables.segment_tree_probs[i] = if update { b.read_literal(8) } else { 255 };
            }
        }

        Ok(b.check(())?)
    }

    fn read_loop_filter_adjustments(
        &mut self,
        b: &mut VP8HeaderBitReader,
    ) -> Result<(), DecodeError> {
        if b.read_flag() {
            for i in 0..4 {
                self.tables.ref_delta[i] = b.read_optional_signed_value(6);
            }
            for i in 0..4 {
                self.tables.mode_delta[i] = b.read_optional_signed_value(6);
            }
        }

        Ok(b.check(())?)
    }

    fn init_partitions(&mut self, r: &mut SliceReader<'_>, n: usize) -> Result<(), DecodeError> {
        use byteorder_lite::{ByteOrder, LittleEndian};

        let remaining = r.remaining();
        let mut all_data = Vec::new();
        let mut boundaries = Vec::with_capacity(n);

        if n > 1 {
            // Partition size table: 3 bytes per partition for n-1 partitions
            let size_table_bytes = 3 * (n - 1);
            if size_table_bytes > remaining {
                return Err(DecodeError::BitStreamError);
            }
            let mut sizes = vec![0u8; size_table_bytes];
            r.read_exact(sizes.as_mut_slice())?;

            for s in sizes.chunks(3) {
                let size = LittleEndian::read_u24(s) as usize;
                // Validate partition size against remaining data
                if size > r.remaining() {
                    return Err(DecodeError::BitStreamError);
                }
                let start = all_data.len();
                all_data.resize(start + size, 0);
                r.read_exact(&mut all_data[start..start + size])?;
                boundaries.push((start, size));
            }
        }

        // Last partition: read to end
        let start = all_data.len();
        r.read_to_end(&mut all_data)?;
        let size = all_data.len() - start;
        if size == 0 {
            return Err(DecodeError::BitStreamError);
        }
        boundaries.push((start, size));

        self.partitions.init(all_data, &boundaries);

        Ok(())
    }

    fn read_quantization_indices(&mut self, b: &mut VP8HeaderBitReader) -> Result<(), DecodeError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }
        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let yac_abs = b.read_literal(7);
        let ydc_delta = b.read_optional_signed_value(4);
        let y2dc_delta = b.read_optional_signed_value(4);
        let y2ac_delta = b.read_optional_signed_value(4);
        let uvdc_delta = b.read_optional_signed_value(4);
        let uvac_delta = b.read_optional_signed_value(4);

        let n = if self.tables.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };

        for i in 0..n {
            let base = i32::from(if self.tables.segments_enabled {
                if self.tables.segment_delta_values[i] {
                    i16::from(self.tables.segment_quantizer_level[i]) + i16::from(yac_abs)
                } else {
                    i16::from(self.tables.segment_quantizer_level[i])
                }
            } else {
                i16::from(yac_abs)
            });

            let ydc = dc_quant(base + ydc_delta);
            let yac = ac_quant(base);

            let y2dc = dc_quant(base + y2dc_delta) * 2;
            // ac_quant max is 1172, so 1172 * 155 / 100 = 1817, fits i16.
            let y2ac_i32 = i32::from(ac_quant(base + y2ac_delta)) * 155 / 100;
            let mut y2ac = y2ac_i32.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            if y2ac < 8 {
                y2ac = 8;
            }

            let mut uvdc = dc_quant(base + uvdc_delta);
            if uvdc > 132 {
                uvdc = 132;
            }
            let uvac = ac_quant(base + uvac_delta);

            // Plane indices: 0 = Y, 1 = Y2, 2 = UV
            self.tables.dequant[i][0] = DequantPair { dc: ydc, ac: yac };
            self.tables.dequant[i][1] = DequantPair { dc: y2dc, ac: y2ac };
            self.tables.dequant[i][2] = DequantPair { dc: uvdc, ac: uvac };

            // Store UV AC quantizer index for dithering amplitude computation
            self.tables.uv_quant_indices[i] = base + uvac_delta;
        }

        Ok(b.check(())?)
    }

    fn update_token_probabilities(
        &mut self,
        b: &mut VP8HeaderBitReader,
    ) -> Result<(), DecodeError> {
        // Start from the default probabilities on each frame
        // (matching v1 which initializes from COEFF_PROBS at construction)
        let mut probs = COEFF_PROBS;

        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, &update_prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if b.read_bool(update_prob) {
                            probs[i][j][k][t] = b.read_literal(8);
                        }
                    }
                }
            }
        }

        // Populate position-indexed probability table directly.
        // Maps coefficient position -> band -> flat u8 probs.
        let out = &mut *self.tables.probs_by_pos;
        for (plane_idx, plane_probs) in probs.iter().enumerate() {
            for pos in 0..17 {
                let band = if pos < 16 {
                    // COEFF_BANDS values are 0..=7, always fits usize.
                    COEFF_BANDS[pos] as usize
                } else {
                    7
                };
                for (ctx, ctx_probs) in plane_probs[band].iter().enumerate() {
                    out[plane_idx][pos][ctx].copy_from_slice(&ctx_probs[..11]);
                }
            }
        }

        Ok(b.check(())?)
    }

    fn populate_probs_by_position(&mut self) {
        // Already done inline in update_token_probabilities for v2,
        // since we store flat u8 probs rather than TreeNode arrays.
        // This is a no-op but kept for API symmetry with v1.
    }

    fn precompute_filter_params(&mut self) {
        let base_filter_level = i32::from(self.tables.filter_level);

        for seg_id in 0..MAX_SEGMENTS {
            for is_b in 0..2u8 {
                let mut filter_level = base_filter_level;

                if filter_level == 0 {
                    self.tables.filter[seg_id][is_b as usize] = PrecomputedFilterParams::default();
                    continue;
                }

                if self.tables.segments_enabled {
                    if self.tables.segment_delta_values[seg_id] {
                        filter_level += i32::from(self.tables.segment_loopfilter_level[seg_id]);
                    } else {
                        filter_level = i32::from(self.tables.segment_loopfilter_level[seg_id]);
                    }
                }

                filter_level = filter_level.clamp(0, 63);

                if self.tables.loop_filter_adjustments_enabled {
                    filter_level += self.tables.ref_delta[0];
                    if is_b != 0 {
                        filter_level += self.tables.mode_delta[0];
                    }
                }

                // Clamped to 0..=63, always fits u8.
                let filter_level = filter_level.clamp(0, 63) as u8;

                // Interior limit
                let mut interior_limit = filter_level;
                if self.tables.sharpness_level > 0 {
                    interior_limit >>= if self.tables.sharpness_level > 4 {
                        2
                    } else {
                        1
                    };
                    if interior_limit > 9 - self.tables.sharpness_level {
                        interior_limit = 9 - self.tables.sharpness_level;
                    }
                }
                if interior_limit == 0 {
                    interior_limit = 1;
                }

                // HEV threshold
                let hev_threshold = if filter_level >= 40 {
                    2
                } else if filter_level >= 15 {
                    1
                } else {
                    0
                };

                let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
                let sub_bedge_limit = (filter_level * 2) + interior_limit;

                self.tables.filter[seg_id][is_b as usize] = PrecomputedFilterParams {
                    filter_level,
                    interior_limit,
                    hev_threshold,
                    mbedge_limit,
                    sub_bedge_limit,
                };
            }
        }
    }
}
