//! Frame header parsing for VP8 decoder.
//!
//! Contains `read_frame_header`, `read_segment_updates`, `read_quantization_indices`,
//! `read_loop_filter_adjustments`, `init_partitions`, `update_token_probabilities`,
//! and `populate_probs_by_position`.

use super::*;

impl<'a> Vp8Decoder<'a> {
    pub(super) fn update_token_probabilities(&mut self) -> Result<(), DecodeError> {
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if self.b.read_bool(*prob) {
                            let v = self.b.read_literal(8);
                            self.token_probs[i][j][k][t].prob = v;
                        }
                    }
                }
            }
        }
        Ok(self.b.check(())?)
    }

    /// Populate the position-indexed probability table from token_probs.
    /// This eliminates the COEFF_BANDS lookup in the coefficient reading hot path.
    pub(super) fn populate_probs_by_position(&mut self) {
        for plane in 0..4 {
            for pos in 0..17 {
                // Position 16 uses band 7 (sentinel for n+1 lookahead)
                let band = if pos < 16 {
                    COEFF_BANDS[pos] as usize
                } else {
                    7
                };
                for ctx in 0..3 {
                    self.token_probs_by_pos[plane][pos][ctx] = self.token_probs[plane][band][ctx];
                }
            }
        }
    }

    pub(super) fn init_partitions(&mut self, n: usize) -> Result<(), DecodeError> {
        use byteorder_lite::{ByteOrder, LittleEndian};

        let mut all_data = Vec::new();
        let mut boundaries = Vec::with_capacity(n);

        if n > 1 {
            let mut sizes = vec![0; 3 * n - 3];
            self.r.read_exact(sizes.as_mut_slice())?;

            for s in sizes.chunks(3) {
                let size = LittleEndian::read_u24(s) as usize;

                let start = all_data.len();
                all_data.resize(start + size, 0);
                self.r.read_exact(&mut all_data[start..start + size])?;
                boundaries.push((start, size));
            }
        }

        // Last partition - read to end
        let start = all_data.len();
        self.r.read_to_end(&mut all_data)?;
        let size = all_data.len() - start;
        boundaries.push((start, size));

        self.partitions.init(all_data, &boundaries);

        Ok(())
    }

    pub(super) fn read_quantization_indices(&mut self) -> Result<(), DecodeError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let yac_abs = self.b.read_literal(7);
        let ydc_delta = self.b.read_optional_signed_value(4);
        let y2dc_delta = self.b.read_optional_signed_value(4);
        let y2ac_delta = self.b.read_optional_signed_value(4);
        let uvdc_delta = self.b.read_optional_signed_value(4);
        let uvac_delta = self.b.read_optional_signed_value(4);

        let n = if self.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for i in 0usize..n {
            let base = i32::from(if self.segments_enabled {
                if self.segment[i].delta_values {
                    i16::from(self.segment[i].quantizer_level) + i16::from(yac_abs)
                } else {
                    i16::from(self.segment[i].quantizer_level)
                }
            } else {
                i16::from(yac_abs)
            });

            self.segment[i].ydc = dc_quant(base + ydc_delta);
            self.segment[i].yac = ac_quant(base);

            self.segment[i].y2dc = dc_quant(base + y2dc_delta) * 2;
            // The intermediate result (max`284*155`) can be larger than the `i16` range.
            self.segment[i].y2ac = (i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16;

            self.segment[i].uvdc = dc_quant(base + uvdc_delta);
            self.segment[i].uvac = ac_quant(base + uvac_delta);
            // Store UV AC quantizer index for dithering amplitude computation
            self.uv_quant_indices[i] = base + uvac_delta;

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        Ok(self.b.check(())?)
    }

    pub(super) fn read_loop_filter_adjustments(&mut self) -> Result<(), DecodeError> {
        if self.b.read_flag() {
            for i in 0usize..4 {
                self.ref_delta[i] = self.b.read_optional_signed_value(6);
            }

            for i in 0usize..4 {
                self.mode_delta[i] = self.b.read_optional_signed_value(6);
            }
        }

        Ok(self.b.check(())?)
    }

    pub(super) fn read_segment_updates(&mut self) -> Result<(), DecodeError> {
        // Section 9.3
        self.segments_update_map = self.b.read_flag();
        let update_segment_feature_data = self.b.read_flag();

        if update_segment_feature_data {
            let segment_feature_mode = self.b.read_flag();

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].quantizer_level = self.b.read_optional_signed_value(7) as i8;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].loopfilter_level = self.b.read_optional_signed_value(6) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0usize..3 {
                let update = self.b.read_flag();

                let prob = if update { self.b.read_literal(8) } else { 255 };
                self.segment_tree_nodes[i].prob = prob;
            }
        }

        Ok(self.b.check(())?)
    }

    pub(super) fn read_frame_header(&mut self) -> Result<(), DecodeError> {
        let tag = self.r.read_u24_le()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodeError::UnsupportedFeature(
                "Non-keyframe frames".into(),
            ));
        }

        self.frame.version = ((tag >> 1) & 7) as u8;
        self.frame.for_display = (tag >> 4) & 1 != 0;

        let first_partition_size = tag >> 5;
        self.first_partition_size = first_partition_size;

        let mut tag = [0u8; 3];
        self.r.read_exact(&mut tag)?;

        if tag != [0x9d, 0x01, 0x2a] {
            return Err(DecodeError::Vp8MagicInvalid(tag));
        }

        let w = self.r.read_u16_le()?;
        let h = self.r.read_u16_le()?;

        self.frame.width = w & 0x3FFF;
        self.frame.height = h & 0x3FFF;

        self.mbwidth = self.frame.width.div_ceil(16);
        self.mbheight = self.frame.height.div_ceil(16);

        // defaults are intra mode DC and complexity 0
        self.top = vec![PreviousMacroBlock::default(); self.mbwidth.into()];
        self.left = PreviousMacroBlock::default();

        // Pre-allocate macroblocks to avoid repeated Vec reallocation in decode loop
        self.macroblocks =
            Vec::with_capacity(usize::from(self.mbwidth) * usize::from(self.mbheight));

        // Frame buffers don't need FILTER_PADDING — padding is only needed on
        // cache buffers where loop filtering operates. Frame buffers receive the
        // final output from output_row_from_cache via copy_from_slice within bounds.
        self.frame.ybuf =
            vec![0u8; usize::from(self.mbwidth) * 16 * usize::from(self.mbheight) * 16];
        self.frame.ubuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];
        self.frame.vbuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];

        self.top_border_y = vec![127u8; self.frame.width as usize + 4 + 16];
        self.left_border_y = vec![129u8; 1 + 16];

        // 8 pixels per macroblock
        self.top_border_u = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_u = vec![129u8; 1 + 8];

        self.top_border_v = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_v = vec![129u8; 1 + 8];

        // Initialize row cache for better cache locality during loop filtering
        // We allocate with max extra_rows (8 for normal filter) - actual value set after reading filter_type
        self.cache_y_stride = usize::from(self.mbwidth) * 16;
        self.cache_uv_stride = usize::from(self.mbwidth) * 8;
        // extra_y_rows will be set properly after we read filter_type, for now use 0
        self.extra_y_rows = 0;

        let size = first_partition_size as usize;
        let mut data = vec![0u8; size];
        self.r.read_exact(&mut data)?;

        // initialise binary decoder
        self.b.init(data)?;

        let color_space = self.b.read_literal(1);
        self.frame.pixel_type = self.b.read_literal(1);

        if color_space != 0 {
            return Err(DecodeError::ColorSpaceInvalid(color_space));
        }

        self.segments_enabled = self.b.read_flag();
        if self.segments_enabled {
            self.read_segment_updates()?;
        }

        self.frame.filter_type = self.b.read_flag();
        self.frame.filter_level = self.b.read_literal(6);
        self.frame.sharpness_level = self.b.read_literal(3);

        self.loop_filter_adjustments_enabled = self.b.read_flag();
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments()?;
        }

        let num_partitions = 1 << self.b.read_literal(2) as usize;
        self.b.check(())?;

        // Now that we know filter_type, allocate the row cache
        // extra_rows: 8 for normal filter, 2 for simple filter, 0 for no filter
        self.extra_y_rows = if self.frame.filter_level == 0 {
            0
        } else if self.frame.filter_type {
            2 // simple filter
        } else {
            8 // normal filter
        };
        let extra_uv_rows = self.extra_y_rows / 2;

        // Cache layout: [extra_rows][16 rows for current macroblock row]
        // extra_rows holds bottom rows from previous MB row for filter context
        // FILTER_PADDING allows fixed-size region extraction for bounds-check-free filtering.
        // Uses MAX_FILTER_STRIDE to match compile-time constant region sizes in loop_filter.
        let cache_y_rows = self.extra_y_rows + 16;
        let cache_uv_rows = extra_uv_rows + 8;
        self.cache_y = vec![128u8; cache_y_rows * self.cache_y_stride + FILTER_PADDING];
        self.cache_u = vec![128u8; cache_uv_rows * self.cache_uv_stride + FILTER_PADDING];
        self.cache_v = vec![128u8; cache_uv_rows * self.cache_uv_stride + FILTER_PADDING];

        self.num_partitions = num_partitions as u8;
        self.init_partitions(num_partitions)?;

        self.read_quantization_indices()?;

        // Refresh entropy probs ?????
        let _ = self.b.read_literal(1);

        self.update_token_probabilities()?;
        self.populate_probs_by_position();

        let mb_no_skip_coeff = self.b.read_literal(1);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(self.b.read_literal(8))
        } else {
            None
        };
        self.b.check(())?;
        Ok(())
    }

    pub(super) fn read_macroblock_header(
        &mut self,
        mbx: usize,
    ) -> Result<MacroBlock, InternalDecodeError> {
        let mut mb = MacroBlock::default();

        if self.segments_enabled && self.segments_update_map {
            mb.segmentid = self.b.read_with_tree(&self.segment_tree_nodes) as u8;
        };

        mb.coeffs_skipped = if let Some(prob) = self.prob_skip_false {
            self.b.read_bool(prob)
        } else {
            false
        };

        // intra prediction
        let luma = self.b.read_with_tree(&KEYFRAME_YMODE_NODES);
        mb.luma_mode =
            LumaMode::from_i8(luma).ok_or(InternalDecodeError::LumaPredictionModeInvalid)?;

        // Extract top[mbx] once to eliminate per-access bounds checks in the B-mode loop
        let top_mb = &mut self.top[mbx];
        match mb.luma_mode.into_intra() {
            // `LumaMode::B` - This is predicted individually
            None => {
                for y in 0usize..4 {
                    for x in 0usize..4 {
                        let top = top_mb.bpred[x];
                        let left = self.left.bpred[y];
                        let intra = self.b.read_with_tree(
                            &KEYFRAME_BPRED_MODE_NODES[top as usize][left as usize],
                        );
                        let bmode = IntraMode::from_i8(intra)
                            .ok_or(InternalDecodeError::IntraPredictionModeInvalid)?;
                        mb.bpred[x + y * 4] = bmode;

                        top_mb.bpred[x] = bmode;
                        self.left.bpred[y] = bmode;
                    }
                }
            }
            Some(mode) => {
                for i in 0usize..4 {
                    mb.bpred[12 + i] = mode;
                    self.left.bpred[i] = mode;
                }
            }
        }

        let chroma = self.b.read_with_tree(&KEYFRAME_UV_MODE_NODES);
        mb.chroma_mode =
            ChromaMode::from_i8(chroma).ok_or(InternalDecodeError::ChromaPredictionModeInvalid)?;

        // top should store the bottom of the current bpred, which is the final 4 values
        top_mb.bpred = mb.bpred[12..].try_into().unwrap();

        self.b.check(mb)
    }
}
