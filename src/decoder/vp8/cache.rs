//! Row cache management for VP8 decoder loop filtering.
//!
//! Contains `filter_row_in_cache`, `output_row_from_cache`, `rotate_extra_rows`,
//! and `calculate_filter_parameters`.

use super::*;

impl<'a> Vp8Decoder<'a> {
    /// Filters a row of macroblocks in the cache.
    /// Used by the diagnostic path which populates filter params from the macroblocks vec.
    pub(super) fn filter_row_in_cache(&mut self, mby: usize, simd_token: SimdTokenType) {
        let mbwidth = self.mbwidth as usize;

        // Build filter parameters from the macroblocks vec (diagnostic path).
        self.mb_filter_params.clear();
        if self.mb_filter_params.capacity() < mbwidth {
            self.mb_filter_params
                .reserve(mbwidth - self.mb_filter_params.capacity());
        }
        for mbx in 0..mbwidth {
            let mb = self.macroblocks[mby * mbwidth + mbx];
            let is_b = mb.luma_mode == LumaMode::B;
            let p = &self.precomputed_filter[mb.segmentid as usize][is_b as usize];
            let do_subblock_filtering =
                is_b || (!mb.coeffs_skipped && mb.non_zero_dct);

            self.mb_filter_params.push(loop_filter::MbFilterParams {
                filter_level: p.filter_level,
                interior_limit: p.interior_limit,
                hev_threshold: p.hev_threshold,
                mbedge_limit: p.mbedge_limit,
                sub_bedge_limit: p.sub_bedge_limit,
                do_subblock_filtering,
            });
        }

        self.filter_row_in_cache_precomputed(mby, simd_token);
    }

    /// Filters a row of macroblocks in the cache.
    /// Assumes `self.mb_filter_params` is already populated for this row.
    /// Used by the main decode path (single-pass: params computed during decode).
    pub(super) fn filter_row_in_cache_precomputed(
        &mut self,
        mby: usize,
        simd_token: SimdTokenType,
    ) {
        let mbwidth = self.mbwidth as usize;
        let cache_y_stride = self.cache_y_stride;
        let cache_uv_stride = self.cache_uv_stride;
        let extra_y_rows = self.extra_y_rows;

        // Take ownership of mb_params to allow split borrows with cache buffers
        let mb_params = core::mem::take(&mut self.mb_filter_params);

        // Use the single #[arcane] entry point when SIMD is available.
        // All #[rite] filter functions inline into this one target_feature region,
        // eliminating per-call dispatch overhead (~7.6M instructions saved).
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "wasm32",
        ))]
        if let Some(token) = simd_token {
            loop_filter::filter_row_simd(
                token,
                &mut self.cache_y[..],
                &mut self.cache_u[..],
                &mut self.cache_v[..],
                cache_y_stride,
                cache_uv_stride,
                extra_y_rows,
                self.frame.filter_type,
                mby,
                &mb_params,
            );
            self.mb_filter_params = mb_params;
            return;
        }

        // Scalar fallback (no SIMD token available)
        let extra_uv_rows = extra_y_rows / 2;
        for mbx in 0..mbwidth {
            let p = &mb_params[mbx];
            if p.filter_level == 0 {
                continue;
            }

            let mbedge_limit = p.mbedge_limit;
            let sub_bedge_limit = p.sub_bedge_limit;
            let hev_threshold = p.hev_threshold;
            let interior_limit = p.interior_limit;
            let do_subblock_filtering = p.do_subblock_filtering;

            if mbx > 0 {
                if self.frame.filter_type {
                    simple_filter_horizontal_16_rows(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        mbedge_limit,
                    );
                } else {
                    normal_filter_horizontal_mb_16_rows(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        hev_threshold,
                        interior_limit,
                        mbedge_limit,
                    );
                    normal_filter_horizontal_uv_mb(
                        &mut self.cache_u[..],
                        &mut self.cache_v[..],
                        extra_uv_rows,
                        mbx * 8,
                        cache_uv_stride,
                        hev_threshold,
                        interior_limit,
                        mbedge_limit,
                    );
                }
            }

            if do_subblock_filtering {
                if self.frame.filter_type {
                    for x in (4usize..16 - 1).step_by(4) {
                        simple_filter_horizontal_16_rows(
                            &mut self.cache_y[..],
                            extra_y_rows,
                            mbx * 16 + x,
                            cache_y_stride,
                            sub_bedge_limit,
                        );
                    }
                } else {
                    for x in (4usize..16 - 3).step_by(4) {
                        normal_filter_horizontal_sub_16_rows(
                            &mut self.cache_y[..],
                            extra_y_rows,
                            mbx * 16 + x,
                            cache_y_stride,
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                        );
                    }
                    normal_filter_horizontal_uv_sub(
                        &mut self.cache_u[..],
                        &mut self.cache_v[..],
                        extra_uv_rows,
                        mbx * 8 + 4,
                        cache_uv_stride,
                        hev_threshold,
                        interior_limit,
                        sub_bedge_limit,
                    );
                }
            }

            if mby > 0 {
                if self.frame.filter_type {
                    simple_filter_vertical_16_cols(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        mbedge_limit,
                    );
                } else {
                    normal_filter_vertical_mb_16_cols(
                        &mut self.cache_y[..],
                        extra_y_rows,
                        mbx * 16,
                        cache_y_stride,
                        hev_threshold,
                        interior_limit,
                        mbedge_limit,
                    );
                    normal_filter_vertical_uv_mb(
                        &mut self.cache_u[..],
                        &mut self.cache_v[..],
                        extra_uv_rows,
                        mbx * 8,
                        cache_uv_stride,
                        hev_threshold,
                        interior_limit,
                        mbedge_limit,
                    );
                }
            }

            if do_subblock_filtering {
                if self.frame.filter_type {
                    for y in (4usize..16 - 1).step_by(4) {
                        simple_filter_vertical_16_cols(
                            &mut self.cache_y[..],
                            extra_y_rows + y,
                            mbx * 16,
                            cache_y_stride,
                            sub_bedge_limit,
                        );
                    }
                } else {
                    for y in (4usize..16 - 3).step_by(4) {
                        normal_filter_vertical_sub_16_cols(
                            &mut self.cache_y[..],
                            extra_y_rows + y,
                            mbx * 16,
                            cache_y_stride,
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                        );
                    }
                    normal_filter_vertical_uv_sub(
                        &mut self.cache_u[..],
                        &mut self.cache_v[..],
                        extra_uv_rows + 4,
                        mbx * 8,
                        cache_uv_stride,
                        hev_threshold,
                        interior_limit,
                        sub_bedge_limit,
                    );
                }
            }
        }

        self.mb_filter_params = mb_params;
    }

    /// Copy the filtered row from cache to final output buffers
    /// Uses delayed output: filter modifies pixels above and below the edge,
    /// so we delay outputting the bottom extra_rows until the next row is filtered.
    pub(super) fn output_row_from_cache(&mut self, mby: usize) {
        let mbwidth = self.mbwidth as usize;
        let mbheight = self.mbheight as usize;
        let luma_w = mbwidth * 16;
        let chroma_w = mbwidth * 8;
        let extra_y_rows = self.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;
        let is_first_row = mby == 0;
        let is_last_row = mby == mbheight - 1;

        // Determine which rows to output:
        // - First row: output rows extra_y_rows to extra_y_rows + (16 - extra_y_rows) = rows up to 16
        //   but skip the bottom extra_y_rows (they'll be output with next row after filtering)
        // - Middle rows: output extra area (0 to extra_y_rows) + current row minus bottom extra_y_rows
        // - Last row: output extra area + full current row

        let (src_start_row, num_y_rows, dst_start_y_row) = if is_first_row {
            (extra_y_rows, 16 - extra_y_rows, 0usize)
        } else if is_last_row {
            (0, extra_y_rows + 16, mby * 16 - extra_y_rows)
        } else {
            (0, 16, mby * 16 - extra_y_rows)
        };

        // Copy luma rows from cache to frame buffer.
        // cache_y_stride == luma_w, so source rows are contiguous — use bulk copy.
        {
            let src_start = src_start_row * self.cache_y_stride;
            let dst_start = dst_start_y_row * luma_w;
            let total = num_y_rows * luma_w;
            self.frame.ybuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_y[src_start..src_start + total]);
        }

        // Same logic for chroma but with half the rows
        let (src_start_row_uv, num_uv_rows, dst_start_uv_row) = if is_first_row {
            (extra_uv_rows, 8 - extra_uv_rows, 0usize)
        } else if is_last_row {
            (0, extra_uv_rows + 8, mby * 8 - extra_uv_rows)
        } else {
            (0, 8, mby * 8 - extra_uv_rows)
        };

        // Copy chroma rows from cache to frame buffer.
        // cache_uv_stride == chroma_w, so source rows are contiguous — use bulk copy.
        {
            let src_start = src_start_row_uv * self.cache_uv_stride;
            let dst_start = dst_start_uv_row * chroma_w;
            let total = num_uv_rows * chroma_w;
            self.frame.ubuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_u[src_start..src_start + total]);
            self.frame.vbuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_v[src_start..src_start + total]);
        }
    }

    /// Copy bottom rows of current cache to extra area for next row's filtering
    pub(super) fn rotate_extra_rows(&mut self) {
        let extra_y_rows = self.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;

        if extra_y_rows == 0 {
            return;
        }

        // Copy bottom extra_y_rows of current MB row to the extra area
        // Source: rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16) = rows 16..(extra_y_rows + 16)
        // Actually: the bottom extra_y_rows of the 16-row area
        // Which is rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16)
        // = rows 16..(16 + extra_y_rows)... wait that's wrong
        // The current row is at rows extra_y_rows..(extra_y_rows + 16)
        // The bottom extra_y_rows are at rows (extra_y_rows + 16 - extra_y_rows)..(extra_y_rows + 16)
        // = rows 16..(extra_y_rows + 16) -- no that's still wrong
        // Let me think again:
        // - Current row occupies rows extra_y_rows to extra_y_rows + 15 (16 rows)
        // - We want the bottom extra_y_rows of these, which are rows (extra_y_rows + 16 - extra_y_rows) to (extra_y_rows + 15)
        // - Wait, extra_y_rows + 16 - extra_y_rows = 16, so rows 16..16+extra_y_rows = rows 16..16+extra_y_rows
        // - Hmm, that's outside the current row area...

        // Let me reconsider. Current row (16 pixels):
        // - Starts at row index extra_y_rows
        // - Ends at row index extra_y_rows + 15
        // - Bottom extra_y_rows rows are at indices (extra_y_rows + 16 - extra_y_rows) to (extra_y_rows + 15)
        // - = indices 16 to (extra_y_rows + 15)... that's wrong

        // Actually: the 16-row area is at indices extra_y_rows..(extra_y_rows + 16)
        // The last extra_y_rows rows of this area are at indices:
        //   (extra_y_rows + 16 - extra_y_rows) .. (extra_y_rows + 16)
        //   = 16 .. (extra_y_rows + 16)
        // Wait, that gives 16..24 for extra_y_rows=8, which is 8 rows. That's correct!
        // But 16 > extra_y_rows when extra_y_rows = 8, so indices 16..24 are valid.

        // Destination: rows 0..extra_y_rows

        // For luma:
        let src_start_row = 16; // = extra_y_rows + 16 - extra_y_rows = 16
        let src_start = src_start_row * self.cache_y_stride;
        let copy_size = extra_y_rows * self.cache_y_stride;
        // Copy from src_start to 0
        self.cache_y
            .copy_within(src_start..src_start + copy_size, 0);

        // For chroma:
        let src_start_row_uv = 8; // = extra_uv_rows + 8 - extra_uv_rows = 8
        let src_start_uv = src_start_row_uv * self.cache_uv_stride;
        let copy_size_uv = extra_uv_rows * self.cache_uv_stride;
        self.cache_u
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
        self.cache_v
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
    }

    // calculate_filter_parameters removed — replaced by precomputed_filter table
    // (populated once in precompute_filter_params after frame header parsing).
}
