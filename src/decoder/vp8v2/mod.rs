//! Lossy VP8 decoder targeting libwebp C parity.
//!
//! Key design points:
//! - `DecoderContext` with buffer reuse (no memset per decode)
//! - Streaming row pipeline (no full-frame Y/U/V buffers)
//! - Single `#[arcane]` per MB row (predict+IDCT+filter in one region)
//! - Precomputed filter/dequant tables in `FrameTables`
//! - Flat u8 probability tables (no TreeNode indirection)
//! - Fixed-size arrays for per-MB working storage

#[allow(dead_code)]
mod animation;
mod coefficients;
mod context;
mod header;
mod pipeline;
pub(super) mod predict_fused;
mod tables;
pub(crate) mod yuv_exact;

pub(crate) use context::DecoderContext;

use alloc::vec::Vec;

use crate::common::types::Frame;
use crate::common::types::{ChromaMode, IntraMode, LumaMode};
use crate::decoder::api::DecodeError;
use crate::decoder::internal_error::InternalDecodeError;
use crate::decoder::loop_filter::MbFilterParams;

use context::PreviousMacroBlock;

/// Per-macroblock data from coefficient parsing, consumed by the
/// prediction + reconstruction + filter pipeline.
#[derive(Clone, Copy, Default)]
pub(super) struct MbRowEntry {
    /// Luma prediction mode for this macroblock.
    pub(super) luma_mode: LumaMode,
    /// Chroma prediction mode for this macroblock.
    pub(super) chroma_mode: ChromaMode,
    /// I4 sub-block prediction modes (only valid when `luma_mode == LumaMode::B`).
    pub(super) bpred: [IntraMode; 16],
    /// Segment index (0-3).
    pub(super) segmentid: u8,
    /// Whether all coefficients were skipped (zero block).
    pub(super) coeffs_skipped: bool,
    /// Per-block non-zero bitmap. Bit i set = block i has non-zero coefficients.
    /// Blocks 0-15 = Y, 16-19 = U, 20-23 = V.
    pub(super) non_zero_blocks: u32,
    /// True if any block in this MB has non-zero DCT coefficients.
    pub(super) non_zero_dct: bool,
    /// True if any UV sub-block has non-zero AC coefficients.
    /// Used to suppress dithering on blocks with actual chroma detail.
    pub(super) has_nonzero_uv_ac: bool,
}

impl DecoderContext {
    /// Decode a VP8 bitstream to a `Frame` (Y/U/V planes).
    ///
    /// The `data` must be the raw VP8 bitstream (inside the RIFF/WebP
    /// container, after the VP8 chunk header has been stripped).
    pub fn decode_to_frame(&mut self, data: &[u8]) -> Result<Frame, DecodeError> {
        self.read_frame_header(data)?;
        self.decode_to_frame_internal().map_err(DecodeError::from)
    }

    /// Internal: decode MB rows to full-frame Y/U/V and build a Frame.
    fn decode_to_frame_internal(&mut self) -> Result<Frame, InternalDecodeError> {
        self.decode_mb_rows()?;

        let tables = &self.tables;

        // Build Frame from YUV plane buffers
        let frame = Frame {
            width: tables.width,
            height: tables.height,
            ybuf: core::mem::take(&mut self.ybuf),
            ubuf: core::mem::take(&mut self.ubuf),
            vbuf: core::mem::take(&mut self.vbuf),
            version: tables.version,
            for_display: tables.for_display,
            pixel_type: tables.pixel_type,
            filter_type: tables.filter_type,
            filter_level: tables.filter_level,
            sharpness_level: tables.sharpness_level,
        };

        // Reclaim empty vecs (will be reallocated on next decode if needed)
        self.ybuf = Vec::new();
        self.ubuf = Vec::new();
        self.vbuf = Vec::new();

        Ok(frame)
    }

    /// Decode a VP8 bitstream to RGB or RGBA pixels.
    ///
    /// `data` is the raw VP8 bitstream. `output` receives the pixel data.
    /// `bpp` is 3 for RGB or 4 for RGBA.
    ///
    /// Uses streaming cache-to-RGB conversion: each MB row's filtered cache
    /// is converted directly to the output buffer. No full-frame Y/U/V
    /// buffers are allocated (~4 MB saved for 4K images).
    ///
    /// Returns `(width, height)` on success.
    pub fn decode_to_rgb(
        &mut self,
        data: &[u8],
        output: &mut Vec<u8>,
        bpp: usize,
    ) -> Result<(u16, u16), DecodeError> {
        if bpp != 3 && bpp != 4 {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "unsupported bpp: {bpp}"
            )));
        }

        self.read_frame_header(data)?;

        let w = self.tables.width;
        let h = self.tables.height;
        let pixel_count = usize::from(w)
            .checked_mul(usize::from(h))
            .ok_or(DecodeError::ImageTooLarge)?;
        let output_size = pixel_count
            .checked_mul(bpp)
            .ok_or(DecodeError::ImageTooLarge)?;
        output.resize(output_size, 0);

        if self.tables.extra_y_rows >= 2 {
            // Streaming path: convert cache rows directly to RGB.
            // Requires extra_y_rows >= 2 so the cache has enough UV rows
            // for the fancy upsampler at MB row boundaries.
            self.decode_mb_rows_to_rgb(output, bpp)
                .map_err(DecodeError::from)?;
        } else {
            // Fallback for no-filter case (extra_y_rows=0): use full-frame
            // YUV buffers. This is rare (very high quality / filter disabled).
            let frame = self.decode_to_frame_internal()?;
            let fw = usize::from(frame.width);
            let fh = usize::from(frame.height);
            let mbwidth = (fw + 15) / 16;
            let y_stride = mbwidth * 16;
            let uv_stride = mbwidth * 8;
            yuv_exact::yuv420_to_rgb_exact(
                &frame.ybuf,
                &frame.ubuf,
                &frame.vbuf,
                fw,
                fh,
                y_stride,
                uv_stride,
                output,
                bpp,
            );
        }

        Ok((w, h))
    }

    /// Main decode loop for full-frame Y/U/V output. For each row: parse +
    /// predict/IDCT each MB individually, filter the row, copy to ybuf/ubuf/vbuf.
    fn decode_mb_rows(&mut self) -> Result<(), InternalDecodeError> {
        let mbwidth = usize::from(self.tables.mbwidth);
        let mbheight = usize::from(self.tables.mbheight);

        // Allocate full-frame Y/U/V buffers (overflow → error, not panic)
        let luma_w = mbwidth
            .checked_mul(16)
            .ok_or(InternalDecodeError::BitStreamError)?;
        let chroma_w = mbwidth
            .checked_mul(8)
            .ok_or(InternalDecodeError::BitStreamError)?;
        let chroma_h = mbheight
            .checked_mul(8)
            .and_then(|n| n.checked_add(1))
            .ok_or(InternalDecodeError::BitStreamError)?;
        let ybuf_len = mbheight
            .checked_mul(16)
            .and_then(|n| n.checked_mul(luma_w))
            .ok_or(InternalDecodeError::BitStreamError)?;
        let uvbuf_len = chroma_h
            .checked_mul(chroma_w)
            .ok_or(InternalDecodeError::BitStreamError)?;

        self.ybuf.resize(ybuf_len, 0);
        self.ubuf.resize(uvbuf_len, 0);
        self.vbuf.resize(uvbuf_len, 0);

        for mby in 0..mbheight {
            self.process_mb_row(mby)?;

            // Output cache to Y/U/V frame buffers
            self.output_row_from_cache(mby, mbheight, mbwidth);
            self.rotate_extra_rows();

            // Reset left borders for next row
            self.left_border_y.fill(129u8);
            self.left_border_u.fill(129u8);
            self.left_border_v.fill(129u8);
        }

        Ok(())
    }

    /// Streaming decode loop: converts cache rows directly to RGB/RGBA output.
    /// No full-frame Y/U/V buffers are allocated.
    fn decode_mb_rows_to_rgb(
        &mut self,
        output: &mut [u8],
        bpp: usize,
    ) -> Result<(), InternalDecodeError> {
        let mbheight = usize::from(self.tables.mbheight);
        let width = usize::from(self.tables.width);
        let height = usize::from(self.tables.height);
        let extra_y_rows = self.tables.extra_y_rows;
        let chroma_width = (width + 1) / 2;

        // Resize boundary UV row buffers (used for fancy upsampling at MB boundaries)
        self.prev_last_u_row.resize(chroma_width, 128);
        self.prev_last_v_row.resize(chroma_width, 128);

        for mby in 0..mbheight {
            self.process_mb_row(mby)?;

            // Convert cache rows directly to RGB output
            yuv_exact::convert_cache_rows_to_rgb(
                &self.cache_y,
                &self.cache_u,
                &self.cache_v,
                self.cache_y_stride,
                self.cache_uv_stride,
                extra_y_rows,
                mby,
                mbheight,
                width,
                height,
                output,
                bpp,
                &self.prev_last_u_row,
                &self.prev_last_v_row,
            );

            // Save the boundary UV row for the next iteration's fancy upsampling.
            // Cache UV row 7 is the last UV row in the visible region that won't
            // survive rotation. The next MB row's first even Y row needs it as
            // the "far" chroma reference.
            if mby + 1 < mbheight {
                let boundary_cache_uv_row = 7;
                let uv_start = boundary_cache_uv_row * self.cache_uv_stride;
                self.prev_last_u_row[..chroma_width]
                    .copy_from_slice(&self.cache_u[uv_start..uv_start + chroma_width]);
                self.prev_last_v_row[..chroma_width]
                    .copy_from_slice(&self.cache_v[uv_start..uv_start + chroma_width]);
            }

            self.rotate_extra_rows();

            // Reset left borders for next row
            self.left_border_y.fill(129u8);
            self.left_border_u.fill(129u8);
            self.left_border_v.fill(129u8);
        }

        Ok(())
    }

    /// Decode one MB row and convert to RGB/RGBA in a strip buffer.
    ///
    /// The strip buffer must be at least `width * max_strip_rows * bpp` bytes.
    /// Returns `(y_start, num_rows)` — the absolute Y row and row count written.
    ///
    /// After the last MB row (returns `num_rows > 0` for the final call),
    /// the caller should stop. This method handles all internal state
    /// (rotate_extra_rows, left border resets, boundary UV saves).
    pub(crate) fn decode_strip_mb_row(
        &mut self,
        mby: usize,
        strip: &mut [u8],
        bpp: usize,
    ) -> Result<(usize, usize), InternalDecodeError> {
        let mbheight = usize::from(self.tables.mbheight);
        let width = usize::from(self.tables.width);
        let height = usize::from(self.tables.height);
        let extra_y_rows = self.tables.extra_y_rows;
        let chroma_width = (width + 1) / 2;

        self.process_mb_row(mby)?;

        let (y_start, num_rows) = yuv_exact::convert_cache_rows_to_strip(
            &self.cache_y,
            &self.cache_u,
            &self.cache_v,
            self.cache_y_stride,
            self.cache_uv_stride,
            extra_y_rows,
            mby,
            mbheight,
            width,
            height,
            strip,
            bpp,
            &self.prev_last_u_row,
            &self.prev_last_v_row,
        );

        // Save boundary UV row for next MB row's fancy upsampling
        if mby + 1 < mbheight {
            let boundary_cache_uv_row = 7;
            let uv_start = boundary_cache_uv_row * self.cache_uv_stride;
            self.prev_last_u_row[..chroma_width]
                .copy_from_slice(&self.cache_u[uv_start..uv_start + chroma_width]);
            self.prev_last_v_row[..chroma_width]
                .copy_from_slice(&self.cache_v[uv_start..uv_start + chroma_width]);
        }

        self.rotate_extra_rows();

        // Reset left borders for next row
        self.left_border_y.fill(129u8);
        self.left_border_u.fill(129u8);
        self.left_border_v.fill(129u8);

        Ok((y_start, num_rows))
    }

    /// Initialize boundary UV row buffers for streaming decode.
    /// Must be called after `read_frame_header` and before `decode_strip_mb_row`.
    pub(crate) fn init_streaming_uv_buffers(&mut self) {
        let width = usize::from(self.tables.width);
        let chroma_width = (width + 1) / 2;
        self.prev_last_u_row.resize(chroma_width, 128);
        self.prev_last_v_row.resize(chroma_width, 128);
    }

    /// Parse + predict/IDCT + filter one MB row. After this returns, the cache
    /// contains filtered Y/U/V data ready for output (or direct conversion).
    fn process_mb_row(&mut self, mby: usize) -> Result<(), InternalDecodeError> {
        let mbwidth = usize::from(self.tables.mbwidth);
        let extra_y_rows = self.tables.extra_y_rows;
        let filter_type = self.tables.filter_type;
        let dither_enabled = self.dither_enabled;
        // num_partitions is 1, 2, 4, or 8 — always fits usize.
        let p = mby % self.tables.num_partitions as usize;
        self.left = PreviousMacroBlock::default();

        // Parse + predict/IDCT each MB immediately (single pass).
        // Coefficients in self.coeff_blocks are consumed and cleared
        // per MB, matching the expected decoder semantics.
        for mbx in 0..mbwidth {
            let mb = &mut self.mb_row_data[mbx];
            *mb = MbRowEntry::default();

            coefficients::read_macroblock_header(
                &mut self.header_reader,
                &self.tables,
                &mut self.top[mbx],
                &mut self.left,
                mb,
            )?;

            if !mb.coeffs_skipped {
                {
                    let mut reader = self.partitions.active_reader(p);
                    coefficients::read_residual_data(
                        &mut reader,
                        mb,
                        &mut self.coeff_blocks,
                        &self.tables.probs_by_pos,
                        // segmentid is 0..=3, always a valid index.
                        &self.tables.dequant[mb.segmentid as usize],
                        &mut self.top[mbx],
                        &mut self.left,
                    )?;
                    // reader auto-saves state on drop
                }
            } else {
                // Clear complexity context for skipped blocks
                let top_mb = &mut self.top[mbx];
                if mb.luma_mode != LumaMode::B {
                    self.left.complexity[0] = 0;
                    top_mb.complexity[0] = 0;
                }
                for i in 1usize..9 {
                    self.left.complexity[i] = 0;
                    top_mb.complexity[i] = 0;
                }
            }

            // Immediately predict+IDCT this MB (consumes coeff_blocks)
            predict_fused::process_luma_mb(
                &mut self.luma_ws,
                &mut self.coeff_blocks,
                mb,
                &mut self.cache_y,
                self.cache_y_stride,
                extra_y_rows,
                mbx,
                mby,
                mbwidth,
                &mut self.top_border_y,
                &mut self.left_border_y,
            );
            predict_fused::process_chroma_mb(
                &mut self.chroma_u_ws,
                &mut self.chroma_v_ws,
                &mut self.coeff_blocks,
                mb,
                &mut self.cache_u,
                &mut self.cache_v,
                self.cache_uv_stride,
                extra_y_rows,
                mbx,
                mby,
                &mut self.top_border_u,
                &mut self.left_border_u,
                &mut self.top_border_v,
                &mut self.left_border_v,
            );

            // Compute filter params from precomputed table
            let is_b = mb.luma_mode == LumaMode::B;
            // segmentid is 0..=3 (from read_segment_id), is_b is bool.
            let fp = &self.tables.filter[mb.segmentid as usize][is_b as usize];
            let do_subblock_filtering = is_b || (!mb.coeffs_skipped && mb.non_zero_dct);
            self.mb_filter_params[mbx] = MbFilterParams {
                filter_level: fp.filter_level,
                interior_limit: fp.interior_limit,
                hev_threshold: fp.hev_threshold,
                mbedge_limit: fp.mbedge_limit,
                sub_bedge_limit: fp.sub_bedge_limit,
                do_subblock_filtering,
            };

            // Compute dither amplitude inline.
            // Dithering is suppressed for skipped MBs and MBs with UV AC content.
            if dither_enabled {
                self.mb_dither_buf[mbx] = if mb.coeffs_skipped || mb.has_nonzero_uv_ac {
                    0
                } else {
                    self.dither_amp[mb.segmentid as usize]
                };
            }
        }

        // Filter the entire row (single SIMD boundary)
        pipeline::filter_mb_row(
            &mut self.cache_y,
            &mut self.cache_u,
            &mut self.cache_v,
            self.cache_y_stride,
            self.cache_uv_stride,
            extra_y_rows,
            filter_type,
            mby,
            &self.mb_filter_params[..mbwidth],
        );

        // Apply chroma dithering after filtering, before output.
        if dither_enabled {
            let extra_uv_rows = extra_y_rows / 2;
            let cache_uv_stride = self.cache_uv_stride;
            let dither_buf = core::mem::take(&mut self.mb_dither_buf);
            crate::decoder::dither::dither_row(
                &mut self.dither_rg,
                crate::decoder::dither::DitherRowParams {
                    cache_u: &mut self.cache_u,
                    cache_v: &mut self.cache_v,
                    cache_uv_stride,
                    extra_uv_rows,
                    mb_dither_amps: &dither_buf[..mbwidth],
                },
            );
            self.mb_dither_buf = dither_buf;
        }

        Ok(())
    }

    /// Copy the filtered row from cache to the final Y/U/V frame buffers.
    fn output_row_from_cache(&mut self, mby: usize, mbheight: usize, mbwidth: usize) {
        let luma_w = mbwidth * 16;
        let chroma_w = mbwidth * 8;
        let extra_y_rows = self.tables.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;
        let is_first_row = mby == 0;
        let is_last_row = mby == mbheight - 1;

        let (src_start_row, num_y_rows, dst_start_y_row) = if is_first_row && is_last_row {
            (extra_y_rows, 16usize, 0usize)
        } else if is_first_row {
            (extra_y_rows, 16 - extra_y_rows, 0usize)
        } else if is_last_row {
            (0, extra_y_rows + 16, mby * 16 - extra_y_rows)
        } else {
            (0, 16, mby * 16 - extra_y_rows)
        };

        // Copy Y
        {
            let src_start = src_start_row * self.cache_y_stride;
            let dst_start = dst_start_y_row * luma_w;
            let total = num_y_rows * luma_w;
            self.ybuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_y[src_start..src_start + total]);
        }

        // Copy U/V
        let (src_start_row_uv, num_uv_rows, dst_start_uv_row) = if is_first_row && is_last_row {
            (extra_uv_rows, 8usize, 0usize)
        } else if is_first_row {
            (extra_uv_rows, 8 - extra_uv_rows, 0usize)
        } else if is_last_row {
            (0, extra_uv_rows + 8, mby * 8 - extra_uv_rows)
        } else {
            (0, 8, mby * 8 - extra_uv_rows)
        };

        {
            let src_start = src_start_row_uv * self.cache_uv_stride;
            let dst_start = dst_start_uv_row * chroma_w;
            let total = num_uv_rows * chroma_w;
            self.ubuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_u[src_start..src_start + total]);
            self.vbuf[dst_start..dst_start + total]
                .copy_from_slice(&self.cache_v[src_start..src_start + total]);
        }
    }

    /// Copy bottom rows of current cache to extra area for next row's filtering.
    fn rotate_extra_rows(&mut self) {
        let extra_y_rows = self.tables.extra_y_rows;
        let extra_uv_rows = extra_y_rows / 2;

        if extra_y_rows == 0 {
            return;
        }

        let src_start = 16 * self.cache_y_stride;
        let copy_size = extra_y_rows * self.cache_y_stride;
        self.cache_y
            .copy_within(src_start..src_start + copy_size, 0);

        let src_start_uv = 8 * self.cache_uv_stride;
        let copy_size_uv = extra_uv_rows * self.cache_uv_stride;
        self.cache_u
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
        self.cache_v
            .copy_within(src_start_uv..src_start_uv + copy_size_uv, 0);
    }
}
