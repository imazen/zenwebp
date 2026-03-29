//! Animation decoding support for the v2 decoder.
//!
//! Provides [`DecoderContext::decode_animation`] which uses the
//! [`WebPDemuxer`](crate::mux::WebPDemuxer) to parse the container and
//! decodes each lossy VP8 frame through a single reused `DecoderContext`.
//! Lossless VP8L frames are decoded via [`LosslessDecoder`].

use alloc::vec;
use alloc::vec::Vec;

use crate::decoder::api::DecodeError;
use crate::decoder::extended::{get_alpha_predictor, read_alpha_chunk};
use crate::decoder::lossless::LosslessDecoder;
use crate::mux::{BlendMethod, DemuxFrame, DisposeMethod, MuxError, WebPDemuxer};

use super::DecoderContext;

/// Metadata for a composited animation frame delivered to the callback.
pub struct AnimationFrame<'a> {
    /// Composited RGBA canvas pixels for this frame.
    pub pixels: &'a [u8],
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// 1-based frame number.
    pub frame_num: u32,
    /// Horizontal offset of the sub-frame on the canvas.
    pub x_offset: u32,
    /// Vertical offset of the sub-frame on the canvas.
    pub y_offset: u32,
    /// Frame duration in milliseconds.
    pub duration_ms: u32,
    /// Cumulative timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// How the frame area was disposed before rendering.
    pub dispose: DisposeMethod,
    /// How the frame was blended onto the canvas.
    pub blend: BlendMethod,
}

impl DecoderContext {
    /// Decode an animated WebP file, calling the callback for each composited frame.
    ///
    /// Uses the [`WebPDemuxer`] to parse the container and decodes each frame
    /// through this `DecoderContext` (reusing buffers across lossy frames).
    /// Lossless VP8L frames are decoded via `LosslessDecoder`.
    ///
    /// The callback receives an [`AnimationFrame`] with the composited RGBA
    /// canvas. Return `true` to continue decoding, `false` to stop early.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use zenwebp::DecoderContext;
    ///
    /// let webp_data: &[u8] = &[]; // animated WebP data
    /// let mut ctx = DecoderContext::new();
    /// ctx.decode_animation(webp_data, |frame| {
    ///     println!("frame {} at {}ms: {}x{}",
    ///         frame.frame_num, frame.timestamp_ms, frame.width, frame.height);
    ///     true // continue
    /// }).unwrap();
    /// ```
    pub fn decode_animation(
        &mut self,
        data: &[u8],
        mut callback: impl FnMut(AnimationFrame<'_>) -> bool,
    ) -> Result<(), DecodeError> {
        let demuxer = WebPDemuxer::new(data).map_err(mux_to_decode)?;

        if !demuxer.is_animated() {
            return Err(DecodeError::InvalidParameter("not an animated WebP".into()));
        }

        let canvas_w = demuxer.canvas_width();
        let canvas_h = demuxer.canvas_height();
        let bg_color = demuxer.background_color();

        let canvas_size: usize = (canvas_w as usize)
            .checked_mul(canvas_h as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or(DecodeError::ImageTooLarge)?;

        let mut canvas = vec![0u8; canvas_size];
        // Initialize canvas with background color
        for pixel in canvas.chunks_exact_mut(4) {
            pixel.copy_from_slice(&bg_color);
        }

        let mut frame_scratch = Vec::new();
        let mut prev_x = 0u32;
        let mut prev_y = 0u32;
        let mut prev_w = 0u32;
        let mut prev_h = 0u32;
        let mut dispose_current = false;
        let mut timestamp_ms = 0u32;

        for demux_frame in demuxer.frames() {
            // Dispose previous frame if needed
            if dispose_current {
                clear_rect(
                    &mut canvas,
                    canvas_w,
                    &bg_color,
                    prev_x,
                    prev_y,
                    prev_w,
                    prev_h,
                );
            }

            // Decode this frame's pixels
            let frame_has_alpha = self.decode_single_frame(&demux_frame, &mut frame_scratch)?;

            // Composite onto canvas
            crate::decoder::extended::composite_frame(
                &mut canvas,
                canvas_w,
                canvas_h,
                None, // clearing was already handled above
                &frame_scratch,
                demux_frame.x_offset,
                demux_frame.y_offset,
                demux_frame.width,
                demux_frame.height,
                frame_has_alpha,
                demux_frame.blend == BlendMethod::AlphaBlend,
                prev_w,
                prev_h,
                prev_x,
                prev_y,
            )?;

            // Deliver frame to callback
            let frame = AnimationFrame {
                pixels: &canvas,
                width: canvas_w,
                height: canvas_h,
                frame_num: demux_frame.frame_num,
                x_offset: demux_frame.x_offset,
                y_offset: demux_frame.y_offset,
                duration_ms: demux_frame.duration_ms,
                timestamp_ms,
                dispose: demux_frame.dispose,
                blend: demux_frame.blend,
            };

            let should_continue = callback(frame);

            // Update state for next frame
            timestamp_ms = timestamp_ms.saturating_add(demux_frame.duration_ms);
            prev_x = demux_frame.x_offset;
            prev_y = demux_frame.y_offset;
            prev_w = demux_frame.width;
            prev_h = demux_frame.height;
            dispose_current = demux_frame.dispose == DisposeMethod::Background;

            if !should_continue {
                break;
            }
        }

        Ok(())
    }

    /// Decode a single demuxed frame into the scratch buffer.
    ///
    /// Returns whether the frame has alpha data.
    fn decode_single_frame(
        &mut self,
        frame: &DemuxFrame<'_>,
        scratch: &mut Vec<u8>,
    ) -> Result<bool, DecodeError> {
        if frame.is_lossy {
            if let Some(alpha_data) = frame.alpha_data {
                // Lossy with alpha: decode VP8 to RGBA, then apply alpha
                let (_w, _h) = self.decode_to_rgb(frame.bitstream, scratch, 4)?;

                let alpha_w: u16 = frame
                    .width
                    .try_into()
                    .map_err(|_| DecodeError::ImageTooLarge)?;
                let alpha_h: u16 = frame
                    .height
                    .try_into()
                    .map_err(|_| DecodeError::ImageTooLarge)?;
                let alpha_chunk = read_alpha_chunk(alpha_data, alpha_w, alpha_h)?;

                let fw = frame.width as usize;
                let fh = frame.height as usize;
                for y in 0..fh {
                    for x in 0..fw {
                        let predictor =
                            get_alpha_predictor(x, y, fw, alpha_chunk.filtering_method, scratch);

                        let alpha_index = y * fw + x;
                        let buffer_index = alpha_index * 4 + 3;
                        scratch[buffer_index] =
                            predictor.wrapping_add(alpha_chunk.data[alpha_index]);
                    }
                }

                Ok(true)
            } else {
                // Lossy without alpha: decode VP8 to RGB
                self.decode_to_rgb(frame.bitstream, scratch, 3)?;
                Ok(false)
            }
        } else {
            // Lossless VP8L
            let alloc_size = (frame.width as usize)
                .checked_mul(frame.height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or(DecodeError::ImageTooLarge)?;
            scratch.resize(alloc_size, 0);
            let mut decoder = LosslessDecoder::new(frame.bitstream);
            decoder.decode_frame(frame.width, frame.height, false, scratch)?;
            Ok(true) // VP8L can always carry alpha
        }
    }
}

/// Clear a rectangle on the RGBA canvas to the background color.
fn clear_rect(
    canvas: &mut [u8],
    canvas_w: u32,
    bg_color: &[u8; 4],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    // canvas_w is at most 16383, so stride fits usize on all platforms.
    let stride = canvas_w as usize * 4;
    for row in y..y.saturating_add(h) {
        let row_start = row as usize * stride + x as usize * 4;
        for px in 0..w as usize {
            let offset = row_start + px * 4;
            if offset + 4 <= canvas.len() {
                canvas[offset..offset + 4].copy_from_slice(bg_color);
            }
        }
    }
}

/// Convert a `MuxError` to a `DecodeError`.
fn mux_to_decode(e: whereat::At<MuxError>) -> DecodeError {
    DecodeError::InvalidParameter(alloc::format!("demux error: {e}"))
}
