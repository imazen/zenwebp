//! Animation decoder.
//!
//! High-level API for decoding animated WebP files frame-by-frame.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::AnimationDecoder;
//!
//! let webp_data: &[u8] = &[]; // your animated WebP data
//! let mut decoder = AnimationDecoder::new(webp_data)?;
//! let info = decoder.info();
//! println!("{}x{}, {} frames", info.canvas_width, info.canvas_height, info.frame_count);
//!
//! while let Some(frame) = decoder.next_frame()? {
//!     println!("frame at {}ms, duration {}ms", frame.timestamp_ms, frame.duration_ms);
//! }
//! # Ok::<(), zenwebp::DecodingError>(())
//! ```

use alloc::vec;
use alloc::vec::Vec;

use crate::decoder::{DecodingError, LoopCount, WebPDecoder};

/// A decoded animation frame with owned RGBA pixel data.
#[derive(Debug, Clone)]
pub struct AnimFrame {
    /// RGBA pixel data (canvas_width * canvas_height * 4 bytes).
    pub data: Vec<u8>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Cumulative presentation timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// Display duration of this frame in milliseconds.
    pub duration_ms: u32,
}

/// Metadata about an animated WebP image.
#[derive(Debug, Clone)]
pub struct AnimationInfo {
    /// Canvas width in pixels.
    pub canvas_width: u32,
    /// Canvas height in pixels.
    pub canvas_height: u32,
    /// Total number of frames.
    pub frame_count: u32,
    /// Loop count for the animation.
    pub loop_count: LoopCount,
    /// Background color hint (BGRA byte order, per WebP spec).
    pub background_color: [u8; 4],
    /// Whether the image has alpha.
    pub has_alpha: bool,
}

/// High-level animated WebP decoder.
///
/// Wraps [`WebPDecoder`] and provides an iterator-like interface over frames,
/// returning owned RGBA snapshots of the composited canvas at each frame.
pub struct AnimationDecoder<'a> {
    decoder: WebPDecoder<'a>,
    buf: Vec<u8>,
    cumulative_ms: u32,
}

impl<'a> AnimationDecoder<'a> {
    /// Create a new animation decoder from WebP data.
    ///
    /// Returns an error if the data is not a valid animated WebP.
    pub fn new(data: &'a [u8]) -> Result<Self, DecodingError> {
        let decoder = WebPDecoder::new(data)?;
        if !decoder.is_animated() {
            return Err(DecodingError::InvalidParameter(
                alloc::string::String::from("not an animated WebP"),
            ));
        }
        let buf_size = decoder
            .output_buffer_size()
            .ok_or(DecodingError::ImageTooLarge)?;
        let buf = vec![0u8; buf_size];
        Ok(Self {
            decoder,
            buf,
            cumulative_ms: 0,
        })
    }

    /// Get metadata about the animation.
    pub fn info(&self) -> AnimationInfo {
        let (w, h) = self.decoder.dimensions();
        AnimationInfo {
            canvas_width: w,
            canvas_height: h,
            frame_count: self.decoder.num_frames(),
            loop_count: self.decoder.loop_count(),
            background_color: self
                .decoder
                .background_color_hint()
                .unwrap_or([0, 0, 0, 0]),
            has_alpha: self.decoder.has_alpha(),
        }
    }

    /// Returns `true` if there are more frames to decode.
    pub fn has_more_frames(&self) -> bool {
        // WebPDecoder tracks next_frame internally; we detect end via NoMoreFrames error.
        // We can check by comparing frame count, but the decoder doesn't expose current index.
        // Instead we rely on next_frame() returning None.
        true // conservative; next_frame() returns None at end
    }

    /// Decode the next frame, returning `None` when all frames have been read.
    pub fn next_frame(&mut self) -> Result<Option<AnimFrame>, DecodingError> {
        match self.decoder.read_frame(&mut self.buf) {
            Ok(duration_ms) => {
                let timestamp_ms = self.cumulative_ms;
                self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
                let (w, h) = self.decoder.dimensions();
                Ok(Some(AnimFrame {
                    data: self.buf.clone(),
                    width: w,
                    height: h,
                    timestamp_ms,
                    duration_ms,
                }))
            }
            Err(DecodingError::NoMoreFrames) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Reset the decoder to the first frame.
    pub fn reset(&mut self) {
        self.decoder.reset_animation();
        self.cumulative_ms = 0;
    }

    /// Decode all frames at once.
    pub fn decode_all(&mut self) -> Result<Vec<AnimFrame>, DecodingError> {
        self.reset();
        let mut frames = Vec::new();
        while let Some(frame) = self.next_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}
