//! Animation encoder.
//!
//! High-level API for encoding animated WebP files frame-by-frame.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::mux::{AnimationEncoder, AnimationConfig};
//! use zenwebp::{EncoderConfig, ColorType, LoopCount};
//!
//! let config = AnimationConfig {
//!     background_color: [0, 0, 0, 0],
//!     loop_count: LoopCount::Forever,
//! };
//! let mut anim = AnimationEncoder::new(320, 240, config)?;
//!
//! let frame_config = EncoderConfig::new().quality(75.0);
//!
//! // Frame pixels (320x240 RGBA)
//! let pixels = vec![255u8; 320 * 240 * 4];
//! anim.add_frame(&pixels, ColorType::Rgba8, 0, &frame_config)?;
//! anim.add_frame(&pixels, ColorType::Rgba8, 100, &frame_config)?;
//!
//! let webp = anim.finalize(100)?;
//! # Ok::<(), zenwebp::mux::MuxError>(())
//! ```

use alloc::vec::Vec;

use super::assemble::{MuxFrame, WebPMux};
use super::demux::{BlendMethod, DisposeMethod};
use super::error::MuxError;
use crate::decoder::LoopCount;
use crate::encoder::vp8::encode_frame_lossy;
use crate::encoder::{
    encode_alpha_lossless, encode_frame_lossless, ColorType, EncoderConfig, EncoderParams,
    NoProgress,
};
use enough::Unstoppable;

/// Configuration for an animated WebP.
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Background color in BGRA byte order (as per the WebP spec).
    pub background_color: [u8; 4],
    /// Loop count for the animation.
    pub loop_count: LoopCount,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            background_color: [0, 0, 0, 0],
            loop_count: LoopCount::Forever,
        }
    }
}

/// Pending frame waiting for the next timestamp to compute its duration.
struct PendingFrame {
    mux_frame: MuxFrame,
    timestamp_ms: u32,
}

/// Animated WebP encoder.
///
/// Encodes individual frames using the existing VP8/VP8L encoder and assembles
/// them into an animated WebP container.
pub struct AnimationEncoder {
    width: u32,
    height: u32,
    mux: WebPMux,
    pending: Option<PendingFrame>,
}

impl AnimationEncoder {
    /// Create a new animation encoder.
    ///
    /// The canvas dimensions must be between 1 and 16384 (inclusive).
    pub fn new(width: u32, height: u32, config: AnimationConfig) -> Result<Self, MuxError> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(MuxError::InvalidDimensions { width, height });
        }
        let mut mux = WebPMux::new(width, height);
        mux.set_animation(config.background_color, config.loop_count);
        Ok(Self {
            width,
            height,
            mux,
            pending: None,
        })
    }

    /// Add a frame to the animation.
    ///
    /// Each frame is encoded using the provided [`EncoderConfig`]. The
    /// `timestamp_ms` is the presentation time of this frame; the previous
    /// frame's duration is computed as `this_timestamp - previous_timestamp`.
    ///
    /// The first frame's timestamp is typically 0.
    ///
    /// # Frame options
    ///
    /// This method encodes a full-canvas frame with default dispose/blend
    /// settings (`DisposeMethod::Background` / `BlendMethod::Overwrite`).
    /// For sub-frame control, use [`add_frame_advanced`](Self::add_frame_advanced).
    pub fn add_frame(
        &mut self,
        pixels: &[u8],
        color_type: ColorType,
        timestamp_ms: u32,
        encoder_config: &EncoderConfig,
    ) -> Result<(), MuxError> {
        self.add_frame_advanced(
            pixels,
            color_type,
            self.width,
            self.height,
            0,
            0,
            timestamp_ms,
            encoder_config,
            DisposeMethod::Background,
            BlendMethod::Overwrite,
        )
    }

    /// Add a frame with full control over placement, dispose, and blend.
    ///
    /// Frame offsets must be even. The frame region must fit within the canvas.
    #[allow(clippy::too_many_arguments)]
    pub fn add_frame_advanced(
        &mut self,
        pixels: &[u8],
        color_type: ColorType,
        frame_width: u32,
        frame_height: u32,
        x_offset: u32,
        y_offset: u32,
        timestamp_ms: u32,
        encoder_config: &EncoderConfig,
        dispose: DisposeMethod,
        blend: BlendMethod,
    ) -> Result<(), MuxError> {
        // Flush previous pending frame
        if let Some(prev) = self.pending.take() {
            let duration = timestamp_ms.saturating_sub(prev.timestamp_ms);
            let mut frame = prev.mux_frame;
            frame.duration_ms = duration;
            self.mux.push_frame(frame)?;
        }

        // Encode this frame
        let params = encoder_config.to_params();
        let encoded = encode_frame_data(pixels, frame_width, frame_height, color_type, &params)?;

        let mux_frame = MuxFrame {
            x_offset,
            y_offset,
            width: frame_width,
            height: frame_height,
            duration_ms: 0, // will be set when next frame arrives or on finalize
            dispose,
            blend,
            bitstream: encoded.bitstream,
            alpha_data: encoded.alpha_data,
            is_lossless: encoded.is_lossless,
        };

        self.pending = Some(PendingFrame {
            mux_frame,
            timestamp_ms,
        });

        Ok(())
    }

    /// Finalize the animation and return the assembled WebP bytes.
    ///
    /// `last_frame_duration_ms` is the display duration for the final frame,
    /// since there is no subsequent timestamp to derive it from.
    pub fn finalize(mut self, last_frame_duration_ms: u32) -> Result<Vec<u8>, MuxError> {
        // Flush the last pending frame
        if let Some(prev) = self.pending.take() {
            let mut frame = prev.mux_frame;
            frame.duration_ms = last_frame_duration_ms;
            self.mux.push_frame(frame)?;
        }

        self.mux.assemble()
    }

    /// Set ICC profile on the output.
    pub fn set_icc_profile(&mut self, data: Vec<u8>) {
        self.mux.set_icc_profile(data);
    }

    /// Set EXIF metadata on the output.
    pub fn set_exif(&mut self, data: Vec<u8>) {
        self.mux.set_exif(data);
    }

    /// Set XMP metadata on the output.
    pub fn set_xmp(&mut self, data: Vec<u8>) {
        self.mux.set_xmp(data);
    }
}

/// Encoded frame data: bitstream, optional alpha data, and whether it's lossless.
struct EncodedFrame {
    bitstream: Vec<u8>,
    alpha_data: Option<Vec<u8>>,
    is_lossless: bool,
}

/// Encode a single frame to raw bitstream data (no RIFF container).
fn encode_frame_data(
    pixels: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    params: &EncoderParams,
) -> Result<EncodedFrame, MuxError> {
    let mut bitstream = Vec::new();
    let lossy_with_alpha = params.use_lossy && color.has_alpha();

    if params.use_lossy {
        encode_frame_lossy(
            &mut bitstream,
            pixels,
            width,
            height,
            color,
            params,
            &Unstoppable,
            &NoProgress,
        )?;

        let alpha_data = if lossy_with_alpha {
            let mut alpha = Vec::new();
            encode_alpha_lossless(
                &mut alpha,
                pixels,
                width,
                height,
                color,
                params.alpha_quality,
            )?;
            Some(alpha)
        } else {
            None
        };

        Ok(EncodedFrame {
            bitstream,
            alpha_data,
            is_lossless: false,
        })
    } else {
        encode_frame_lossless(
            &mut bitstream,
            pixels,
            width,
            height,
            color,
            params.clone(),
            false,
        )?;
        Ok(EncodedFrame {
            bitstream,
            alpha_data: None,
            is_lossless: true,
        })
    }
}
