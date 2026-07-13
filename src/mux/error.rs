//! Error types for mux/demux operations.

use alloc::string::String;
use thiserror::Error;

use crate::decoder::DecodeError;
use crate::encoder::EncodeError;

/// Result type alias using `At<MuxError>` for automatic location tracking.
pub type MuxResult<T> = core::result::Result<T, whereat::At<MuxError>>;

/// Errors that can occur during mux/demux operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MuxError {
    /// The data is not a valid WebP file.
    #[error("Invalid WebP format: {0}")]
    InvalidFormat(String),

    /// Frame dimensions are invalid (zero, too large, or don't fit the canvas).
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// The invalid width.
        width: u32,
        /// The invalid height.
        height: u32,
    },

    /// A frame index is out of bounds.
    #[error("Frame {index} out of bounds (total: {total})")]
    FrameOutOfBounds {
        /// The requested frame index.
        index: u32,
        /// The total number of frames.
        total: u32,
    },

    /// An error occurred during encoding.
    #[error("Encoding error: {0}")]
    EncodeError(#[from] EncodeError),

    /// An error occurred during decoding/parsing.
    #[error("Decoding error: {0}")]
    DecodeError(#[from] DecodeError),

    /// No frames were added before assembly.
    #[error("No frames to assemble")]
    NoFrames,

    /// Frame offset is not a multiple of 2 (WebP spec requirement).
    #[error("Frame offset must be even: ({x}, {y})")]
    OddFrameOffset {
        /// The invalid x offset.
        x: u32,
        /// The invalid y offset.
        y: u32,
    },

    /// Frame extends beyond the canvas boundary.
    #[error(
        "Frame at ({x}, {y}) size {width}x{height} exceeds canvas {canvas_width}x{canvas_height}"
    )]
    FrameOutsideCanvas {
        /// Frame x offset.
        x: u32,
        /// Frame y offset.
        y: u32,
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Canvas width.
        canvas_width: u32,
        /// Canvas height.
        canvas_height: u32,
    },
}

// Codec-agnostic error taxonomy (zencodec PR #103/#116, origin-first two-level
// reshape). Maps every `MuxError` variant to exactly one coarse
// `ErrorCategory` so consumers can route on the category (HTTP status, retry
// policy, logging) without naming this enum. The wrapped
// `EncodeError`/`DecodeError` arms delegate to their own mappings.
impl zencodec::CategorizedError for MuxError {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenwebp")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        use zencodec::ErrorCategory as C;
        use zencodec::{ImageError as IE, InvalidKind as IK, RequestError as RE};
        match self {
            // Demux/parse failure: the bytes are not a valid WebP container.
            MuxError::InvalidFormat(_) => C::Image(IE::Malformed),

            // Assembly-side caller parameters: the caller-supplied frame
            // geometry/offsets violate the WebP spec when building an animation.
            MuxError::InvalidDimensions { .. }
            | MuxError::OddFrameOffset { .. }
            | MuxError::FrameOutsideCanvas { .. }
            // Caller asked for a frame index past the end of the sequence.
            | MuxError::FrameOutOfBounds { .. } => C::Request(RE::Invalid(IK::Parameters)),

            // API-protocol violation: assembly requested before any frame added.
            MuxError::NoFrames => C::Request(RE::Invalid(IK::State)),

            // Delegate to the wrapped codec error's own mapping.
            MuxError::EncodeError(e) => e.category(),
            MuxError::DecodeError(e) => e.category(),
        }
    }
}

#[cfg(test)]
mod mux_category_tests {
    use super::MuxError;
    use crate::decoder::DecodeError;
    use crate::encoder::EncodeError;
    use zencodec::{
        CategorizedError, ErrorCategory as C, ImageError as IE, InvalidKind as IK,
        RequestError as RE,
    };

    #[test]
    fn mux_error_category_mapping() {
        assert_eq!(MuxError::NoFrames.codec_name(), Some("zenwebp"));

        // Demux parse failure.
        assert_eq!(
            MuxError::InvalidFormat("bad".into()).category(),
            C::Image(IE::Malformed)
        );

        // Assembly-side caller parameters.
        assert_eq!(
            MuxError::InvalidDimensions {
                width: 0,
                height: 0
            }
            .category(),
            C::Request(RE::Invalid(IK::Parameters))
        );
        assert_eq!(
            MuxError::OddFrameOffset { x: 1, y: 1 }.category(),
            C::Request(RE::Invalid(IK::Parameters))
        );
        assert_eq!(
            MuxError::FrameOutsideCanvas {
                x: 0,
                y: 0,
                width: 9,
                height: 9,
                canvas_width: 4,
                canvas_height: 4,
            }
            .category(),
            C::Request(RE::Invalid(IK::Parameters))
        );
        assert_eq!(
            MuxError::FrameOutOfBounds { index: 9, total: 2 }.category(),
            C::Request(RE::Invalid(IK::Parameters))
        );

        // API-protocol violation.
        assert_eq!(
            MuxError::NoFrames.category(),
            C::Request(RE::Invalid(IK::State))
        );

        // Delegation to the wrapped codec error's own mapping.
        assert_eq!(
            MuxError::EncodeError(EncodeError::InvalidDimensions).category(),
            C::Request(RE::Invalid(IK::Parameters))
        );
        assert_eq!(
            MuxError::DecodeError(DecodeError::HuffmanError).category(),
            C::Image(IE::Malformed)
        );

        // The At<E> blanket impl forwards both category and codec name.
        let traced = whereat::at!(MuxError::NoFrames);
        assert_eq!(traced.category(), C::Request(RE::Invalid(IK::State)));
        assert_eq!(traced.codec_name(), Some("zenwebp"));
    }
}
