//! Lightweight error type for decoder hot paths.
//!
//! [`InternalDecodeError`] is a `#[repr(u8)]` enum with no heap allocations,
//! keeping `Result<T, InternalDecodeError>` at 2 bytes for most `T`. This
//! avoids the 32+ byte stack bloat of [`super::api::DecodeError`] (which
//! contains `String` variants) in every `?` operator inside the decode loop.
//!
//! Convert to [`super::api::DecodeError`] at the API boundary via `From`.

use super::api::DecodeError;

/// Compact error type for internal decoder operations.
///
/// No heap allocation, no drop glue. Enum discriminant fits in a single byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum InternalDecodeError {
    /// Bitstream is corrupt or truncated
    BitStreamError = 0,
    /// Invalid Huffman code encountered
    HuffmanError = 1,
    /// Invalid lossless transform
    TransformError = 2,
    /// Invalid color cache bits
    InvalidColorCacheBits = 3,
    /// Luma prediction mode invalid
    #[allow(dead_code)]
    LumaPredictionModeInvalid = 4,
    /// Intra prediction mode invalid
    #[allow(dead_code)]
    IntraPredictionModeInvalid = 5,
    /// Chroma prediction mode invalid
    #[allow(dead_code)]
    ChromaPredictionModeInvalid = 6,
    /// Decoding was cancelled via a cooperative stop token
    Cancelled = 7,
}

impl From<InternalDecodeError> for DecodeError {
    fn from(e: InternalDecodeError) -> Self {
        match e {
            InternalDecodeError::BitStreamError => DecodeError::BitStreamError,
            InternalDecodeError::HuffmanError => DecodeError::HuffmanError,
            InternalDecodeError::TransformError => DecodeError::TransformError,
            InternalDecodeError::InvalidColorCacheBits => {
                // We lose the specific bits value in the conversion, but this
                // error is only constructed with the value at the call site in
                // lossless.rs where we can use DecodeError directly (not hot path).
                DecodeError::BitStreamError
            }
            InternalDecodeError::LumaPredictionModeInvalid => {
                DecodeError::LumaPredictionModeInvalid(0)
            }
            InternalDecodeError::IntraPredictionModeInvalid => {
                DecodeError::IntraPredictionModeInvalid(0)
            }
            InternalDecodeError::ChromaPredictionModeInvalid => {
                DecodeError::ChromaPredictionModeInvalid(0)
            }
            InternalDecodeError::Cancelled => DecodeError::Cancelled(enough::StopReason::Cancelled),
        }
    }
}

impl From<enough::StopReason> for InternalDecodeError {
    fn from(_reason: enough::StopReason) -> Self {
        Self::Cancelled
    }
}
