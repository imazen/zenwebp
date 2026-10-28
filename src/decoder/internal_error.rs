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
    /// A cooperative stop token reported [`enough::StopReason::Cancelled`]
    /// (or a reason this crate does not know yet)
    Cancelled = 7,
    /// An intermediate allocation would exceed `Limits::max_memory`
    MemoryLimitExceeded = 8,
    /// A cooperative stop token reported [`enough::StopReason::TimedOut`]
    TimedOut = 9,
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
            InternalDecodeError::TimedOut => DecodeError::Cancelled(enough::StopReason::TimedOut),
            InternalDecodeError::MemoryLimitExceeded => DecodeError::MemoryLimitExceeded,
        }
    }
}

/// Each reason keeps its own variant, so the one-byte error still carries it.
impl From<enough::StopReason> for InternalDecodeError {
    fn from(reason: enough::StopReason) -> Self {
        match reason {
            enough::StopReason::TimedOut => Self::TimedOut,
            _ => Self::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::StopReason;

    /// A stop's reason survives the round trip through the hot-path error.
    #[test]
    fn stop_reasons_survive_the_internal_error() {
        for reason in [StopReason::Cancelled, StopReason::TimedOut] {
            let error = DecodeError::from(InternalDecodeError::from(reason));
            assert!(
                matches!(error, DecodeError::Cancelled(r) if r == reason),
                "{reason:?} became {error:?}"
            );
        }
    }
}
