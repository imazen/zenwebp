//! Error type for [`crate::recompress`].

use thiserror::Error;
use zenwebp::detect::ProbeError;

/// Top-level error for [`crate::recompress`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The input bytes are not a parseable WebP file.
    #[error("invalid WebP input: {0}")]
    InvalidInput(#[from] ProbeError),

    /// `target_zensim_a` is outside `[0.0, 100.0]` or is NaN.
    #[error("target_zensim_a={0} out of [0.0, 100.0]")]
    TargetOutOfRange(f32),

    /// The source is animated; 0.1.x routes animated input through
    /// [`crate::StrategyKind::LosslessRemux`] only. This variant is reserved
    /// for when the caller explicitly forbids that fallback.
    #[error("animated WebP not supported in lossy recompression")]
    AnimationNotSupported,

    /// The decoder failed mid-stream. Inner error is opaque (zenwebp's
    /// decoder error type is not part of zenwebp-recompress's public API).
    #[error("WebP decode failed: {0}")]
    DecodeFailed(String),

    /// The encoder failed. Inner error is opaque.
    #[error("WebP encode failed: {0}")]
    EncodeFailed(String),

    /// The calibration table is missing a cell required to dispatch a
    /// strategy. This is a defect in the table, not a user error.
    #[error("calibration table missing cell for {0}")]
    CalibrationMissing(&'static str),

    /// Generic I/O error from the surrounding harness. Library code never
    /// produces this directly; it exists so [`std::io::Error`] can be
    /// propagated when callers iterate.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The cancellation token signaled cancel before work completed.
    #[error("operation cancelled")]
    Cancelled,
}
