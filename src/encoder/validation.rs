//! Configuration validation.
//!
//! Each public `Config` type has a `validate()` method that returns
//! `Result<(), ValidationError>`. The encode entry points keep their
//! existing clamping behaviour for backwards compatibility — `validate()`
//! is an opt-in fail-fast check for batch-job callers who would rather
//! get a typed error than have the encoder silently change their inputs.
//!
//! # Why opt-in?
//!
//! `LossyConfig::with_quality(150.0)` clamps to `100.0` today. Callers
//! who care about matching the requested config (calibration sweeps,
//! provenance-bound pipelines, training-data generation) need to know
//! when they asked for something out-of-range, but downstream consumers
//! that have shipped against the clamping contract for years don't.
//! `validate()` lets the picky callers opt in without breaking the rest.
//!
//! # What we validate
//!
//! Range checks for every primitive field with a documented valid range,
//! plus a small number of cross-parameter invariants that the encoder
//! cannot represent in its types:
//!
//! - **`LossyConfig`**: `target_size` and `target_psnr` are mutually
//!   exclusive (the encoder picks one path; combining them is
//!   undefined). When the unstable `target-zensim` cargo feature is
//!   enabled, `target_zensim` joins the same exclusivity rule.
//!   Validated in [`LossyConfig::validate`].
//! - **`SharpYuvConfig`** (when set on `LossyConfig`): `convergence_threshold`
//!   must be finite and non-negative.
//! - **`LosslessConfig`**: simple range checks; no cross-param invariants.
//!
//! Spec references: VP8/VP8L parameter ranges follow libwebp's
//! `WebPConfigInit` and `VP8EncProc` (effort/method 0..=6, sns/filter
//! 0..=100, sharpness 0..=7, segments 1..=4, quality 0..=100). PSNR
//! upper bound (80 dB) reflects libwebp's documented sane cap; values
//! above that are practically unreachable for natural content.

use core::ops::RangeInclusive;

use thiserror::Error;

/// Invalid configuration. Returned by `validate()` methods on the
/// public `Config` types.
///
/// Marked `#[non_exhaustive]` because new validation rules will be
/// added over time as additional config fields gain documented ranges
/// or new cross-parameter invariants are discovered.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ValidationError {
    /// `quality` is outside the valid range.
    #[error("quality {value} out of valid range {valid:?}")]
    QualityOutOfRange {
        /// The invalid value.
        value: f32,
        /// The valid inclusive range.
        valid: RangeInclusive<f32>,
    },

    /// `quality` is not a finite number (NaN or infinity).
    #[error("quality must be finite, got {value}")]
    QualityNotFinite {
        /// The invalid value.
        value: f32,
    },

    /// `method` is outside the valid range.
    #[error("method {value} out of valid range {valid:?}")]
    MethodOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `alpha_quality` is outside the valid range.
    #[error("alpha_quality {value} out of valid range {valid:?}")]
    AlphaQualityOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `sns_strength` is outside the valid range.
    #[error("sns_strength {value} out of valid range {valid:?}")]
    SnsStrengthOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `filter_strength` is outside the valid range.
    #[error("filter_strength {value} out of valid range {valid:?}")]
    FilterStrengthOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `filter_sharpness` is outside the valid range.
    #[error("filter_sharpness {value} out of valid range {valid:?}")]
    FilterSharpnessOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `segments` is outside the valid range.
    #[error("segments {value} out of valid range {valid:?}")]
    SegmentsOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `partition_limit` is outside the valid range.
    #[error("partition_limit {value} out of valid range {valid:?}")]
    PartitionLimitOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `target_psnr` is outside the valid range or non-finite.
    #[error("target_psnr {value} out of valid range {valid:?}")]
    TargetPsnrOutOfRange {
        /// The invalid value.
        value: f32,
        /// The valid inclusive range.
        valid: RangeInclusive<f32>,
    },

    /// `target_zensim.target` is outside the valid range or non-finite.
    ///
    /// Only emitted when the unstable `target-zensim` cargo feature is
    /// enabled.
    #[cfg(feature = "target-zensim")]
    #[error("target_zensim.target {value} out of valid range {valid:?}")]
    TargetZensimOutOfRange {
        /// The invalid value.
        value: f32,
        /// The valid inclusive range.
        valid: RangeInclusive<f32>,
    },

    /// `target_zensim.max_passes` is zero (must run at least one pass).
    ///
    /// Only emitted when the unstable `target-zensim` cargo feature is
    /// enabled.
    #[cfg(feature = "target-zensim")]
    #[error("target_zensim.max_passes must be >= 1, got {value}")]
    TargetZensimMaxPassesZero {
        /// The invalid value.
        value: u8,
    },

    /// A tolerance field on `target_zensim` is non-finite or negative.
    ///
    /// Only emitted when the unstable `target-zensim` cargo feature is
    /// enabled.
    #[cfg(feature = "target-zensim")]
    #[error("target_zensim.{field} must be finite and >= 0.0, got {value}")]
    TargetZensimToleranceInvalid {
        /// Which tolerance field (`max_overshoot`, `max_undershoot`,
        /// `max_undershoot_ship`).
        field: &'static str,
        /// The invalid value.
        value: f32,
    },

    /// More than one mutually exclusive target was set. Only one target
    /// mode may be active per encode.
    ///
    /// Without the `target-zensim` cargo feature this only covers the
    /// (`target_size`, `target_psnr`) pair. With the feature enabled,
    /// `target_zensim` joins the same exclusivity rule.
    #[error("targets {first} and {second} are mutually exclusive")]
    TargetMutuallyExclusive {
        /// First target field that was set.
        first: &'static str,
        /// Second target field that was set.
        second: &'static str,
    },

    /// `near_lossless` is outside the valid range (lossless only).
    #[error("near_lossless {value} out of valid range {valid:?}")]
    NearLosslessOutOfRange {
        /// The invalid value.
        value: u8,
        /// The valid inclusive range.
        valid: RangeInclusive<u8>,
    },

    /// `SharpYuvConfig::convergence_threshold` is non-finite or negative.
    #[error("sharp_yuv.convergence_threshold must be finite and >= 0.0, got {value}")]
    SharpYuvConvergenceThresholdInvalid {
        /// The invalid value.
        value: f32,
    },
}

// Codec-agnostic error taxonomy (zencodec PR #103) so consumers can route on the
// coarse category without naming this enum.
impl zencodec::CategorizedError for ValidationError {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenwebp")
    }

    fn category(&self) -> zencodec::ErrorCategory {
        // Every `ValidationError` variant reports a caller-supplied configuration
        // value that failed a documented range or cross-parameter invariant — the
        // type's entire contract is "invalid configuration", so the whole type
        // maps to `InvalidParameters` (future variants are validation failures by
        // construction and share this mapping).
        zencodec::ErrorCategory::InvalidParameters
    }
}

// ============================================================================
// Range constants. Centralised so callers can also introspect the
// valid range without round-tripping through `validate()`.
// ============================================================================

/// Valid range for `quality` on lossy and lossless configs.
pub const QUALITY_RANGE: RangeInclusive<f32> = 0.0..=100.0;
/// Valid range for `method` on lossy and lossless configs (libwebp parity).
pub const METHOD_RANGE: RangeInclusive<u8> = 0..=6;
/// Valid range for `alpha_quality`.
pub const ALPHA_QUALITY_RANGE: RangeInclusive<u8> = 0..=100;
/// Valid range for `sns_strength`.
pub const SNS_STRENGTH_RANGE: RangeInclusive<u8> = 0..=100;
/// Valid range for `filter_strength`.
pub const FILTER_STRENGTH_RANGE: RangeInclusive<u8> = 0..=100;
/// Valid range for `filter_sharpness` (VP8 loop filter sharpness field is 3 bits).
pub const FILTER_SHARPNESS_RANGE: RangeInclusive<u8> = 0..=7;
/// Valid range for `segments` (VP8 supports 1-4 segments).
pub const SEGMENTS_RANGE: RangeInclusive<u8> = 1..=4;
/// Valid range for `partition_limit` (zenwebp-internal mode-decision lever).
pub const PARTITION_LIMIT_RANGE: RangeInclusive<u8> = 0..=100;
/// Valid range for `target_psnr` (dB). The lower bound is 0 dB; the
/// upper bound (80 dB) reflects libwebp's documented sane cap.
pub const TARGET_PSNR_RANGE: RangeInclusive<f32> = 0.0..=80.0;
/// Valid range for `target_zensim.target` (zensim score; same scale as
/// the metric's own 0..=100 output).
///
/// Only available when the unstable `target-zensim` cargo feature is
/// enabled.
#[cfg(feature = "target-zensim")]
pub const TARGET_ZENSIM_RANGE: RangeInclusive<f32> = 0.0..=100.0;
/// Valid range for `near_lossless`. 100 = off, 0 = max preprocessing
/// (libwebp's `WebPConfig.near_lossless`).
pub const NEAR_LOSSLESS_RANGE: RangeInclusive<u8> = 0..=100;

// ============================================================================
// Helper checks
// ============================================================================

#[inline]
pub(super) fn check_quality(q: f32) -> Result<(), ValidationError> {
    if !q.is_finite() {
        return Err(ValidationError::QualityNotFinite { value: q });
    }
    if !QUALITY_RANGE.contains(&q) {
        return Err(ValidationError::QualityOutOfRange {
            value: q,
            valid: QUALITY_RANGE,
        });
    }
    Ok(())
}

#[inline]
pub(super) fn check_method(m: u8) -> Result<(), ValidationError> {
    if !METHOD_RANGE.contains(&m) {
        return Err(ValidationError::MethodOutOfRange {
            value: m,
            valid: METHOD_RANGE,
        });
    }
    Ok(())
}

#[inline]
pub(super) fn check_alpha_quality(a: u8) -> Result<(), ValidationError> {
    if !ALPHA_QUALITY_RANGE.contains(&a) {
        return Err(ValidationError::AlphaQualityOutOfRange {
            value: a,
            valid: ALPHA_QUALITY_RANGE,
        });
    }
    Ok(())
}

#[inline]
pub(super) fn check_target_psnr(p: f32) -> Result<(), ValidationError> {
    // 0.0 means "disabled" — accept it without further checks.
    if p == 0.0 {
        return Ok(());
    }
    if !p.is_finite() || !TARGET_PSNR_RANGE.contains(&p) {
        return Err(ValidationError::TargetPsnrOutOfRange {
            value: p,
            valid: TARGET_PSNR_RANGE,
        });
    }
    Ok(())
}

#[cfg(test)]
mod validation_category_tests {
    use super::ValidationError;
    use zencodec::{CategorizedError, ErrorCategory as C};

    #[test]
    fn validation_error_category_is_invalid_parameters() {
        assert_eq!(
            ValidationError::QualityNotFinite { value: f32::NAN }.codec_name(),
            Some("zenwebp")
        );
        assert_eq!(
            ValidationError::QualityOutOfRange {
                value: 150.0,
                valid: 0.0..=100.0,
            }
            .category(),
            C::InvalidParameters
        );
        assert_eq!(
            ValidationError::TargetMutuallyExclusive {
                first: "target_size",
                second: "target_psnr",
            }
            .category(),
            C::InvalidParameters
        );

        // The At<E> blanket impl forwards both category and codec name.
        let traced = whereat::at!(ValidationError::QualityNotFinite { value: f32::NAN });
        assert_eq!(traced.category(), C::InvalidParameters);
        assert_eq!(traced.codec_name(), Some("zenwebp"));
    }
}
