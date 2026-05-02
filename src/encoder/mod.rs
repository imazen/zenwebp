//! WebP encoder implementation.
//!
//! This module provides lossy (VP8) and lossless (VP8L) WebP encoding with
//! perceptual optimizations for improved rate-quality trade-offs.
//!
//! # Module Organization
//!
//! **Core encoding:**
//! - [`vp8`]: Lossy VP8 encoding (DCT-based, like JPEG)
//! - [`vp8l`]: Lossless VP8L encoding (LZ77 + entropy coding)
//!
//! **Rate-distortion optimization:**
//! - [`analysis`]: Segment-based adaptive quantization (k-means clustering)
//! - [`cost`]: RD cost estimation for mode selection
//! - trellis: Trellis quantization for optimal coefficient selection
//!
//! **Perceptual models:**
//! - psy: CSF weighting, JND thresholds, masking-based AQ
//!
//! **Low-level utilities:**
//! - [`quantize`]: Quantization matrices and coefficient quantization
//! - [`tables`]: Lookup tables (entropy costs, zigzag order, etc.)
//! - arithmetic: Arithmetic/range coding
//! - residual_cost: SIMD-optimized residual cost estimation

/// Image analysis and auto-detection.
#[doc(hidden)]
pub mod analysis;
mod api;
mod arithmetic;
/// Type-safe encoder configuration.
#[doc(hidden)]
pub mod config;
#[cfg(all(test, feature = "__expert"))]
mod config_expert_tests;
/// Rate-distortion cost estimation.
#[doc(hidden)]
pub mod cost;
pub(crate) mod fast_math;
/// Perceptual distortion model (CSF tables, psy-rd, psy-trellis)
pub(crate) mod psy;
/// Quantization matrix and coefficient quantization.
#[doc(hidden)]
pub mod quantize;
/// Residual cost estimation (SIMD-optimized)
mod residual_cost;
/// Codec lookup tables.
#[doc(hidden)]
pub mod tables;
/// Trellis quantization for RD-optimized coefficient selection
mod trellis;
/// Configuration validation (opt-in fail-fast checks).
pub mod validation;
mod vec_writer;
/// VP8 lossy encoder implementation.
#[doc(hidden)]
pub mod vp8;
/// VP8L lossless encoder implementation.
#[doc(hidden)]
pub mod vp8l;
/// Closed-loop target-zensim adaptive encoder (perceptual quality search).
#[doc(hidden)]
pub mod zensim_target;

/// zenpicker-driven encoder-knob picker (research spike, gated on
/// the `picker` cargo feature). See `picker::spec` for the cell grid.
#[doc(hidden)]
pub mod picker;

// Re-export public API
pub use analysis::{ClassifierDiag, ImageContentType};
pub(crate) use api::EncoderParams;
pub use api::{
    CostModel, EncodeError, EncodeProgress, EncodeRequest, EncodeResult, EncodeStats,
    ImageMetadata, NoProgress, PixelLayout, Preset,
};
pub use config::{EncoderConfig, LosslessConfig, LossyConfig};
#[cfg(feature = "__expert")]
pub use config::{InternalParams, SharpYuvSetting};
pub use validation::ValidationError;
#[doc(hidden)]
pub use vp8l::{Vp8lConfig, Vp8lQuality, encode_vp8l};
#[cfg(feature = "ablation")]
pub use zensim_target::{AblationToggles, set_toggles as set_ablation_toggles};
pub use zensim_target::{ZensimEncodeMetrics, ZensimTarget};

// Crate-internal re-exports for mux module
pub(crate) use api::{chunk_size, encode_alpha_lossless, encode_frame_lossless, write_chunk};
pub(crate) use vec_writer::VecWriter;
