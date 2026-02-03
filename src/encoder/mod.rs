//! WebP encoder implementation

/// Image analysis and auto-detection
pub mod analysis;
mod api;
mod arithmetic;
pub mod cost;
mod fast_math;
/// Quantization matrix and coefficient quantization
pub mod quantize;
/// Residual cost estimation (SIMD-optimized)
mod residual_cost;
pub mod tables;
/// Trellis quantization for RD-optimized coefficient selection
mod trellis;
mod vec_writer;
/// VP8 encoder implementation
pub mod vp8;
/// VP8L (lossless) encoder implementation
pub mod vp8l;

// Re-export public API
pub use analysis::{ClassifierDiag, ContentType};
pub use api::{
    ColorType, Encoder, EncoderConfig, EncoderParams, EncodingError, Preset, WebPEncoder,
};
pub use vp8l::{encode_vp8l, Vp8lConfig, Vp8lQuality};
