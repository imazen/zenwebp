//! WebP encoder implementation

mod analysis;
mod api;
mod arithmetic;
pub mod cost;
mod fast_math;
pub mod tables;
mod vec_writer;
/// VP8 encoder implementation
pub mod vp8;

// Re-export public API
pub use api::{ColorType, Encoder, EncoderConfig, EncoderParams, EncodingError, Preset, WebPEncoder};
