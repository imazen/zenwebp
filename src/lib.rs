//! Decoding and Encoding of WebP Images
//!
//! This crate provides both encoding and decoding of WebP images.
//!
//! # Features
//!
//! - `std` (default): Enable standard library support including lossy encoding.
//! - `simd`: Enable SIMD optimizations for faster encoding/decoding.
//! - `fast-yuv`: Enable optimized YUV conversion.
//!
//! # no_std Support
//!
//! Decoding is fully supported in `no_std` environments (requires `alloc`):
//! ```toml
//! [dependencies]
//! zenwebp = { version = "...", default-features = false }
//! ```
//!
//! All decoding functions take `&[u8]` slices directly - no Read/Seek traits required.
//!
//! # Encoding (requires `std`)
//!
//! Use the [`Encoder`] builder for a fluent API:
//!
//! ```rust
//! use zenwebp::{Encoder, Preset};
//!
//! let rgba_data = vec![255u8; 4 * 4 * 4]; // 4x4 RGBA image
//! let webp = Encoder::new_rgba(&rgba_data, 4, 4)
//!     .preset(Preset::Photo)
//!     .quality(85.0)
//!     .encode()?;
//! # Ok::<(), zenwebp::EncodingError>(())
//! ```
//!
//! Or use [`EncoderConfig`] for reusable configuration:
//!
//! ```rust
//! use zenwebp::{EncoderConfig, Preset};
//!
//! let config = EncoderConfig::new().quality(85.0).preset(Preset::Photo);
//! let rgba_data = vec![255u8; 4 * 4 * 4];
//! let webp = config.encode_rgba(&rgba_data, 4, 4)?;
//! # Ok::<(), zenwebp::EncodingError>(())
//! ```
//!
//! # Decoding
//!
//! Use the convenience functions:
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::decode_rgba(webp_data)?;
//! # Ok::<(), zenwebp::DecodingError>(())
//! ```
//!
//! Or the [`WebPDecoder`] for more control:
//!
//! ```rust,no_run
//! use zenwebp::WebPDecoder;
//!
//! let webp_data: &[u8] = &[]; // your WebP data
//! let mut decoder = WebPDecoder::new(webp_data)?;
//! let (width, height) = decoder.dimensions();
//! let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
//! decoder.read_image(&mut output)?;
//! # Ok::<(), zenwebp::DecodingError>(())
//! ```
//!
//! # Safety
//!
//! This crate uses `#![forbid(unsafe_code)]` to prevent direct unsafe usage in source.
//! However, when the `simd` feature is enabled, we rely on the [`archmage`] crate for
//! safe SIMD intrinsics. The `#[arcane]` proc macro generates unsafe blocks internally
//! (which bypass the `forbid` lint due to proc-macro span handling). The soundness of
//! our SIMD code depends on archmage's token-based safety model being correct.
//!
//! Without the `simd` feature, this crate contains no unsafe code whatsoever.
//!
//! [`archmage`]: https://docs.rs/archmage

#![cfg_attr(not(feature = "std"), no_std)]
// Forbid unsafe unless the "unchecked" feature enables it for performance
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![deny(missing_docs)]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]

extern crate alloc;

#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

// Core modules
pub mod common;
pub mod decoder;
pub mod encoder;
/// WebP mux/demux and animation encoding.
pub mod mux;

// Slice reader utility (used by decoder and mux)
mod slice_reader;

/// Type-safe pixel format traits for decoding and encoding.
#[cfg(feature = "pixel-types")]
pub mod pixel;

// Re-export decoder public API
pub use decoder::{
    decode_bgr, decode_bgr_into, decode_bgra, decode_bgra_into, decode_rgb, decode_rgb_into,
    decode_rgba, decode_rgba_into, decode_yuv420, DecodingError, ImageInfo, LoopCount,
    UpsamplingMethod, WebPDecodeOptions, WebPDecoder, YuvPlanes,
};

// Re-export encoder public API
pub use encoder::{
    ClassifierDiag, ColorType, ContentType, EncodeProgress, Encoder, EncoderConfig, EncoderParams,
    EncodingError, EncodingStats, NoProgress, Preset, WebPEncoder,
};

// Re-export mux/demux public API
pub use mux::{
    AnimFrame, AnimationConfig, AnimationDecoder, AnimationEncoder, AnimationInfo, BlendMethod,
    DemuxFrame, DisposeMethod, MuxError, MuxFrame, WebPDemuxer, WebPMux,
};

// Re-export cooperative cancellation types
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export VP8 decoder (public module)
pub use decoder::vp8;
