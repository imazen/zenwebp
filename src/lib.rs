//! Decoding and Encoding of WebP Images
//!
//! Copyright (C) 2025 Imazen LLC
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Affero General Public License as published
//! by the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! For commercial licensing inquiries: support@imazen.io
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
//! # Encoding
//!
//! Use [`LossyConfig`] or [`LosslessConfig`] with [`EncodeRequest`]:
//!
//! ```rust
//! use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};
//!
//! let config = LossyConfig::new().with_quality(85.0).with_method(4);
//! let rgba_data = vec![255u8; 4 * 4 * 4];
//! let webp = EncodeRequest::lossy(&config, &rgba_data, PixelLayout::Rgba8, 4, 4)
//!     .encode()?;
//! # Ok::<(), zenwebp::EncodeError>(())
//! ```
//!
//! # Decoding
//!
//! Use the convenience functions:
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::decode_rgba(webp_data)?;
//! # Ok::<(), zenwebp::DecodeError>(())
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
//! # Ok::<(), zenwebp::DecodeError>(())
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

/// Resource estimation heuristics for encoding and decoding operations.
pub mod heuristics;

// Re-export decoder public API
pub use decoder::{
    decode_bgr, decode_bgr_into, decode_bgra, decode_bgra_into, decode_rgb, decode_rgb_into,
    decode_rgba, decode_rgba_into, decode_yuv420, BitstreamFormat, DecodeConfig, DecodeError,
    DecodeRequest, DecodeResult, ImageInfo, Limits, LoopCount, StreamStatus, StreamingDecoder,
    UpsamplingMethod, WebPDecoder, YuvPlanes,
};

// Re-export encoder public API
pub use encoder::{
    ClassifierDiag, ContentType, EncodeError, EncodeProgress, EncodeRequest, EncodeResult,
    EncodeStats, EncoderConfig, ImageMetadata, LosslessConfig, LossyConfig, NoProgress,
    PixelLayout, Preset,
};

#[allow(deprecated)]
pub use encoder::ColorType;
// Re-export mux/demux public API
pub use mux::{
    AnimFrame, AnimationConfig, AnimationDecoder, AnimationEncoder, AnimationInfo, BlendMethod,
    DemuxFrame, DisposeMethod, MuxError, MuxFrame, WebPDemuxer, WebPMux,
};

// Re-export cooperative cancellation types
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export VP8 decoder (public module)
pub use decoder::vp8;

// Deprecated type aliases (backwards compatibility)
#[allow(deprecated)]
pub use decoder::DecodingError;
#[allow(deprecated)]
pub use encoder::{EncodingError, EncodingStats};

/// Standalone metadata convenience functions for already-encoded WebP data.
///
/// For embedding metadata during encoding, use
/// [`EncodeRequest::with_metadata`] instead.
pub mod metadata;
