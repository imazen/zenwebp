//! Decoding and Encoding of WebP Images
//!
//! This crate provides both encoding and decoding of WebP images with a
//! webpx-compatible API.
//!
//! # Features
//!
//! - `std` (default): Enable standard library support including stream-based APIs.
//! - `simd`: Enable SIMD optimizations for faster encoding/decoding.
//! - `fast-yuv`: Enable optimized YUV conversion.
//!
//! For `no_std` environments, disable default features:
//! ```toml
//! [dependencies]
//! zenwebp = { version = "...", default-features = false }
//! ```
//!
//! # Encoding
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
#![cfg_attr(feature = "std", doc = "Or the [`WebPDecoder`] for more control:")]
#![cfg_attr(feature = "std", doc = "")]
#![cfg_attr(feature = "std", doc = "```rust,no_run")]
#![cfg_attr(feature = "std", doc = "use zenwebp::WebPDecoder;")]
#![cfg_attr(feature = "std", doc = "use std::io::Cursor;")]
#![cfg_attr(feature = "std", doc = "")]
#![cfg_attr(feature = "std", doc = "let webp_data: &[u8] = &[]; // your WebP data")]
#![cfg_attr(feature = "std", doc = "let mut decoder = WebPDecoder::new(Cursor::new(webp_data))?;")]
#![cfg_attr(feature = "std", doc = "let (width, height) = decoder.dimensions();")]
#![cfg_attr(feature = "std", doc = "let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];")]
#![cfg_attr(feature = "std", doc = "decoder.read_image(&mut output)?;")]
#![cfg_attr(feature = "std", doc = "# Ok::<(), zenwebp::DecodingError>(())")]
#![cfg_attr(feature = "std", doc = "```")]

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]

extern crate alloc;

#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

// Decoder exports (requires std for now due to internal Read/BufRead usage)
#[cfg(feature = "std")]
pub use self::decoder::{
    decode_rgb, decode_rgb_into, decode_rgba, decode_rgba_into, DecodingError, ImageInfo,
    LoopCount, UpsamplingMethod, WebPDecodeOptions, WebPDecoder,
};

// DecodingError is available in no_std for type compatibility
#[cfg(not(feature = "std"))]
pub use self::decoder::DecodingError;

// Encoder exports (error type always available for compatibility)
pub use self::encoder::EncodingError;

// Encoder exports (requires std for Write trait)
#[cfg(feature = "std")]
pub use self::encoder::{ColorType, Encoder, EncoderConfig, EncoderParams, Preset, WebPEncoder};

// Decoder support modules (require std for Read/BufRead traits)
#[cfg(feature = "std")]
mod alpha_blending;
mod decoder;
#[cfg(feature = "std")]
mod extended;
#[cfg(feature = "std")]
mod huffman;
#[cfg(feature = "std")]
mod loop_filter;
#[cfg(all(feature = "std", feature = "simd", target_arch = "x86_64"))]
mod loop_filter_avx2;
#[cfg(feature = "std")]
mod lossless;
#[cfg(feature = "std")]
mod lossless_transform;

// Encoder module (requires std for Write trait)
mod encoder;

// Shared modules (for encoder)
#[cfg(feature = "simd")]
mod simd_sse;
#[cfg(feature = "std")]
mod transform;
#[cfg(feature = "simd")]
mod transform_simd_intrinsics;
#[cfg(feature = "std")]
mod vp8_arithmetic_decoder;
#[cfg(feature = "std")]
mod vp8_arithmetic_encoder;
#[cfg(feature = "std")]
mod vp8_bit_reader;
#[cfg(feature = "std")]
mod vp8_common;
#[cfg(feature = "std")]
mod vp8_cost;
#[cfg(feature = "std")]
mod vp8_encoder;
#[cfg(feature = "std")]
mod vp8_prediction;
#[cfg(feature = "std")]
mod yuv;
#[cfg(all(feature = "std", feature = "simd", target_arch = "x86_64"))]
mod yuv_simd;

// Public VP8 module (decoder only, requires std)
#[cfg(feature = "std")]
pub mod vp8;
