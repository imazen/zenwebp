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

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]

extern crate alloc;

#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

// Decoder exports - now available in no_std via slice-based API
pub use self::decoder::{
    decode_rgb, decode_rgb_into, decode_rgba, decode_rgba_into, DecodingError, ImageInfo,
    LoopCount, UpsamplingMethod, WebPDecodeOptions, WebPDecoder,
};

// Encoder exports (error type always available for compatibility)
pub use self::encoder::EncodingError;

// Encoder exports - now work with alloc (no std required)
pub use self::encoder::{ColorType, Encoder, EncoderConfig, EncoderParams, Preset, WebPEncoder};

// Slice reader for no_std support
mod slice_reader;

// Decoder support modules
mod alpha_blending;
mod decoder;
mod extended;
mod huffman;
mod loop_filter;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod loop_filter_avx2;
mod vp8_loop_filter_dispatch;
mod lossless;
mod lossless_transform;

// Encoder modules - now work with alloc (no std required)
mod encoder;
mod fast_math;
mod vec_writer;
mod vp8_arithmetic_encoder;
mod vp8_analysis;
mod vp8_cost;
mod vp8_encoder;
mod vp8_tables;

// Shared modules (for encoder and decoder)
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
mod simd_sse;
mod transform;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod transform_aarch64;
#[cfg(feature = "simd")]
mod transform_simd_intrinsics;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
mod transform_wasm;
mod vp8_arithmetic_decoder;
mod vp8_bit_reader;
mod vp8_common;
mod vp8_prediction;
mod yuv;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod yuv_simd;

// Public VP8 module (decoder)
pub mod vp8;
