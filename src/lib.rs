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
//! - `std` (default): Enables `encode_to_writer()`. Everything else works without it.
//! - `fast-yuv` (default): Optimized YUV conversion via the `yuv` crate.
//! - `pixel-types`: Type-safe pixel formats via the `rgb` crate.
//!
//! # no_std Support
//!
//! Both encoding and decoding work in `no_std` environments (requires `alloc`):
//! ```toml
//! [dependencies]
//! zenwebp = { version = "...", default-features = false }
//! ```
//!
//! Only [`EncodeRequest::encode_to`] requires `std` (for `std::io::Write`).
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
//! # Ok::<(), whereat::At<zenwebp::EncodeError>>(())
//! ```
//!
//! # Decoding
//!
//! Use the [`oneshot`] convenience functions:
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::oneshot::decode_rgba(webp_data)?;
//! # Ok::<(), whereat::At<zenwebp::DecodeError>>(())
//! ```
//!
//! Or [`WebPDecoder`] for two-phase decoding (inspect headers before allocating):
//!
//! ```rust,no_run
//! use zenwebp::WebPDecoder;
//!
//! let webp_data: &[u8] = &[]; // your WebP data
//! let mut decoder = WebPDecoder::build(webp_data)?;
//! let info = decoder.info();
//! println!("{}x{}, alpha={}", info.width, info.height, info.has_alpha);
//!
//! let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
//! decoder.read_image(&mut output)?;
//! # Ok::<(), zenwebp::DecodeError>(())
//! ```
//!
//! # ICC Color Profiles
//!
//! WebP supports embedded ICC profiles via the ICCP chunk (VP8X extended format).
//! zenwebp preserves ICC profiles through encode and decode but does **not** apply
//! color management — pixels are returned in whatever color space they were encoded
//! in. This matches libwebp's behavior.
//!
//! **Decoding:** Use [`ImageInfo::icc_profile`] to extract the ICC profile after
//! probing headers. Pass it to your color management library (e.g., `lcms2`) to
//! convert pixels to your target color space.
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[];
//! let info = zenwebp::ImageInfo::from_webp(webp_data)?;
//! if let Some(icc) = &info.icc_profile {
//!     // Pass icc bytes to your CMS for color conversion
//! }
//! # Ok::<(), whereat::At<zenwebp::DecodeError>>(())
//! ```
//!
//! **Encoding:** Embed an ICC profile with [`EncodeRequest::with_icc_profile()`]:
//!
//! ```rust,no_run
//! # let icc_bytes: &[u8] = &[];
//! # let rgba_data = vec![255u8; 4 * 4 * 4];
//! use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};
//! let webp = EncodeRequest::lossy(&LossyConfig::new(), &rgba_data, PixelLayout::Rgba8, 4, 4)
//!     .with_icc_profile(icc_bytes)
//!     .encode()?;
//! # Ok::<(), whereat::At<zenwebp::EncodeError>>(())
//! ```
//!
//! **Post-hoc:** The [`metadata`] module can extract, embed, or remove ICC profiles
//! from already-encoded WebP data without re-encoding pixels.
//!
//! # Safety
//!
//! This crate uses `#![forbid(unsafe_code)]` to prevent direct unsafe usage in source.
//! We rely on the [`archmage`] crate for safe SIMD intrinsics. The `#[arcane]` proc
//! macro generates unsafe blocks internally (which bypass the `forbid` lint due to
//! proc-macro span handling). The soundness of our SIMD code depends on archmage's
//! token-based safety model being correct.
//!
//! [`archmage`]: https://docs.rs/archmage

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]

extern crate alloc;

whereat::define_at_crate_info!();

#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

// Core modules (internal — public API is re-exported at crate root)
pub(crate) mod common;
pub mod decoder;
/// Encoder detection and quality estimation from WebP file headers.
pub mod detect;
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

/// One-shot decode convenience functions (`decode_rgba`, `decode_rgb`, etc.).
pub mod oneshot;

// Re-export core decoder types
pub use decoder::{
    DecodeConfig, DecodeError, DecodeRequest, DecodeResult, DecoderContext, ImageInfo, Limits,
    StreamingDecoder, WebPDecoder,
};

// Re-export core encoder types
pub use encoder::{
    EncodeError, EncodeRequest, EncodeResult, EncoderConfig, ImageMetadata, LosslessConfig,
    LossyConfig, PixelLayout, Preset,
};

#[cfg(feature = "zennode")]
pub mod zennode_defs;

#[cfg(feature = "zencodec")]
mod codec;

/// zencodec trait implementations for WebP encoding and decoding.
#[cfg(feature = "zencodec")]
pub mod zencodec {
    pub use crate::codec::{
        WebpAnimationFrameDecoder, WebpAnimationFrameEncoder, WebpDecodeJob, WebpDecoder,
        WebpDecoderConfig, WebpEncodeJob, WebpEncoder, WebpEncoderConfig, WebpStreamingDecoder,
    };
}

/// Standalone metadata convenience functions for already-encoded WebP data.
///
/// For embedding metadata during encoding, use
/// [`EncodeRequest::with_metadata`] instead.
pub mod metadata;

/// Test-only helpers exposed for integration tests.
///
/// Not part of the public API; do not use in production code.
#[doc(hidden)]
pub mod test_helpers {
    /// Scalar RGB->YUV420 conversion (same path the encoder uses).
    ///
    /// Returns (y, u, v) planes with macroblock-aligned dimensions.
    pub fn convert_image_yuv_rgb(
        image_data: &[u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> (
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
    ) {
        crate::decoder::yuv::convert_image_yuv::<3>(image_data, width, height, stride)
    }

    /// Expose the forward gamma LUT for verification tests.
    /// sRGB byte -> linear^0.80 (scale 0..4095), 256 entries.
    pub fn gamma_to_linear_tab() -> &'static [u16; 256] {
        &crate::decoder::yuv::GAMMA_TO_LINEAR_TAB
    }

    /// Expose the inverse gamma LUT for verification tests.
    /// Linear^0.80 -> gamma-space byte, 33 entries.
    pub fn linear_to_gamma_tab() -> &'static [u8; 33] {
        &crate::decoder::yuv::LINEAR_TO_GAMMA_TAB
    }
}
