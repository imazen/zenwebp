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
// Clippy style lints — intentional patterns throughout the codebase.
#![allow(
    clippy::needless_range_loop,   // explicit `for i in 0..n { arr[i] }` is clearer in codec code
    clippy::too_many_arguments,    // codec functions pass many buffers/params
    clippy::manual_div_ceil,       // `(x + y - 1) / y` is the standard idiom
    clippy::manual_is_multiple_of, // `x % y == 0` is clearer than `.is_multiple_of()`
    clippy::manual_repeat_n,       // `repeat().take()` vs `repeat_n` — not available on MSRV
)]
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
mod exif_orientation;
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
    DecodeConfig, DecodeError, DecodeRequest, DecodeResult, ImageInfo, Limits, StreamingDecoder,
    WebPDecoder,
};

// Re-export Orientation from zenpixels (canonical EXIF orientation for the zen ecosystem)
pub use zenpixels::Orientation;

// Re-export core encoder types
pub use encoder::{
    EncodeError, EncodeRequest, EncodeResult, EncoderConfig, ImageMetadata, LosslessConfig,
    LossyConfig, PixelLayout, Preset,
};

/// Re-export sharp YUV configuration from zenyuv.
pub use zenyuv::SharpYuvConfig;

// #[cfg(feature = "zennode")]
// pub mod zennode_defs;

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

    /// Fast RGB->YUV420 conversion (zenyuv SIMD Y + gamma-corrected scalar chroma).
    pub fn convert_image_yuv_rgb_fast(
        image_data: &[u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> (
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
    ) {
        crate::decoder::yuv::convert_image_yuv_fast(
            image_data,
            crate::encoder::PixelLayout::Rgb8,
            width,
            height,
            stride,
        )
    }

    /// L8 → YUV path used by the encoder for grayscale input.
    pub fn convert_image_y_l8(
        image_data: &[u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> (
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
    ) {
        crate::decoder::yuv::convert_image_y::<1>(image_data, width, height, stride)
    }

    /// SSE (sum-of-squared-errors) for a 4x4 luma block.
    /// Dispatches to the same SIMD kernel the encoder uses.
    pub fn sse4x4_dispatch(src: &[u8; 16], pred: &[u8; 16]) -> u32 {
        crate::encoder::vp8::mode_selection::test_only_sse4x4_dispatch(src, pred)
    }

    /// Forward DCT used by the encoder; dispatched (SSE2 / NEON / etc).
    pub fn ftransform_from_u8_4x4_dispatch(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
        crate::common::transform::ftransform_from_u8_4x4(src, ref_)
    }

    /// Scalar reference forward DCT.
    pub fn ftransform_from_u8_4x4_scalar(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
        crate::common::transform::ftransform_from_u8_4x4_scalar(src, ref_)
    }

    /// Inverse DCT used by the encoder; dispatched.
    pub fn idct4x4_dispatch(block: &mut [i32; 16]) {
        crate::common::transform::idct4x4(block);
    }

    /// Scalar reference inverse DCT.
    pub fn idct4x4_scalar(block: &mut [i32; 16]) {
        crate::common::transform::idct4x4_scalar(block);
    }

    /// Forward DCT used by the encoder; dispatched.
    pub fn dct4x4_dispatch(block: &mut [i32; 16]) {
        crate::common::transform::dct4x4(block);
    }

    /// Scalar reference forward DCT.
    pub fn dct4x4_scalar(block: &mut [i32; 16]) {
        crate::common::transform::dct4x4_scalar(block);
    }

    /// VP8 quantization matrix constructor for tests.
    pub fn make_test_matrix(q_dc: u16, q_ac: u16) -> crate::encoder::quantize::VP8Matrix {
        crate::encoder::quantize::VP8Matrix::new(
            q_dc,
            q_ac,
            crate::encoder::quantize::MatrixType::Y1,
        )
    }

    /// Dispatched fused quantize+dequantize used by the encoder.
    pub fn quantize_dequantize_block_dispatch(
        coeffs: &[i32; 16],
        matrix: &crate::encoder::quantize::VP8Matrix,
        use_sharpen: bool,
        quantized: &mut [i32; 16],
        dequantized: &mut [i32; 16],
    ) -> bool {
        crate::encoder::quantize::quantize_dequantize_block_simd(
            coeffs,
            matrix,
            use_sharpen,
            quantized,
            dequantized,
        )
    }

    /// Scalar reference fused quantize+dequantize (does NOT use sharpen,
    /// matching the scalar dispatch tier's behavior).
    pub fn quantize_dequantize_block_scalar(
        coeffs: &[i32; 16],
        matrix: &crate::encoder::quantize::VP8Matrix,
        quantized: &mut [i32; 16],
        dequantized: &mut [i32; 16],
    ) -> bool {
        crate::encoder::quantize::quantize_dequantize_block_scalar(
            coeffs,
            matrix,
            quantized,
            dequantized,
        )
    }

    /// Dispatched standalone quantize.
    pub fn quantize_block_dispatch(
        coeffs: &mut [i32; 16],
        matrix: &crate::encoder::quantize::VP8Matrix,
        use_sharpen: bool,
    ) -> bool {
        crate::encoder::quantize::quantize_block_simd(coeffs, matrix, use_sharpen)
    }

    /// Scalar reference standalone quantize.
    pub fn quantize_block_scalar(
        coeffs: &mut [i32; 16],
        matrix: &crate::encoder::quantize::VP8Matrix,
    ) -> bool {
        let mut has_nz = false;
        for pos in 0..16 {
            coeffs[pos] = matrix.quantize_coeff(coeffs[pos], pos);
            if coeffs[pos] != 0 {
                has_nz = true;
            }
        }
        has_nz
    }

    /// Dispatched dequantize-only via VP8Matrix::dequantize_block.
    pub fn dequantize_block_dispatch(
        matrix: &crate::encoder::quantize::VP8Matrix,
        coeffs: &mut [i32; 16],
    ) {
        matrix.dequantize_block(coeffs);
    }

    /// Scalar reference dequantize.
    pub fn dequantize_block_scalar(
        matrix: &crate::encoder::quantize::VP8Matrix,
        coeffs: &mut [i32; 16],
    ) {
        for i in 0..16 {
            coeffs[i] *= matrix.q[i] as i32;
        }
    }

    /// Dispatched is_flat_coeffs.
    pub fn is_flat_coeffs_dispatch(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
        crate::encoder::cost::distortion::is_flat_coeffs(levels, num_blocks, thresh)
    }

    /// Scalar reference is_flat_coeffs (matches the canonical scalar
    /// dispatch tier: counts nonzero AC coefficients, returns true if
    /// the count stays at or below `thresh`).
    pub fn is_flat_coeffs_scalar(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
        let mut score = 0i32;
        for block in 0..num_blocks {
            for i in 1..16 {
                if levels[block * 16 + i] != 0 {
                    score += 1;
                    if score > thresh {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Dispatched sse4x4_with_residual.
    pub fn sse4x4_with_residual_dispatch(
        src: &[u8; 16],
        pred: &[u8; 16],
        dequantized: &[i32; 16],
    ) -> u32 {
        crate::encoder::vp8::mode_selection::test_only_sse4x4_with_residual_dispatch(
            src,
            pred,
            dequantized,
        )
    }

    /// Scalar reference sse4x4_with_residual: SSE between src and (pred+dequantized).
    pub fn sse4x4_with_residual_scalar(
        src: &[u8; 16],
        pred: &[u8; 16],
        dequantized: &[i32; 16],
    ) -> u32 {
        let mut sum = 0u32;
        for i in 0..16 {
            let rec = (i32::from(pred[i]) + dequantized[i]).clamp(0, 255) as u8;
            let d = (src[i] as i32 - rec as i32).unsigned_abs();
            sum += d * d;
        }
        sum
    }

    /// Luma block layout constant for prediction buffers.
    pub const LUMA_BLOCK_SIZE: usize = crate::common::prediction::LUMA_BLOCK_SIZE;
    /// Chroma block layout constant for prediction buffers.
    pub const CHROMA_BLOCK_SIZE: usize = crate::common::prediction::CHROMA_BLOCK_SIZE;
    /// Stride of a luma prediction block (bytes per row).
    pub const LUMA_STRIDE: usize = crate::common::prediction::LUMA_STRIDE;
    /// Stride of a chroma prediction block (bytes per row).
    pub const CHROMA_STRIDE: usize = crate::common::prediction::CHROMA_STRIDE;

    /// Dispatched sse_16x16_luma. The `pred` buffer has the bordered
    /// layout: pred[(y+1)*LUMA_STRIDE + 1 + x] is luma block pixel (y,x).
    pub fn sse_16x16_luma_dispatch(
        src_y: &[u8],
        src_width: usize,
        mbx: usize,
        mby: usize,
        pred: &[u8; LUMA_BLOCK_SIZE],
    ) -> u32 {
        crate::encoder::vp8::sse_16x16_luma(src_y, src_width, mbx, mby, pred)
    }

    /// Scalar reference sse_16x16_luma.
    pub fn sse_16x16_luma_scalar(
        src_y: &[u8],
        src_width: usize,
        mbx: usize,
        mby: usize,
        pred: &[u8; LUMA_BLOCK_SIZE],
    ) -> u32 {
        let mut sse = 0u32;
        let src_base = mby * 16 * src_width + mbx * 16;
        for y in 0..16 {
            let src_row = src_base + y * src_width;
            let pred_row = (y + 1) * LUMA_STRIDE + 1;
            for x in 0..16 {
                let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
    }

    /// Dispatched sse_8x8_chroma. Same bordered layout as luma but with
    /// CHROMA_STRIDE.
    pub fn sse_8x8_chroma_dispatch(
        src_uv: &[u8],
        src_width: usize,
        mbx: usize,
        mby: usize,
        pred: &[u8; CHROMA_BLOCK_SIZE],
    ) -> u32 {
        crate::encoder::vp8::sse_8x8_chroma(src_uv, src_width, mbx, mby, pred)
    }

    /// Scalar reference sse_8x8_chroma.
    pub fn sse_8x8_chroma_scalar(
        src_uv: &[u8],
        src_width: usize,
        mbx: usize,
        mby: usize,
        pred: &[u8; CHROMA_BLOCK_SIZE],
    ) -> u32 {
        let mut sse = 0u32;
        let src_base = mby * 8 * src_width + mbx * 8;
        for y in 0..8 {
            let src_row = src_base + y * src_width;
            let pred_row = (y + 1) * CHROMA_STRIDE + 1;
            for x in 0..8 {
                let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
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
