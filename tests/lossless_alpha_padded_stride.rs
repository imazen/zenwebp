//! Regression test for issue #10: lossless encoder with alpha + padded stride.
//!
//! Imageflow's pipeline feeds zenwebp a `PixelSlice` whose `stride` is larger
//! than `width * bpp` because the backing `Bitmap` allocates padded rows for
//! SIMD alignment. When the encoder is lossless and the image has an alpha
//! channel, the output is catastrophically corrupted (horizontal banding,
//! alpha destruction).
//!
//! Reproduces the end-to-end imageflow flow:
//!   synthetic lossless-alpha WebP
//!     → `WebpDecoderConfig::push_decoder` (negotiates BGRA8)
//!     → copy into Bitmap-like buffer with padded stride
//!     → `WebpEncoderConfig::lossless().encode(PixelSlice)`
//!     → decode again, compare pixel-for-pixel.

#![cfg(all(feature = "std", feature = "zencodec", not(target_arch = "wasm32")))]

use std::borrow::Cow;

use zencodec::decode::{DecodeJob, DecoderConfig as _};
use zencodec::decode::{DecodeRowSink, SinkError};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig as _};
use zenpixels::{PixelDescriptor, PixelSlice, PixelSliceMut};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

/// Build a small RGBA image with a gradient and varying alpha.
fn make_rgba_image(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            out[i] = ((x * 255 / w.max(1)) & 0xFF) as u8;
            out[i + 1] = ((y * 255 / h.max(1)) & 0xFF) as u8;
            out[i + 2] = 128u8.wrapping_add(((x ^ y) & 0x3F) as u8);
            out[i + 3] = if x > w / 2 && y > h / 2 {
                0
            } else if x > w / 2 {
                128
            } else {
                255
            };
        }
    }
    out
}

/// Encode RGBA bytes as a lossless WebP with alpha (reference input).
///
/// Uses `with_exact(true)` so RGB under α=0 is preserved — these tests verify
/// byte-exact roundtrip and the input contains α=0 pixels with nonzero RGB.
fn rgba_to_lossless_webp(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let cfg = WebpEncoderConfig::lossless().with_exact(true);
    let input_slice =
        PixelSlice::new(rgba, w, h, (w as usize) * 4, PixelDescriptor::RGBA8_SRGB).expect("slice");
    cfg.job()
        .encoder()
        .expect("encoder")
        .encode(input_slice)
        .expect("seed encode")
        .data()
        .to_vec()
}

/// Collect full-image decoded output into a single contiguous buffer.
struct CollectSink {
    buf: Vec<u8>,
    width: u32,
    height: u32,
    descriptor: Option<PixelDescriptor>,
    began: bool,
}

impl CollectSink {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            width: 0,
            height: 0,
            descriptor: None,
            began: false,
        }
    }
}

impl DecodeRowSink for CollectSink {
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: PixelDescriptor,
    ) -> Result<(), SinkError> {
        self.width = width;
        self.height = height;
        self.descriptor = Some(descriptor);
        let bpp = descriptor.bytes_per_pixel();
        self.buf.clear();
        self.buf
            .resize((width as usize) * (height as usize) * bpp, 0);
        self.began = true;
        Ok(())
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        descriptor: PixelDescriptor,
    ) -> Result<PixelSliceMut<'_>, SinkError> {
        assert!(self.began);
        let bpp = descriptor.bytes_per_pixel();
        let row_bytes = (width as usize) * bpp;
        let start = (y as usize) * row_bytes;
        let end = start + (height as usize) * row_bytes;
        PixelSliceMut::new(
            &mut self.buf[start..end],
            width,
            height,
            row_bytes,
            descriptor,
        )
        .map_err(|e| SinkError::from(format!("slice: {e}")))
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

/// Decode a WebP via `push_decoder`, preferring BGRA8_SRGB (imageflow's choice).
fn push_decode_bgra(webp: &[u8]) -> (Vec<u8>, u32, u32, PixelDescriptor) {
    let mut sink = CollectSink::new();
    let dec = WebpDecoderConfig::new();
    let preferred = [
        PixelDescriptor::BGRA8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB,
    ];
    dec.job()
        .push_decoder(Cow::Borrowed(webp), &mut sink, &preferred)
        .expect("push_decoder failed");
    let desc = sink.descriptor.expect("descriptor set");
    (sink.buf, sink.width, sink.height, desc)
}

/// Copy contiguous rows into a padded-stride buffer and return it.
fn pad_stride(contiguous: &[u8], w: u32, h: u32, bpp: usize, stride_bytes: usize) -> Vec<u8> {
    let row_bytes = (w as usize) * bpp;
    assert!(stride_bytes >= row_bytes);
    let mut padded = vec![0xA5u8; stride_bytes * (h as usize)];
    for y in 0..(h as usize) {
        padded[y * stride_bytes..y * stride_bytes + row_bytes]
            .copy_from_slice(&contiguous[y * row_bytes..(y + 1) * row_bytes]);
    }
    padded
}

/// Compare two RGBA buffers (contiguous), return (diff_pixels, max_rgb_delta, max_alpha_delta).
fn compare_rgba(a: &[u8], b: &[u8]) -> (u32, u32, u32) {
    let mut diff = 0u32;
    let mut max_rgb = 0u32;
    let mut max_alpha = 0u32;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            diff += 1;
        }
        for c in 0..3 {
            max_rgb = max_rgb.max((x[c] as i32 - y[c] as i32).unsigned_abs());
        }
        max_alpha = max_alpha.max((x[3] as i32 - y[3] as i32).unsigned_abs());
    }
    (diff, max_rgb, max_alpha)
}

/// Run the end-to-end imageflow-equivalent flow and return the decoded RGBA result.
fn end_to_end(w: u32, h: u32, padding_bytes: usize) -> (Vec<u8>, Vec<u8>) {
    let rgba = make_rgba_image(w, h);
    let seed_webp = rgba_to_lossless_webp(&rgba, w, h);

    // Imageflow decode path: push_decoder, preferring BGRA8.
    let (decoded_buf, dw, dh, dec_desc) = push_decode_bgra(&seed_webp);
    assert_eq!((dw, dh), (w, h));
    assert_eq!(
        dec_desc,
        PixelDescriptor::BGRA8_SRGB,
        "expected BGRA8 negotiation"
    );

    // Place decoded BGRA into a padded-stride buffer (imageflow Bitmap layout).
    let stride_bytes = (w as usize) * 4 + padding_bytes;
    let padded_bgra = pad_stride(&decoded_buf, w, h, 4, stride_bytes);

    // Re-encode as lossless WebP directly from BGRA8 padded slice.
    let input_slice = PixelSlice::new(
        &padded_bgra,
        w,
        h,
        stride_bytes,
        PixelDescriptor::BGRA8_SRGB,
    )
    .expect("input slice");
    // Use the same quality imageflow passes (85.0) to exercise that code path.
    let cfg = WebpEncoderConfig::lossless()
        .with_exact(true)
        .with_quality(85.0);
    let reencoded = cfg
        .job()
        .encoder()
        .expect("encoder")
        .encode(input_slice)
        .expect("re-encode")
        .data()
        .to_vec();

    // Decode the re-encoded output and convert back to contiguous RGBA.
    let (out, ow, oh) = zenwebp::oneshot::decode_rgba(&reencoded).expect("final decode");
    assert_eq!((ow, oh), (w, h));
    (rgba, out)
}

#[test]
fn lossless_alpha_roundtrip_contiguous() {
    // Baseline with no stride padding — must be exact.
    let (orig, decoded) = end_to_end(100, 100, 0);
    let (diff, max_rgb, max_a) = compare_rgba(&orig, &decoded);
    assert_eq!(
        (diff, max_rgb, max_a),
        (0, 0, 0),
        "lossless contiguous roundtrip must be exact: diff={diff} max_rgb={max_rgb} max_alpha={max_a}"
    );
}

#[test]
fn lossless_alpha_roundtrip_padded_stride() {
    // Imageflow-style padded stride: 100x4=400 → 448 (64-byte aligned), i.e. 48 bytes padding.
    let (orig, decoded) = end_to_end(100, 100, 48);
    let (diff, max_rgb, max_a) = compare_rgba(&orig, &decoded);
    assert_eq!(
        (diff, max_rgb, max_a),
        (0, 0, 0),
        "lossless padded-stride roundtrip must be exact: diff={diff} max_rgb={max_rgb} max_alpha={max_a}"
    );
}

#[test]
fn lossless_alpha_roundtrip_padded_stride_large() {
    // Larger image to exercise longer strip boundaries in the encoder.
    let (orig, decoded) = end_to_end(128, 128, 64);
    let (diff, max_rgb, max_a) = compare_rgba(&orig, &decoded);
    assert_eq!(
        (diff, max_rgb, max_a),
        (0, 0, 0),
        "lossless padded-stride (large) roundtrip must be exact: diff={diff} max_rgb={max_rgb} max_alpha={max_a}"
    );
}

/// Exercise the exact imageflow pattern:
/// 1. Store decoded BGRA8 in a padded-stride bitmap.
/// 2. Swizzle BGRA→RGBA in-place via `garb::bytes::bgra_to_rgba_inplace_strided`.
/// 3. Encode as lossless WebP with RGBA8_SRGB descriptor using that same padded slice.
#[test]
fn lossless_alpha_imageflow_pattern_inplace_swizzle() {
    let (w, h) = (100u32, 100u32);
    let rgba = make_rgba_image(w, h);
    let seed_webp = rgba_to_lossless_webp(&rgba, w, h);

    // Decode preferring BGRA8 — imageflow's decode output.
    let (decoded_bgra, dw, dh, dec_desc) = push_decode_bgra(&seed_webp);
    assert_eq!((dw, dh), (w, h));
    assert_eq!(dec_desc, PixelDescriptor::BGRA8_SRGB);

    // Copy BGRA into imageflow-style padded-stride bitmap (64-byte row alignment).
    let bpp = 4;
    let row_bytes = (w as usize) * bpp;
    let align = 64;
    let pad = (align - row_bytes % align) % align;
    let stride_bytes = row_bytes + pad;
    let mut padded = vec![0xA5u8; stride_bytes * (h as usize)];
    for y in 0..(h as usize) {
        padded[y * stride_bytes..y * stride_bytes + row_bytes]
            .copy_from_slice(&decoded_bgra[y * row_bytes..(y + 1) * row_bytes]);
    }

    // Swizzle BGRA→RGBA in-place — imageflow's exact preparation step.
    garb::bytes::bgra_to_rgba_inplace_strided(&mut padded, w as usize, h as usize, stride_bytes)
        .expect("in-place swizzle");

    // Encode as lossless with RGBA8_SRGB descriptor.
    let slice = PixelSlice::new(&padded, w, h, stride_bytes, PixelDescriptor::RGBA8_SRGB)
        .expect("input slice");
    let cfg = WebpEncoderConfig::lossless()
        .with_exact(true)
        .with_quality(85.0);
    let reencoded = cfg
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("lossless encode")
        .data()
        .to_vec();

    let (out, ow, oh) = zenwebp::oneshot::decode_rgba(&reencoded).expect("final decode");
    assert_eq!((ow, oh), (w, h));
    let (diff, max_rgb, max_a) = compare_rgba(&rgba, &out);
    assert_eq!(
        (diff, max_rgb, max_a),
        (0, 0, 0),
        "imageflow-pattern lossless roundtrip must be exact: diff={diff} max_rgb={max_rgb} max_alpha={max_a}"
    );
}
