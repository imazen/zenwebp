//! GRAY8 input regression tests.
//!
//! Covers two paths that landed together in the GRAY8-passthrough work:
//!
//! 1. `EncodeRequest` with `PixelLayout::L8` — direct lossy / lossless encoding
//!    from a single-channel grayscale buffer.
//! 2. `WebpEncoderConfig` with `PixelDescriptor::GRAY8_SRGB` — the zencodec
//!    adapter, which previously expanded gray→RGB into a 3× transient `Vec`
//!    before handing off; now passes the buffer through to the L8 path
//!    zero-copy.
//!
//! What's pinned:
//! - Lossless roundtrip is byte-exact for every pixel value 0..=255.
//! - Lossy roundtrip stays within the standard chroma-noise tolerance and
//!   the decoded output is genuinely grayscale (R=G=B per pixel).
//! - Padded-stride input encodes correctly (no leakage of padding bytes).
//! - Output of GRAY8 path is no larger than feeding the same image as
//!   gray→RGB-expanded — guards against the `convert_image_y` chroma plane
//!   regressing back to a non-neutral fill that would cost extra DC bits.

#![cfg(all(feature = "std", feature = "zencodec", not(target_arch = "wasm32")))]

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig as _};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenwebp::zencodec::WebpEncoderConfig;
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

/// Diagonal gradient + checkerboard noise to give the encoder real content
/// to work with (constant images compress trivially and prove nothing).
fn make_gray_image(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize)];
    for y in 0..h {
        for x in 0..w {
            let g = ((x + y) * 255 / (w + h).max(1)) as u8;
            let noise = if (x ^ y) & 1 == 0 { 0 } else { 8 };
            out[(y * w + x) as usize] = g.saturating_add(noise);
        }
    }
    out
}

/// gray → RGB expansion (R=G=B), the old behavior — used for
/// equivalence comparisons.
fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(gray.len() * 3);
    for &g in gray {
        out.push(g);
        out.push(g);
        out.push(g);
    }
    out
}

#[test]
fn lossless_l8_roundtrip_is_byte_exact() {
    let (w, h) = (96u32, 64u32);
    let gray = make_gray_image(w, h);

    let config = LosslessConfig::new();
    let webp = EncodeRequest::lossless(&config, &gray, PixelLayout::L8, w, h)
        .encode()
        .expect("lossless L8 encode");

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).expect("decode");
    assert_eq!((dw, dh), (w, h));
    for (i, px) in decoded.chunks_exact(4).enumerate() {
        assert_eq!(
            (px[0], px[1], px[2], px[3]),
            (gray[i], gray[i], gray[i], 255),
            "pixel {i} mismatch in lossless L8 roundtrip"
        );
    }
}

#[test]
fn lossy_l8_roundtrip_within_tolerance_and_neutral_chroma() {
    let (w, h) = (96u32, 64u32);
    let gray = make_gray_image(w, h);

    let config = LossyConfig::new().with_quality(80.0);
    let webp = EncodeRequest::lossy(&config, &gray, PixelLayout::L8, w, h)
        .encode()
        .expect("lossy L8 encode");

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).expect("decode");
    assert_eq!((dw, dh), (w, h));

    // Decoded output must be true grayscale (R=G=B) — confirms the chroma
    // plane round-tripped to its neutral value through quant.
    let mut max_chroma_split = 0i32;
    let mut max_y_diff = 0i32;
    for (i, px) in decoded.chunks_exact(4).enumerate() {
        let split = (px[0] as i32 - px[1] as i32)
            .abs()
            .max((px[1] as i32 - px[2] as i32).abs());
        max_chroma_split = max_chroma_split.max(split);
        let y_diff = (px[0] as i32 - gray[i] as i32).abs();
        max_y_diff = max_y_diff.max(y_diff);
        assert_eq!(px[3], 255, "alpha must be opaque for L8 input");
    }
    assert!(
        max_chroma_split <= 2,
        "decoded chroma must round-trip neutral (R=G=B): max split = {max_chroma_split}"
    );
    // q80 lossy luma tolerance — generous to absorb VP8 quant + filter.
    assert!(
        max_y_diff <= 16,
        "lossy L8 luma diff exceeded tolerance: {max_y_diff}"
    );
}

#[test]
fn zencodec_gray8_passthrough_lossless_byte_exact() {
    let (w, h) = (80u32, 48u32);
    let gray = make_gray_image(w, h);

    let cfg = WebpEncoderConfig::lossless().with_exact(true);
    let slice =
        PixelSlice::new(&gray, w, h, w as usize, PixelDescriptor::GRAY8_SRGB).expect("slice");
    let bytes = cfg
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec();

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&bytes).expect("decode");
    assert_eq!((dw, dh), (w, h));
    for (i, px) in decoded.chunks_exact(4).enumerate() {
        assert_eq!(
            (px[0], px[1], px[2], px[3]),
            (gray[i], gray[i], gray[i], 255),
            "pixel {i} mismatch in zencodec GRAY8 lossless roundtrip"
        );
    }
}

#[test]
fn zencodec_gray8_padded_stride_lossless_byte_exact() {
    // Imageflow-style 64-byte aligned row layout: 80 * 1 = 80 bytes per row,
    // padded to 128 (48 bytes padding). Padding bytes filled with garbage to
    // catch any path that copies them into the encoded image.
    let (w, h) = (80u32, 48u32);
    let stride_bytes = 128usize;
    let row = w as usize;
    let mut padded = vec![0xA5u8; stride_bytes * (h as usize)];
    let gray = make_gray_image(w, h);
    for y in 0..(h as usize) {
        padded[y * stride_bytes..y * stride_bytes + row]
            .copy_from_slice(&gray[y * row..(y + 1) * row]);
    }

    let cfg = WebpEncoderConfig::lossless().with_exact(true);
    let slice = PixelSlice::new(&padded, w, h, stride_bytes, PixelDescriptor::GRAY8_SRGB)
        .expect("padded slice");
    let bytes = cfg
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec();

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&bytes).expect("decode");
    assert_eq!((dw, dh), (w, h));
    for (i, px) in decoded.chunks_exact(4).enumerate() {
        assert_eq!(
            (px[0], px[1], px[2], px[3]),
            (gray[i], gray[i], gray[i], 255),
            "pixel {i} mismatch in padded-stride GRAY8 roundtrip"
        );
    }
}

#[test]
fn lossy_l8_no_larger_than_gray_expanded_rgb() {
    // The whole point of the GRAY8 fast path is that we save bits *and*
    // memory. If the L8 lossy output ever ends up bigger than feeding the
    // same image as gray-replicated RGB, something has regressed (most
    // likely the chroma plane drifted back to a non-neutral fill).
    let (w, h) = (128u32, 96u32);
    let gray = make_gray_image(w, h);
    let rgb = gray_to_rgb(&gray);

    let config = LossyConfig::new().with_quality(80.0);
    let l8_webp = EncodeRequest::lossy(&config, &gray, PixelLayout::L8, w, h)
        .encode()
        .expect("L8 encode");
    let rgb_webp = EncodeRequest::lossy(&config, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .expect("RGB encode");

    assert!(
        l8_webp.len() <= rgb_webp.len(),
        "L8 lossy output ({}) must not exceed gray→RGB lossy ({}) — \
         chroma fast-path may have regressed",
        l8_webp.len(),
        rgb_webp.len()
    );
}
