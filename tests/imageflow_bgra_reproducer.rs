//! Reproduce the imageflow `test_transparent_webp_to_webp` divergence.
//!
//! imageflow encodes via the zencodec dyn-dispatch trait path:
//!
//!   EncoderConfig::new() -> dyn_job() -> into_encoder() -> encoder.encode(PixelSlice)
//!
//! with `PixelDescriptor::BGRA8_SRGB`. This test mimics that exact call
//! chain to surface bugs that the `EncodeRequest::lossless(... PixelLayout::Rgba8 ...)`
//! golden-regression tests don't exercise (because they pass RGBA directly).
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zencodec::encode::DynEncoderConfig as _;
use zenpixels::{PixelDescriptor, PixelSlice};

fn deterministic_rose(w: u32, h: u32) -> Vec<u8> {
    // A warm gradient with alpha fringe. In BGRA memory order: [B, G, R, A].
    // We pick values where R and B are very different so an R↔B swap is
    // immediately visible after decode: R=220, G=80, B=40.
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.45;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let alpha = ((r - d).clamp(0.0, 1.0) * 255.0) as u8;
            // BGRA memory order: [B, G, R, A] = [40, 80, 220, alpha]
            out.extend_from_slice(&[40, 80, 220, alpha]);
        }
    }
    out
}

#[test]
fn zencodec_dispatch_bgra_lossless_preserves_channels() {
    let w = 64u32;
    let h = 64u32;
    let bgra = deterministic_rose(w, h);

    let config = zenwebp::zencodec::WebpEncoderConfig::lossless().with_quality(85.0);
    let job = config.dyn_job();
    let encoder = job.into_encoder().expect("into_encoder");

    let desc = PixelDescriptor::BGRA8_SRGB;
    let stride = w as usize * 4;
    let ps = PixelSlice::new(&bgra, w, h, stride, desc).expect("pixel slice");
    let output = encoder.encode(ps).expect("encode");
    let webp = output.data();

    // Decode with zenwebp's own decoder; round-trip should be byte-exact on
    // visible pixels (α>0). Alpha=0 pixels may have RGB zeroed by default
    // (exact=false), but here all alphas of interest are non-zero inside the
    // disk, so those pixels must match exactly.
    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(webp).expect("decode");
    assert_eq!((dw, dh), (w, h));

    let mut mismatch = 0usize;
    let mut first_diff: Option<(usize, [u8; 4], [u8; 4])> = None;
    for i in 0..(w * h) as usize {
        let src = &bgra[i * 4..i * 4 + 4]; // B, G, R, A
        let dst = &decoded[i * 4..i * 4 + 4]; // R, G, B, A (decode_rgba returns RGBA)
        let expect = [src[2], src[1], src[0], src[3]]; // convert BGRA→RGBA expectation
        if src[3] == 0 {
            // exact=false allows RGB zeroing under transparent pixels
            if dst[3] != 0 {
                mismatch += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, expect, [dst[0], dst[1], dst[2], dst[3]]));
                }
            }
        } else if dst != expect {
            mismatch += 1;
            if first_diff.is_none() {
                first_diff = Some((i, expect, [dst[0], dst[1], dst[2], dst[3]]));
            }
        }
    }
    assert_eq!(
        mismatch, 0,
        "zencodec BGRA lossless roundtrip diverged in {mismatch} pixels; first: {first_diff:?}"
    );
}

#[test]
fn zencodec_dispatch_bgra_lossless_with_padded_stride() {
    // SIMD-aligned imageflow bitmaps have stride > width*bpp. Verify the
    // dispatch path honors stride (not just tight rows).
    let w = 64u32;
    let h = 64u32;
    let bgra = deterministic_rose(w, h);

    // Build a strided buffer: extra 16 bytes per row of padding.
    let padded_stride = w as usize * 4 + 16;
    let mut padded = vec![0xAAu8; padded_stride * h as usize];
    for y in 0..h as usize {
        let src = &bgra[y * w as usize * 4..(y + 1) * w as usize * 4];
        padded[y * padded_stride..y * padded_stride + w as usize * 4].copy_from_slice(src);
    }

    let config = zenwebp::zencodec::WebpEncoderConfig::lossless().with_quality(85.0);
    let job = config.dyn_job();
    let encoder = job.into_encoder().expect("into_encoder");

    let desc = PixelDescriptor::BGRA8_SRGB;
    let ps = PixelSlice::new(&padded, w, h, padded_stride, desc).expect("padded pixel slice");
    let output = encoder.encode(ps).expect("encode");
    let webp = output.data();

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(webp).expect("decode");
    assert_eq!((dw, dh), (w, h));

    let mut mismatch = 0usize;
    let mut first_diff: Option<(usize, [u8; 4], [u8; 4])> = None;
    for i in 0..(w * h) as usize {
        let src = &bgra[i * 4..i * 4 + 4];
        let dst = &decoded[i * 4..i * 4 + 4];
        let expect = [src[2], src[1], src[0], src[3]];
        if src[3] == 0 {
            if dst[3] != 0 {
                mismatch += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, expect, [dst[0], dst[1], dst[2], dst[3]]));
                }
            }
        } else if dst != expect {
            mismatch += 1;
            if first_diff.is_none() {
                first_diff = Some((i, expect, [dst[0], dst[1], dst[2], dst[3]]));
            }
        }
    }
    assert_eq!(
        mismatch, 0,
        "zencodec BGRA lossless padded-stride diverged in {mismatch} pixels; first: {first_diff:?}"
    );
}
