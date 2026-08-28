//! `ResourceLimits::prefer_fallible_allocations` on the ENCODE side (#63).
//!
//! The encoder's top-level O(pixels) buffers (layout-expansion copies, the
//! VP8L ARGB plane, the lossy ARGB→RGBA fix-up, the ALPH alpha planes) honor
//! the preference: `Fallible` routes them through `try_reserve`,
//! `Infallible` through `vec!`, `CodecDefault` keeps the encoder's own
//! (infallible) default. All three MUST produce byte-identical output — the
//! policy changes how memory is obtained, never what is encoded.
//!
//! Real allocation failure cannot be exercised in-process (the dimension cap
//! keeps every request well under what the allocator refuses), so the
//! `AllocFailed → EncodeError::LimitExceeded(Memory)` mapping is covered at
//! the helper level in `decoder::alloc_util` and by the lowering test in
//! `codec.rs`; this file pins byte identity across the three modes on every
//! layout that reaches a routed buffer.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig as _};
use zencodec::{AllocPreference, ResourceLimits};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenwebp::zencodec::WebpEncoderConfig;

/// Gradient + checker noise, with a checker of alpha=0 / alpha=x / opaque so
/// the ALPH path (lossy) and the alpha-aware VP8L path (lossless) both see
/// real transparency.
fn make_image(w: u32, h: u32, bpp: usize, bgra: bool) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize) * bpp];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) as usize) * bpp;
            let noise = if (x ^ y) & 1 == 0 { 0 } else { 9 };
            let r = ((x * 255 / w.max(1)) as u8).wrapping_add(noise);
            let g = ((y * 255 / h.max(1)) as u8).wrapping_add(noise);
            let b = (((x + y) * 255 / (w + h).max(1)) as u8) ^ noise;
            let a = if ((x / 8) + (y / 8)) % 3 == 0 {
                0
            } else if (x + y) % 5 == 0 {
                (x & 255) as u8
            } else {
                255
            };
            match bpp {
                1 => out[i] = g,
                3 => {
                    out[i..i + 3].copy_from_slice(&[r, g, b]);
                }
                4 if bgra => out[i..i + 4].copy_from_slice(&[b, g, r, a]),
                4 => out[i..i + 4].copy_from_slice(&[r, g, b, a]),
                _ => unreachable!(),
            }
        }
    }
    out
}

fn encode_with_pref(
    cfg: &WebpEncoderConfig,
    pixels: &[u8],
    w: u32,
    h: u32,
    desc: PixelDescriptor,
    pref: Option<AllocPreference>,
) -> Vec<u8> {
    let stride_bytes = (w as usize) * desc.bytes_per_pixel();
    let slice = PixelSlice::new(pixels, w, h, stride_bytes, desc).expect("slice");
    let job = cfg.clone().job();
    let job = match pref {
        Some(p) => job.with_limits(ResourceLimits::none().with_prefer_fallible_allocations(p)),
        None => job,
    };
    job.encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec()
}

fn assert_three_modes_identical(cfg: &WebpEncoderConfig, name: &str, desc: PixelDescriptor) {
    let (w, h) = (67u32, 41u32); // odd dims: partial MBs + odd chroma
    let bpp = desc.bytes_per_pixel();
    let bgra = desc == PixelDescriptor::BGRA8_SRGB;
    let px = make_image(w, h, bpp, bgra);
    let default = encode_with_pref(cfg, &px, w, h, desc, None);
    let codec_default = encode_with_pref(cfg, &px, w, h, desc, Some(AllocPreference::CodecDefault));
    let fallible = encode_with_pref(cfg, &px, w, h, desc, Some(AllocPreference::Fallible));
    let infallible = encode_with_pref(cfg, &px, w, h, desc, Some(AllocPreference::Infallible));
    assert!(!default.is_empty(), "{name}: empty output");
    assert_eq!(
        default, codec_default,
        "{name}: CodecDefault differs from unset"
    );
    assert_eq!(default, fallible, "{name}: Fallible path changed the bytes");
    assert_eq!(
        default, infallible,
        "{name}: Infallible path changed the bytes"
    );
}

#[test]
fn lossless_three_modes_byte_identical() {
    let cfg = WebpEncoderConfig::lossless();
    for (name, desc) in [
        ("lossless rgb8", PixelDescriptor::RGB8_SRGB),
        ("lossless rgba8", PixelDescriptor::RGBA8_SRGB),
        ("lossless bgra8", PixelDescriptor::BGRA8_SRGB),
        ("lossless gray8", PixelDescriptor::GRAY8_SRGB),
    ] {
        assert_three_modes_identical(&cfg, name, desc);
    }
}

#[test]
fn lossy_with_alpha_three_modes_byte_identical() {
    // RGBA / BGRA with real transparency: exercises the ALPH alpha-plane
    // extraction + alpha-in-green VP8L plane (both routed buffers).
    let cfg = WebpEncoderConfig::lossy().with_quality(75.0);
    for (name, desc) in [
        ("lossy rgba8", PixelDescriptor::RGBA8_SRGB),
        ("lossy bgra8", PixelDescriptor::BGRA8_SRGB),
        ("lossy rgb8", PixelDescriptor::RGB8_SRGB),
    ] {
        assert_three_modes_identical(&cfg, name, desc);
    }
}
