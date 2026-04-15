//! Golden regression: zenwebp lossless ↔ libwebp lossless RGBA bit-exact.
//!
//! Encodes a small corpus of transparent RGBA images through both zenwebp
//! and libwebp lossless encoders, decodes each output, and asserts both
//! recover the original pixels byte-exactly. "Lossless is lossless" — there
//! should be no deviation in either direction on any channel (including RGB
//! under alpha=0, which some encoders silently zero).
//!
//! Tracks imazen/zenwebp#15.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

/// Encode → decode through zenwebp's lossless path.
fn zen_roundtrip(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let cfg = EncoderConfig::new_lossless();
    let webp = EncodeRequest::new(&cfg, rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .expect("zen encode");
    let (out, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).expect("zen decode");
    assert_eq!((dw, dh), (w, h), "zen roundtrip dimensions");
    out
}

/// Encode → decode through libwebp's lossless path (webpx).
/// `exact=true` disables libwebp's default behavior of zeroing RGB under
/// alpha=0; without this flag libwebp "silently loses" RGB data behind
/// transparent pixels and would diverge from zenwebp (which preserves it).
fn lib_roundtrip(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let webp = webpx::EncoderConfig::new_lossless()
        .exact(true)
        .encode_rgba(rgba, w, h, webpx::Unstoppable)
        .expect("lib encode");
    // Decode with zenwebp for an apples-to-apples RGBA output (both paths
    // therefore share the same decoder; any divergence is encoder-only).
    let (out, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).expect("lib decode via zen");
    assert_eq!((dw, dh), (w, h), "lib roundtrip dimensions");
    out
}

/// Compare two RGBA buffers and report the worst per-channel delta.
fn max_delta(a: &[u8], b: &[u8]) -> [u16; 4] {
    let mut d = [0u16; 4];
    for (ap, bp) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..4 {
            d[c] = d[c].max((ap[c] as i16 - bp[c] as i16).unsigned_abs());
        }
    }
    d
}

fn deterministic_alpha_fringe(w: u32, h: u32, seed: u64) -> Vec<u8> {
    // Radial alpha fringe with varying RGB — mimics an anti-aliased transparent subject.
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    let mut s = seed;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let alpha = ((r - d).clamp(0.0, 1.0) * 255.0) as u8;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rv = ((s >> 33) & 0xff) as u8;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let gv = ((s >> 33) & 0xff) as u8;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bv = ((s >> 33) & 0xff) as u8;
            out.extend_from_slice(&[rv, gv, bv, alpha]);
        }
    }
    out
}

fn assert_exact_lossless(name: &str, rgba: &[u8], w: u32, h: u32) {
    let zen = zen_roundtrip(rgba, w, h);
    let lib = lib_roundtrip(rgba, w, h);
    let zen_d = max_delta(rgba, &zen);
    let lib_d = max_delta(rgba, &lib);

    // Byte-exact expectation: both encoders are lossless, both must recover
    // the source pixels exactly — including RGB under alpha=0.
    assert_eq!(
        zen_d,
        [0, 0, 0, 0],
        "{name}: zenwebp lossless roundtrip drifted R={} G={} B={} A={}",
        zen_d[0],
        zen_d[1],
        zen_d[2],
        zen_d[3]
    );
    assert_eq!(
        lib_d,
        [0, 0, 0, 0],
        "{name}: libwebp lossless roundtrip drifted R={} G={} B={} A={}",
        lib_d[0],
        lib_d[1],
        lib_d[2],
        lib_d[3]
    );

    // Cross-encoder: both decoders see identical pixels for the same input.
    assert_eq!(
        zen, lib,
        "{name}: zenwebp and libwebp lossless outputs diverge after decode"
    );
}

#[test]
fn rgba_alpha_fringe_100x100() {
    let rgba = deterministic_alpha_fringe(100, 100, 0xcafe);
    assert_exact_lossless("alpha_fringe_100x100", &rgba, 100, 100);
}

#[test]
fn rgba_alpha_fringe_64x64() {
    let rgba = deterministic_alpha_fringe(64, 64, 0xbeef);
    assert_exact_lossless("alpha_fringe_64x64", &rgba, 64, 64);
}

#[test]
fn rgba_alpha_fringe_odd_dims() {
    let rgba = deterministic_alpha_fringe(73, 47, 0xf00d);
    assert_exact_lossless("alpha_fringe_73x47", &rgba, 73, 47);
}

#[test]
fn rgba_gallery_rose_100x100() {
    // Decode the gallery rose (400x301, full alpha range) and crop to 100x100
    // — this matches the shape of imageflow's test_transparent_webp_to_webp.
    let Ok(data) = std::fs::read("tests/images/gallery2/1_webp_a.webp") else {
        return;
    };
    let (src, w, h) = zenwebp::oneshot::decode_rgba(&data).expect("decode gallery rose");
    let cw = 100u32.min(w);
    let ch = 100u32.min(h);
    let mut crop = Vec::with_capacity((cw * ch * 4) as usize);
    for y in 0..ch {
        let row_off = (y as usize) * (w as usize) * 4;
        crop.extend_from_slice(&src[row_off..row_off + (cw as usize) * 4]);
    }
    assert_exact_lossless("gallery_rose_crop", &crop, cw, ch);
}
