//! Method-0 (low-effort) lossless tier: roundtrip + libwebp interop.
//!
//! m0 mirrors libwebp's `low_effort` mode: no entropy analysis (palette else
//! SubtractGreen+Predictor), fixed Select predictor for every tile, no
//! cross-color, no color cache, single plain-LZ77 pass, 4-bin unconditional
//! histogram merging without stochastic/greedy refinement. These tests pin the
//! contract that the fast tier is still *lossless* and that its streams decode
//! identically through libwebp.
//!
//! The multi-region image also exercises the unreferenced-cluster compaction
//! in `build_final_histograms`: low-effort clustering + remap can strand a
//! cluster with zero tiles, and the encoder used to write its (empty) Huffman
//! trees anyway — but the decoder sizes the group list from the entropy
//! image's max symbol, so the extra trees shifted the rest of the bitstream
//! (found 2026-07-14 on a photographic RGBA image at m0).

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

/// Smooth base + mild noise: photographic-ish, defeats palette detection.
fn photo_like(w: u32, h: u32) -> Vec<u8> {
    let mut rng = Rng(0xABCD_EF01_2345_6789);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let base_r = (x * 255 / w.max(1)) as u8;
            let base_g = (y * 255 / h.max(1)) as u8;
            let base_b = ((x + y) * 127 / (w + h).max(1)) as u8;
            out.push(base_r.wrapping_add(rng.byte() % 17));
            out.push(base_g.wrapping_add(rng.byte() % 17));
            out.push(base_b.wrapping_add(rng.byte() % 17));
            out.push(255);
        }
    }
    out
}

/// <= 16 colors: forces the palette path at m0.
fn palette_image(w: u32, h: u32) -> Vec<u8> {
    const COLORS: [[u8; 4]; 12] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 255, 255],
        [255, 0, 255, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [128, 64, 32, 255],
        [32, 64, 128, 255],
        [200, 200, 200, 128],
        [10, 20, 30, 0],
    ];
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        let c = COLORS[(rng.next() % COLORS.len() as u64) as usize];
        out.extend_from_slice(&c);
    }
    out
}

/// Anti-aliased alpha fringe over varying RGB.
fn alpha_fringe(w: u32, h: u32) -> Vec<u8> {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.4;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let a = ((r - d) * 32.0).clamp(0.0, 255.0) as u8;
            out.push((x % 251) as u8);
            out.push((y % 241) as u8);
            out.push(((x + y) % 253) as u8);
            out.push(a);
        }
    }
    out
}

/// Four quadrants with very different statistics — drives the low-effort
/// entropy bins apart so clustering + remap get real work (the stranded
/// cluster scenario).
fn quadrants(w: u32, h: u32) -> Vec<u8> {
    let mut rng = Rng(0xFEDC_BA98_7654_3210);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let px: [u8; 4] = match (x < w / 2, y < h / 2) {
                (true, true) => [255, 0, 0, 255],                           // flat
                (false, true) => [rng.byte(), rng.byte(), rng.byte(), 255], // noise
                (true, false) => {
                    let v = (x * 255 / w.max(1)) as u8; // gradient
                    [v, v, v, 255]
                }
                (false, false) => {
                    let v = if (x / 4 + y / 4) % 2 == 0 { 180 } else { 20 }; // checker
                    [v, 40, 200 - v, 255]
                }
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

fn roundtrip_m0(name: &str, rgba: &[u8], w: u32, h: u32) {
    let cfg = EncoderConfig::Lossless(
        zenwebp::LosslessConfig::new()
            .with_method(0)
            .with_exact(true),
    );
    let webp = EncodeRequest::new(&cfg, rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("{name}: m0 encode failed: {e:?}"));

    // zenwebp decode must be pixel-exact
    let (zen, zw, zh) = zenwebp::oneshot::decode_rgba(&webp)
        .unwrap_or_else(|e| panic!("{name}: zenwebp decode failed: {e:?}"));
    assert_eq!((zw, zh), (w, h), "{name}: dimensions");
    assert_eq!(zen, rgba, "{name}: zenwebp roundtrip not lossless at m0");

    // libwebp must agree byte-for-byte (interop: our m0 streams are valid)
    let (lib, lw, lh) = webpx::decode_rgba(&webp)
        .unwrap_or_else(|e| panic!("{name}: libwebp decode failed: {e:?}"));
    assert_eq!((lw, lh), (w, h), "{name}: libwebp dimensions");
    assert_eq!(
        lib, rgba,
        "{name}: libwebp decode of our m0 stream diverges"
    );
}

#[test]
fn m0_roundtrip_photo_like() {
    let (w, h) = (512, 384);
    roundtrip_m0("photo_like", &photo_like(w, h), w, h);
}

#[test]
fn m0_roundtrip_palette() {
    let (w, h) = (320, 200);
    roundtrip_m0("palette", &palette_image(w, h), w, h);
}

#[test]
fn m0_roundtrip_alpha_fringe() {
    let (w, h) = (400, 300);
    roundtrip_m0("alpha_fringe", &alpha_fringe(w, h), w, h);
}

#[test]
fn m0_roundtrip_quadrants() {
    // 800x600 with 128px huffman tiles → 7x5 entropy image, multiple bins.
    let (w, h) = (800, 600);
    roundtrip_m0("quadrants", &quadrants(w, h), w, h);
}

#[test]
fn m0_roundtrip_tiny_sizes() {
    for (w, h) in [
        (1u32, 1u32),
        (2, 1),
        (1, 2),
        (3, 2),
        (16, 16),
        (17, 33),
        (64, 64),
    ] {
        roundtrip_m0(&format!("tiny_{w}x{h}"), &photo_like(w, h), w, h);
    }
}
