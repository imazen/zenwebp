//! Near-lossless (`near_lossless < 100`) encoder output is DECODED and
//! bounded against the original pixels — zenwebp#89 (class-E gap from the
//! 2026-08-26 ultracode sweep).
//!
//! Before this test the only coverage of the near-lossless branch was the
//! byte-identity SIMD tier grid in `simd_tier_parity.rs`, which compares
//! encoder bytes to encoder bytes: a deterministic reconstruction drift in
//! `apply_near_lossless` or the closed-loop residual quantization inside
//! `apply_predictor_transform` would have been tier-identical and blessed.
//!
//! Oracle: the per-channel error budget derived from the encoder's own
//! quantization ladder, `max_quantization_from_quality(q) = 1 << (5 - q/20)`.
//! Both near-lossless paths round to a multiple of a quantization level no
//! larger than `max_quantization`, so the per-sample error is at most
//! `max_quantization / 2` — and that is exactly what is measured on noise
//! (1, 2, 4, 8, 16 at nl 80, 60, 40, 20, 0). The assertion allows
//! `max_quantization` (2x headroom); the measured maxima are printed so a
//! drift is visible before it reaches the bound.
//!
//! This test found zenwebp#89's real bug on its first run: `near_lossless`
//! was dropped when the public config was lowered to the VP8L config
//! (`..Vp8lConfig::default()` reset it to 100), so every quality from 80 to 0
//! produced bytes identical to exact lossless.
//!
//! Liveness: near-lossless must actually ALTER the noise image at every
//! tested quality for methods >= 1 (the branch is skipped for images under
//! 64 px on both axes, so the fixtures are 96 px and up) — a test that
//! passed because the quantizer silently did nothing would prove nothing.
//! Method 0 is libwebp's `low_effort` predictor path, which by design does
//! not apply near-lossless residual quantization (`CopyImageWithPrediction`
//! takes the plain `PredictBatch` route); it is still decoded and bounded
//! here, just not required to change anything.

use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LosslessConfig, PixelLayout};

struct Img {
    name: &'static str,
    w: u32,
    h: u32,
    layout: PixelLayout,
    px: Vec<u8>,
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

fn noise_rgb(w: u32, h: u32, seed: u64) -> Img {
    let mut rng = Lcg::new(seed);
    let px = (0..w * h * 3).map(|_| rng.next_u8()).collect();
    Img {
        name: "noise_rgb",
        w,
        h,
        layout: PixelLayout::Rgb8,
        px,
    }
}

/// Smooth photo-like content with mild noise: exercises the "smooth →
/// smaller quantization" branch of `near_lossless_residual`.
fn gradient_noise_rgb(w: u32, h: u32, seed: u64) -> Img {
    let mut rng = Lcg::new(seed);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let n = (rng.next_u8() % 7) as i32 - 3;
            let r = ((x * 255 / w.max(1)) as i32 + n).clamp(0, 255) as u8;
            let g = ((y * 255 / h.max(1)) as i32 - n).clamp(0, 255) as u8;
            let b = (((x + y) * 255 / (w + h).max(1)) as i32 + n / 2).clamp(0, 255) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    Img {
        name: "gradient_noise_rgb",
        w,
        h,
        layout: PixelLayout::Rgb8,
        px,
    }
}

/// RGBA with a mid-range alpha ramp so the alpha channel is quantized too
/// (fully transparent/opaque alpha is preserved exactly by the residual path).
fn noise_rgba(w: u32, h: u32, seed: u64) -> Img {
    let mut rng = Lcg::new(seed);
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = (16 + ((x + y) * 7) % 224) as u8;
            px.extend_from_slice(&[rng.next_u8(), rng.next_u8(), rng.next_u8(), a]);
        }
    }
    Img {
        name: "noise_rgba",
        w,
        h,
        layout: PixelLayout::Rgba8,
        px,
    }
}

fn max_quantization_from_quality(q: u8) -> i32 {
    1 << (5 - q / 20)
}

fn encode(img: &Img, method: u8, near_lossless: u8) -> Vec<u8> {
    let lc = LosslessConfig::new()
        .with_method(method)
        .with_near_lossless(near_lossless);
    EncodeRequest::lossless(&lc, &img.px, img.layout, img.w, img.h)
        .encode()
        .unwrap_or_else(|e| panic!("{} m{method} nl{near_lossless}: encode: {e:?}", img.name))
}

fn decode_rgba(webp: &[u8]) -> (Vec<u8>, u32, u32) {
    DecodeRequest::new(&DecodeConfig::default(), webp)
        .decode_rgba()
        .expect("decode failed")
}

/// Per-channel max |decoded − source|, and whether ANY sample changed.
fn compare(img: &Img, decoded: &[u8]) -> (i32, bool) {
    let bpp = match img.layout {
        PixelLayout::Rgb8 => 3,
        PixelLayout::Rgba8 => 4,
        other => panic!("unexpected layout {other:?}"),
    };
    let mut max = 0i32;
    let mut changed = false;
    for (i, src) in img.px.chunks_exact(bpp).enumerate() {
        let dst = &decoded[i * 4..i * 4 + 4];
        let src_a = if bpp == 4 { src[3] } else { 255 };
        for (s, d) in [src[0], src[1], src[2], src_a].iter().zip(dst) {
            let e = (i32::from(*s) - i32::from(*d)).abs();
            max = max.max(e);
            changed |= e != 0;
        }
    }
    (max, changed)
}

#[test]
fn near_lossless_decodes_within_quantization_bound() {
    let images = [
        noise_rgb(96, 96, 7),
        gradient_noise_rgb(128, 80, 11),
        noise_rgba(96, 96, 13),
    ];
    let mut worst: Vec<String> = Vec::new();
    for img in &images {
        for &method in &[0u8, 3, 6] {
            for &nl in &[80u8, 60, 40, 20, 0] {
                let bytes = encode(img, method, nl);
                let (decoded, w, h) = decode_rgba(&bytes);
                assert_eq!(
                    (w, h),
                    (img.w, img.h),
                    "{} m{method} nl{nl}: dims",
                    img.name
                );
                assert_eq!(decoded.len(), (w * h * 4) as usize);
                let (max_err, changed) = compare(img, &decoded);
                let bound = max_quantization_from_quality(nl);
                assert!(
                    max_err <= bound,
                    "{} m{method} nl{nl}: max per-channel error {max_err} exceeds \
                     the near-lossless budget {bound} (max_quantization={})",
                    img.name,
                    max_quantization_from_quality(nl)
                );
                if img.name == "noise_rgb" && method != 0 {
                    assert!(
                        changed,
                        "{} m{method} nl{nl}: near-lossless left every sample \
                         untouched — the branch is not live, the bound above proves nothing",
                        img.name
                    );
                }
                worst.push(format!(
                    "{} m{method} nl{nl}: max_err={max_err}/{bound} changed={changed} bytes={}",
                    img.name,
                    bytes.len()
                ));
            }
        }
    }
    eprintln!("[near_lossless_roundtrip]\n  {}", worst.join("\n  "));
}

/// `near_lossless = 100` is exact lossless — the control for the test above.
#[test]
fn near_lossless_100_is_exact() {
    for img in [noise_rgb(96, 96, 7), noise_rgba(96, 96, 13)] {
        for &method in &[0u8, 3, 6] {
            let bytes = encode(&img, method, 100);
            let (decoded, w, h) = decode_rgba(&bytes);
            assert_eq!((w, h), (img.w, img.h));
            let (max_err, _) = compare(&img, &decoded);
            assert_eq!(max_err, 0, "{} m{method} nl100 must be exact", img.name);
        }
    }
}

/// Palette content with spatial structure in INDEX space, so the encoder's
/// `PaletteAndSpatial` crunch mode (predictor on the palette-index plane)
/// is the winning candidate at m5/m6.
fn palette_ramp_rgb(w: u32, h: u32) -> Img {
    // 64 saturated hues around the color circle: consecutive entries are
    // close (so the encoder's delta-minimizing palette sort keeps ramp
    // order) but R, G and B all vary (so subtract-green cannot collapse the
    // RGB residuals the way it does for gray). Laid out as a diagonal ramp
    // of indices, the index plane is smooth and the entropy analysis picks
    // `PaletteAndSpatial` — the mode with the predictor on the index plane.
    let palette: Vec<[u8; 3]> = (0..64u32)
        .map(|i| {
            let h = i * 6; // 0..384 → 6 sextants of 64
            let (sextant, f) = (h / 64, (h % 64 * 4) as u8);
            match sextant {
                0 => [255, f, 0],
                1 => [255 - f, 255, 0],
                2 => [0, 255, f],
                3 => [0, 255 - f, 255],
                4 => [f, 0, 255],
                _ => [255, 0, 255 - f],
            }
        })
        .collect();
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let idx = ((x + y) * 63 / (w + h - 2).max(1)) as usize;
            px.extend_from_slice(&palette[idx.min(63)]);
        }
    }
    Img {
        name: "palette_ramp_rgb",
        w,
        h,
        layout: PixelLayout::Rgb8,
        px,
    }
}

/// Whether the VP8L payload of a simple-container WebP opens with a
/// color-indexing (palette) transform. zenwebp writes the palette transform
/// first whenever it is used, and the VP8L header is exactly 5 bytes, so the
/// first transform's `present` bit and 2-bit type sit in the low bits of
/// payload byte 5 (type 3 = COLOR_INDEXING).
fn uses_palette_transform(webp: &[u8]) -> bool {
    let pos = webp
        .windows(4)
        .position(|w| w == b"VP8L")
        .expect("simple-container VP8L chunk");
    let payload = &webp[pos + 8..];
    assert_eq!(payload[0], 0x2f, "VP8L signature");
    let b = payload[5];
    (b & 1) == 1 && ((b >> 1) & 3) == 3
}

/// Near-lossless must be a no-op on palette images. libwebp forces strength
/// 100 whenever the palette transform is used (`EncodeStreamHook`), because
/// the predictor in `PaletteAndSpatial` mode runs on the palette-INDEX plane
/// and residual quantization there swaps pixels to unrelated palette entries.
/// zenwebp lowered `near_lossless` into `write_predictor_transform` without
/// that guard, so once #89 plumbed the knob, `with_near_lossless(q < 100)`
/// corrupted every PaletteAndSpatial encode (#78-A).
///
/// Only encodes that actually chose the color-indexing transform are held
/// to exactness (the entropy analysis may route a 64-color image to a
/// non-palette mode at lower methods, where near-lossless is legitimately
/// lossy); the test requires that at least the m6/q100 brute-force cells
/// did use it, so the guard is exercised. Watched to FAIL (max error 8-32
/// on the palette cells) with the guard removed.
#[test]
fn near_lossless_is_exact_on_palette_images() {
    let img = palette_ramp_rgb(128, 128);
    let mut palette_cells = 0;
    for method in [0u8, 3, 5, 6] {
        for nl in [0u8, 20, 40, 60, 80] {
            // q100 so m6 brute-forces every crunch mode (and m5 tries the
            // palette variant): PaletteAndSpatial must be on the table.
            let lc = LosslessConfig::new()
                .with_method(method)
                .with_quality(100.0)
                .with_near_lossless(nl);
            let webp = EncodeRequest::lossless(&lc, &img.px, img.layout, img.w, img.h)
                .encode()
                .unwrap_or_else(|e| panic!("{} m{method} nl{nl}: encode: {e:?}", img.name));
            if !uses_palette_transform(&webp) {
                continue;
            }
            palette_cells += 1;
            let (decoded, w, h) = decode_rgba(&webp);
            assert_eq!((w, h), (img.w, img.h));
            let (max, changed) = compare(&img, &decoded);
            assert!(
                !changed,
                "{} m{method} nl{nl}: palette-transform encode must roundtrip exactly, max error {max}",
                img.name
            );
        }
    }
    assert!(
        palette_cells >= 5,
        "only {palette_cells} cells used the palette transform — the fixture no longer exercises the guard"
    );
}
