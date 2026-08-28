// Shared core of the encode-side fuzz targets.
//
// `include!`d by `encode_lossless_roundtrip.rs` / `encode_lossy_roundtrip.rs`
// (the libFuzzer bins) AND by `tests/fuzz_regression.rs` (the stable-toolchain
// replay of `fuzz/regression/` seeds). One source of truth: if the generator
// or the parameter layout drifted between the fuzzer and the replay harness,
// regression seeds would silently stop reproducing what they were minimized
// for.
//
// Oracle design: lossless encode → decode → pixels must equal the ORIGINAL
// input exactly (`exact = true` keeps RGB under transparent pixels, so the
// full 4-channel comparison is sound). This is the guard for the "encoder
// emits a valid-but-wrong stream" class (#72: a stranded meta-huffman
// cluster corrupted 84% of pixels at default settings, and every decoder —
// old zenwebp, current zenwebp, libwebp — accepted the stream and returned
// identical garbage, so self-consistency roundtrips prove nothing). The
// decode side is separately gated bit-exact against libwebp, which makes
// zenwebp's decoder a sound reference for this comparison. For lossy, the
// invariants are: no panic, every produced stream decodes, dimensions
// survive, and the ALPH plane (lossless at `alpha_quality = 100`) roundtrips
// its alpha bytes exactly.
//
// The fuzz input drives a structured content generator rather than raw
// pixels: #72-class triggers are content × clustering lotteries (multi-region
// layouts, palette content, grain) at specific method/quality cells, which
// byte-level mutation of a 640 KB pixel buffer would never reach.

use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LosslessConfig, LossyConfig,
    PixelLayout,
};

/// Deterministic LCG; one multiply per sample keeps generation cheap.
struct FuzzLcg(u32);

impl FuzzLcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// Fill `rgba` from a small content program chosen by the fuzzer (lossless
/// flavor: layouts that stress palette / predictors / meta-huffman clustering).
fn generate_lossless_content(rgba: &mut [u8], w: usize, h: usize, mode: u8, seed: u32, rest: &[u8]) {
    let mut rng = FuzzLcg(seed ^ 0x9E37_79B9);
    match mode % 5 {
        // Raw fuzz bytes tiled over the buffer: direct mutation power.
        0 => {
            if rest.is_empty() {
                rgba.fill(0);
            } else {
                for (i, px) in rgba.iter_mut().enumerate() {
                    *px = rest[i % rest.len()];
                }
            }
        }
        // Multi-region: KxK blocks, each a base color + small noise. This is
        // the layout family that strands meta-huffman clusters (#72's repro
        // is multi_region 511x320).
        1 => {
            let k = 2 + (rng.next() % 7) as usize; // 2..=8 regions per axis
            let bw = w.div_ceil(k).max(1);
            let bh = h.div_ceil(k).max(1);
            let mut bases = [[0u8; 4]; 64];
            for b in bases.iter_mut().take(k * k) {
                *b = [rng.byte(), rng.byte(), rng.byte(), 255];
            }
            let noise = (rng.next() % 24) as u8;
            for y in 0..h {
                for x in 0..w {
                    let region = (y / bh).min(k - 1) * k + (x / bw).min(k - 1);
                    let base = bases[region % 64];
                    let i = (y * w + x) * 4;
                    for c in 0..4 {
                        let n = if noise == 0 { 0 } else { rng.byte() % noise };
                        rgba[i + c] = base[c].wrapping_add(n);
                    }
                }
            }
        }
        // Palette content: N distinct colors, LCG pattern. Exercises the
        // palette transform and color-cache paths.
        2 => {
            let n = 1 + (rng.next() % 32) as usize;
            let mut pal = [[0u8; 4]; 32];
            for p in pal.iter_mut().take(n) {
                *p = [rng.byte(), rng.byte(), rng.byte(), rng.byte()];
            }
            for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
                px.copy_from_slice(&pal[(rng.next() as usize) % n]);
            }
        }
        // Gradients + grain: smooth base defeats palette, grain amplitude
        // sweeps the predictor/LZ77 tradeoff.
        3 => {
            let grain = (rng.next() % 48) as u8;
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let g = if grain == 0 { 0 } else { rng.byte() % grain };
                    rgba[i] = ((x * 255) / w.max(1)) as u8;
                    rgba[i + 1] = ((y * 255) / h.max(1)) as u8;
                    rgba[i + 2] = (((x + y) * 128) / (w + h).max(1)) as u8 ^ g;
                    rgba[i + 3] = 255u8.wrapping_sub(g / 4);
                }
            }
        }
        // Pure noise: worst case for every transform.
        _ => {
            for px in rgba.iter_mut() {
                *px = rng.byte();
            }
        }
    }
}

/// Lossy flavor: alpha-pipeline stress shapes.
fn generate_lossy_content(rgba: &mut [u8], w: usize, h: usize, mode: u8, seed: u32, rest: &[u8]) {
    let mut rng = FuzzLcg(seed ^ 0x51F1_5EED);
    match mode % 4 {
        0 => {
            if rest.is_empty() {
                rgba.fill(128);
            } else {
                for (i, px) in rgba.iter_mut().enumerate() {
                    *px = rest[i % rest.len()];
                }
            }
        }
        // Antialiased disc: many distinct alpha levels (the alpha-pipeline
        // stress shape).
        1 => {
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            let r_out = (w.min(h) as f32) * 0.45;
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let d = (dx * dx + dy * dy).sqrt();
                    let a = ((r_out - d + 1.5) / 3.0).clamp(0.0, 1.0);
                    let i = (y * w + x) * 4;
                    rgba[i] = ((x * 255) / w.max(1)) as u8;
                    rgba[i + 1] = ((y * 255) / h.max(1)) as u8;
                    rgba[i + 2] = ((x ^ y) & 0xff) as u8;
                    rgba[i + 3] = (a * 255.0) as u8;
                }
            }
        }
        // Hard-edged sprite alpha: 0/255 blocks (ALPH filter trials).
        2 => {
            let k = 2 + (rng.next() % 6) as usize;
            let bw = w.div_ceil(k).max(1);
            let bh = h.div_ceil(k).max(1);
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let opaque = ((x / bw) + (y / bh)).is_multiple_of(2);
                    rgba[i] = rng.byte();
                    rgba[i + 1] = rng.byte();
                    rgba[i + 2] = rng.byte();
                    rgba[i + 3] = if opaque { 255 } else { 0 };
                }
            }
        }
        // Opaque noise (no ALPH chunk at all).
        _ => {
            for px in rgba.as_chunks_mut::<4>().0.iter_mut() {
                px[0] = rng.byte();
                px[1] = rng.byte();
                px[2] = rng.byte();
                px[3] = 255;
            }
        }
    }
}

/// Measured per-pixel cost of a lossless encode at each method, relative to
/// m0, on the worst-case content the generator produces (pure RGBA noise:
/// no palette, no matches, maximal histograms — both zenwebp AND libwebp
/// emit the raw pixels and burn the whole method budget getting there).
///
/// Measured 2026-08-27 on the #79 timeout seed (403x147 noise, q100, release
/// build, Apple M4 Pro, zenwebp 0.72x of libwebp wall at m6): µs/pixel
/// m0 0.11 · m1 0.10 · m2 0.31 · m3 1.25 · m4 3.8 · m5 3.8 · m6 26.5. The
/// old budget (`w*h*(1+method) <= 700_000`) modeled cost as LINEAR in method
/// and let m6 run 100k pixels: 1.6 s bare metal, which is ~70 s under the
/// fuzz build's ASAN + sancov instrumentation (#68 measured ~43x) — past the
/// farm's 25 s per-input timeout (#79). m6 is ~250x m0 per pixel, not 7x.
const LOSSLESS_METHOD_COST: [usize; 7] = [1, 1, 3, 12, 38, 38, 260];

/// Per-input work ceiling in m0-pixel-equivalents (~0.11 µs each): 1.2M ≈
/// 130 ms bare metal worst case ≈ 5-6 s instrumented, under both the 10 s
/// slow-unit report and the 25 s timeout with margin for a slow farm box.
///
/// Resulting caps: m0/m1 uncapped inside the 640x640 generator range
/// (≤ 45 ms), m2 400k px (632²), m3 100k px (316²), m4/m5 31k px (177²),
/// m6 4.6k px (68²). Every method still lands in multi-tile entropy-image
/// territory (4 px tiles at m5/m6, 8 px at m4). What this does NOT reach is
/// the m5/m6 multi-sampling predictor search, which only engages above
/// MAX_HUFF_IMAGE_SIZE = 2600 tiles (> 41.6k px at 4 px tiles); that path is
/// covered deterministically by `tests/lossless_roundtrip.rs`
/// (`m5_m6_sampling_search_roundtrips_exactly`), not by the fuzzer.
const LOSSLESS_BUDGET: usize = 1_200_000;

/// Whether a `w x h @ method` lossless cell is cheap enough to fuzz.
#[allow(dead_code)]
pub fn lossless_cell_within_budget(w: usize, h: usize, method: u8) -> bool {
    let cost = LOSSLESS_METHOD_COST[usize::from(method.min(6))];
    w.saturating_mul(h).saturating_mul(cost) <= LOSSLESS_BUDGET
}

/// Lossless encode → decode → exact pixel comparison. Panics on any
/// corruption; clean encode errors (limits) return silently.
#[allow(dead_code)]
pub fn run_encode_lossless_roundtrip(input: &[u8]) {
    if input.len() < 12 {
        return;
    }
    let w = 1 + (u16::from_le_bytes([input[0], input[1]]) % 640) as usize;
    let h = 1 + (u16::from_le_bytes([input[2], input[3]]) % 640) as usize;
    let method = input[4] % 7;
    let quality = (input[5] % 101) as f32;
    let mode = input[6];
    let seed = u32::from_le_bytes([input[7], input[8], input[9], input[10]]);
    // input[11] is a reserved axis (near-lossless has no exact oracle).

    if !lossless_cell_within_budget(w, h, method) {
        return;
    }

    let mut rgba = vec![0u8; w * h * 4];
    generate_lossless_content(&mut rgba, w, h, mode, seed, &input[12..]);

    let config = LosslessConfig::new()
        .with_method(method)
        .with_quality(quality)
        .with_exact(true);
    let Ok(webp) = EncodeRequest::new(
        &EncoderConfig::Lossless(config),
        &rgba,
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode() else {
        return;
    };

    let (pixels, ow, oh, layout) = DecodeRequest::new(&DecodeConfig::default(), &webp)
        .decode()
        .expect("lossless encoder produced an undecodable stream");
    assert_eq!((ow as usize, oh as usize), (w, h), "dimension mismatch");
    match layout {
        PixelLayout::Rgba8 => {
            assert_eq!(pixels, rgba, "lossless roundtrip corrupted pixels");
        }
        PixelLayout::Rgb8 => {
            // Encoder may drop an all-opaque alpha channel; the oracle is
            // then RGB equality + the source really being opaque.
            assert!(
                rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
                "alpha dropped from a non-opaque image"
            );
            let rgb: Vec<u8> = rgba
                .as_chunks::<4>().0.iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            assert_eq!(pixels, rgb, "lossless roundtrip corrupted pixels (rgb)");
        }
        other => panic!("unexpected decode layout {other:?}"),
    }
}

/// Lossy encode across the settings grid: no panic, stream decodes, dims
/// survive, alpha bytes roundtrip exactly at the default alpha_quality=100.
#[allow(dead_code)]
pub fn run_encode_lossy_roundtrip(input: &[u8]) {
    if input.len() < 12 {
        return;
    }
    let w = 1 + (u16::from_le_bytes([input[0], input[1]]) % 512) as usize;
    let h = 1 + (u16::from_le_bytes([input[2], input[3]]) % 512) as usize;
    let method = input[4] % 7;
    let quality = (input[5] % 101) as f32;
    let mode = input[6];
    let alpha_effort = input[7] % 8; // 7 = "unset, follow method"
    let seed = u32::from_le_bytes([input[8], input[9], input[10], input[11]]);

    if w * h * (1 + method as usize) > 700_000 {
        return;
    }

    let mut rgba = vec![0u8; w * h * 4];
    generate_lossy_content(&mut rgba, w, h, mode, seed, &input[12..]);

    let mut config = LossyConfig::new().with_quality(quality).with_method(method);
    if alpha_effort < 7 {
        config = config.with_alpha_effort(alpha_effort);
    }
    let Ok(webp) = EncodeRequest::new(
        &EncoderConfig::Lossy(config),
        &rgba,
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode() else {
        return;
    };

    let (pixels, ow, oh, layout) = DecodeRequest::new(&DecodeConfig::default(), &webp)
        .decode()
        .expect("lossy encoder produced an undecodable stream");
    assert_eq!((ow as usize, oh as usize), (w, h), "dimension mismatch");
    match layout {
        PixelLayout::Rgba8 => {
            // alpha_quality defaults to 100 (lossless alpha): decoded alpha
            // must be bit-exact.
            for (i, (px, src)) in pixels.as_chunks::<4>().0.iter().zip(rgba.as_chunks::<4>().0.iter()).enumerate() {
                assert_eq!(
                    px[3], src[3],
                    "alpha_quality=100 alpha mismatch at pixel {i}"
                );
            }
        }
        PixelLayout::Rgb8 => {
            assert!(
                rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
                "alpha dropped from a non-opaque image"
            );
        }
        other => panic!("unexpected decode layout {other:?}"),
    }
}
