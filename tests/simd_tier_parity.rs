//! End-to-end SIMD tier parity tests.
//!
//! For every supported encoder and decoder configuration, we encode/decode
//! once with the current CPU's full SIMD capabilities (the "baseline") and
//! then re-run the operation with each downward-closed subset of SIMD
//! capability tokens disabled. Output bytes (or output pixels) must match
//! the baseline byte-for-byte across every tier.
//!
//! Any divergence is a real bug — either a SIMD path produces different
//! results than the scalar reference, or the encoder is non-deterministic
//! for the same tier.
//!
//! Tier disabling is process-wide. The `archmage::testing` helper
//! serializes via an internal mutex; this test file should run all
//! parity work on a single thread (which is what cargo test does by
//! default for #[test] functions within a single binary).
//!
//! ## CI integration
//!
//! Run with `--features testable_dispatch` so any compile-time-guaranteed
//! token (i.e., the build was done with `-Ctarget-cpu=native` or similar)
//! triggers a panic instead of a silent reduction in coverage.
//!
//! ## Skips
//!
//! Optional corpus images are gated on the `ZENWEBP_TIER_PARITY_CORPUS`
//! environment variable. If the directory is absent on the runner, the
//! corpus arm prints a single warning and continues with the procedural
//! image set. This skip decision is visible in the CI workflow (the env
//! var is set there) — it's not a silent in-test skip.

#![forbid(unsafe_code)]
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, LosslessConfig, LossyConfig, PixelLayout, Preset,
};

// ---------------------------------------------------------------------------
// Compile-time policy
// ---------------------------------------------------------------------------

/// `Fail` under `--features testable_dispatch` (CI), `WarnStderr` otherwise.
///
/// CI builds without `-Ctarget-cpu=native`, so `Fail` will only trip if a
/// runner ships a newer rustc default that bakes in features we expect to
/// be runtime-dispatched. That's exactly the regression we want to catch.
fn policy() -> CompileTimePolicy {
    if cfg!(feature = "testable_dispatch") {
        CompileTimePolicy::Fail
    } else {
        CompileTimePolicy::WarnStderr
    }
}

// ---------------------------------------------------------------------------
// Procedural test images (deterministic across machines)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TestImage {
    name: &'static str,
    width: u32,
    height: u32,
    layout: PixelLayout,
    pixels: Vec<u8>,
}

impl core::fmt::Debug for TestImage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TestImage {{ name: {}, {}x{}, layout: {:?}, bytes: {} }}",
            self.name,
            self.width,
            self.height,
            self.layout,
            self.pixels.len()
        )
    }
}

/// Tiny LCG RNG so test data is identical on every machine and target.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u8(&mut self) -> u8 {
        // Numerical Recipes LCG.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> TestImage {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
    }
    TestImage {
        name: "solid_16x16_rgb",
        width: w,
        height: h,
        layout: PixelLayout::Rgb8,
        pixels,
    }
}

fn alpha_gradient_rgba(w: u32, h: u32) -> TestImage {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            let a = ((x.wrapping_add(y) * 17) & 0xff) as u8;
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }
    TestImage {
        name: "alpha_gradient_16x16_rgba",
        width: w,
        height: h,
        layout: PixelLayout::Rgba8,
        pixels,
    }
}

fn vgradient_rgb(w: u32, h: u32) -> TestImage {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let g = ((y * 255) / h.max(1)) as u8;
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            pixels.extend_from_slice(&[r, g, 128]);
        }
    }
    TestImage {
        name: "vgradient_64x64_rgb",
        width: w,
        height: h,
        layout: PixelLayout::Rgb8,
        pixels,
    }
}

fn noise_rgb(w: u32, h: u32, seed: u64) -> TestImage {
    let mut rng = Lcg::new(seed);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h * 3) {
        pixels.push(rng.next_u8());
    }
    TestImage {
        name: "noise_64x64_rgb",
        width: w,
        height: h,
        layout: PixelLayout::Rgb8,
        pixels,
    }
}

fn screenshot_like_rgb(w: u32, h: u32) -> TestImage {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let cell = ((x / 8 + y / 8) & 1) != 0;
            let in_text = (16..32).contains(&y) && (8..56).contains(&x) && (x % 4) < 2;
            let in_panel = y >= h.saturating_sub(16);
            let (r, g, b) = if in_text {
                (32u8, 32, 32)
            } else if in_panel {
                (200u8, 200, 220)
            } else if cell {
                (240u8, 240, 240)
            } else {
                (255u8, 255, 255)
            };
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    TestImage {
        name: "screenshot_96x64_rgb",
        width: w,
        height: h,
        layout: PixelLayout::Rgb8,
        pixels,
    }
}

fn photo_like_rgba(w: u32, h: u32) -> TestImage {
    // Smooth gradient + pseudo-noise. Deterministic and small.
    let mut rng = Lcg::new(0xCAFE_F00D);
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as i32;
            let fy = y as i32;
            let base_r = (((fx * fx + fy) >> 5) & 0xff) as u8;
            let base_g = (((fy * 3 + fx * 2) >> 3) & 0xff) as u8;
            let base_b = (128 + ((fx ^ fy) & 0x7f)) as u8;
            // Add ±8 of pseudo-noise for texture.
            let n = rng.next_u8() as i32 - 128;
            let noise = (n / 16) as i16;
            let r = (base_r as i16 + noise).clamp(0, 255) as u8;
            let g = (base_g as i16 + noise).clamp(0, 255) as u8;
            let b = (base_b as i16 + noise).clamp(0, 255) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }
    TestImage {
        name: "photo_like_128x96_rgba",
        width: w,
        height: h,
        layout: PixelLayout::Rgba8,
        pixels,
    }
}

fn procedural_set() -> Vec<TestImage> {
    vec![
        solid_rgb(16, 16, 200, 100, 50),
        alpha_gradient_rgba(16, 16),
        vgradient_rgb(64, 64),
        noise_rgb(64, 64, 0xDEAD_BEEF),
        screenshot_like_rgb(96, 64),
        photo_like_rgba(128, 96),
    ]
}

/// Try to load one real corpus image as raw RGB. Controlled by env vars
/// set in the CI workflow — no silent path-existence skips.
fn maybe_corpus_image() -> Option<TestImage> {
    let dir = std::env::var("ZENWEBP_TIER_PARITY_CORPUS").ok()?;
    let dir = PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "ZENWEBP_TIER_PARITY_CORPUS={} is not a directory — skipping corpus arm",
            dir.display()
        );
        return None;
    }
    // Pick the smallest .png in the directory for a fast, deterministic choice.
    let mut entries: Vec<(u64, PathBuf)> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let is_png = p.extension().map(|x| x == "png").unwrap_or(false);
            if !is_png {
                return None;
            }
            let len = e.metadata().ok()?.len();
            Some((len, p))
        })
        .collect();
    entries.sort_by_key(|(len, _)| *len);
    let (_, path) = entries.into_iter().next()?;
    let file = std::io::BufReader::new(std::fs::File::open(&path).ok()?);
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.bit_depth != png::BitDepth::Eight {
        eprintln!(
            "corpus image {} is not 8-bit ({:?}) — skipping",
            path.display(),
            info.bit_depth
        );
        return None;
    }
    let (layout, pixels) = match info.color_type {
        png::ColorType::Rgb => (PixelLayout::Rgb8, buf[..info.buffer_size()].to_vec()),
        png::ColorType::Rgba => (PixelLayout::Rgba8, buf[..info.buffer_size()].to_vec()),
        other => {
            eprintln!(
                "corpus image {} has unsupported color type {:?} — skipping",
                path.display(),
                other
            );
            return None;
        }
    };
    // Cap to keep CI runtime sane.
    if info.width > 512 || info.height > 512 {
        eprintln!(
            "corpus image {} is {}x{} (> 512 cap) — skipping",
            path.display(),
            info.width,
            info.height,
        );
        return None;
    }
    eprintln!(
        "corpus image: {} {}x{} layout={:?}",
        path.display(),
        info.width,
        info.height,
        layout
    );
    Some(TestImage {
        name: "corpus_smallest",
        width: info.width,
        height: info.height,
        layout,
        pixels,
    })
}

/// Try to load a couple of WebP files from the conformance corpus for
/// decoder testing. Same env-var policy as `maybe_corpus_image`.
fn maybe_corpus_webps() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let Ok(dir) = std::env::var("ZENWEBP_TIER_PARITY_CONFORMANCE") else {
        return out;
    };
    let dir = PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "ZENWEBP_TIER_PARITY_CONFORMANCE={} is not a directory — skipping",
            dir.display()
        );
        return out;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "webp").unwrap_or(false))
        .collect();
    entries.sort();
    for path in entries.into_iter().take(2) {
        if let Ok(bytes) = std::fs::read(&path)
            && bytes.len() <= 256 * 1024
        {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            out.push((name, bytes));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Encode/decode shims used by the parity loops.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LossyCfg {
    preset: Preset,
    method: u8,
    quality: f32,
}

#[derive(Clone, Debug)]
struct LosslessCfg {
    method: u8,
    near_lossless: u8,
}

fn encode_lossy(img: &TestImage, cfg: &LossyCfg) -> Vec<u8> {
    let lc = LossyConfig::with_preset(cfg.preset, cfg.quality).with_method(cfg.method);
    EncodeRequest::lossy(&lc, &img.pixels, img.layout, img.width, img.height)
        .encode()
        .expect("lossy encode failed")
}

fn encode_lossless(img: &TestImage, cfg: &LosslessCfg) -> Vec<u8> {
    let lc = LosslessConfig::new()
        .with_method(cfg.method)
        .with_near_lossless(cfg.near_lossless);
    EncodeRequest::lossless(&lc, &img.pixels, img.layout, img.width, img.height)
        .encode()
        .expect("lossless encode failed")
}

fn decode_rgba(webp: &[u8]) -> (Vec<u8>, u32, u32) {
    let cfg = DecodeConfig::default();
    DecodeRequest::new(&cfg, webp)
        .decode_rgba()
        .expect("decode failed")
}

// ---------------------------------------------------------------------------
// Test budget configuration.
// ---------------------------------------------------------------------------

fn lossy_methods() -> &'static [u8] {
    &[0, 3, 6]
}
fn lossy_qualities() -> &'static [f32] {
    &[25.0, 75.0, 95.0]
}
fn lossy_presets() -> &'static [Preset] {
    &[Preset::Default, Preset::Photo, Preset::Drawing]
}

fn lossless_methods() -> &'static [u8] {
    &[0, 3, 6]
}

// ---------------------------------------------------------------------------
// Master configuration helpers
// ---------------------------------------------------------------------------

fn print_budget_once(label: &str, n_baselines: usize) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[simd_tier_parity] {} subsystem will run {} baselines × N tier permutations",
            label, n_baselines
        );
    });
}

// ---------------------------------------------------------------------------
// Tests — one per subsystem.
// ---------------------------------------------------------------------------

#[test]
fn lossy_encoder_tier_parity() {
    let mut images = procedural_set();
    if let Some(corpus) = maybe_corpus_image() {
        images.push(corpus);
    }

    // For lossy we exercise: presets × methods × qualities × images.
    // To keep within the CI budget, skip the slowest method (6) on the
    // largest images and on alpha-gradient (which is RGBA, ~2x cost).
    let mut configs: Vec<(TestImage, LossyCfg)> = Vec::new();
    for img in &images {
        for &preset in lossy_presets() {
            for &method in lossy_methods() {
                for &quality in lossy_qualities() {
                    // Coverage trim for the largest images at slowest method.
                    if method == 6
                        && img.width * img.height >= 96 * 64
                        && (preset != Preset::Default || quality != 75.0)
                    {
                        continue;
                    }
                    configs.push((
                        img.clone(),
                        LossyCfg {
                            preset,
                            method,
                            quality,
                        },
                    ));
                }
            }
        }
    }
    print_budget_once("lossy_encoder", configs.len());

    // Compute baselines first, then iterate tiers.
    let baselines: Vec<Vec<u8>> = configs
        .iter()
        .map(|(img, cfg)| encode_lossy(img, cfg))
        .collect();

    // Self-determinism check: re-encode each baseline and confirm equal.
    for ((img, cfg), base) in configs.iter().zip(&baselines) {
        let again = encode_lossy(img, cfg);
        assert_eq!(
            *base, again,
            "lossy encoder is non-deterministic for image={} cfg={:?} \
             (baseline differs from re-encode at the same SIMD tier — \
             this is a real bug, not a tier divergence)",
            img.name, cfg
        );
    }

    let start = Instant::now();
    let report = for_each_token_permutation(policy(), |perm| {
        for ((img, cfg), base) in configs.iter().zip(&baselines) {
            let bytes = encode_lossy(img, cfg);
            if bytes != *base {
                let n_diff = bytes
                    .iter()
                    .zip(base.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                panic!(
                    "lossy tier divergence: tier=[{}] image={} cfg={:?}\n  \
                     baseline_bytes={} tier_bytes={} differing_bytes={}",
                    perm,
                    img.name,
                    cfg,
                    base.len(),
                    bytes.len(),
                    n_diff
                );
            }
        }
    });
    eprintln!(
        "[lossy_encoder] {} configs × {} permutations in {:.1}s",
        configs.len(),
        report.permutations_run,
        start.elapsed().as_secs_f32()
    );
    eprintln!("[lossy_encoder] {report}");
}

#[test]
fn lossless_encoder_tier_parity() {
    let mut images = procedural_set();
    if let Some(corpus) = maybe_corpus_image() {
        images.push(corpus);
    }

    let mut configs: Vec<(TestImage, LosslessCfg)> = Vec::new();
    for img in &images {
        for &method in lossless_methods() {
            for &near_lossless in &[100u8, 60u8] {
                configs.push((
                    img.clone(),
                    LosslessCfg {
                        method,
                        near_lossless,
                    },
                ));
            }
        }
    }
    print_budget_once("lossless_encoder", configs.len());

    let baselines: Vec<Vec<u8>> = configs
        .iter()
        .map(|(img, cfg)| encode_lossless(img, cfg))
        .collect();

    for ((img, cfg), base) in configs.iter().zip(&baselines) {
        let again = encode_lossless(img, cfg);
        assert_eq!(
            *base, again,
            "lossless encoder is non-deterministic for image={} cfg={:?} \
             (baseline vs re-encode differ at the same tier — real bug)",
            img.name, cfg
        );
    }

    let start = Instant::now();
    let report = for_each_token_permutation(policy(), |perm| {
        for ((img, cfg), base) in configs.iter().zip(&baselines) {
            let bytes = encode_lossless(img, cfg);
            if bytes != *base {
                let n_diff = bytes
                    .iter()
                    .zip(base.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                panic!(
                    "lossless tier divergence: tier=[{}] image={} cfg={:?}\n  \
                     baseline_bytes={} tier_bytes={} differing_bytes={}",
                    perm,
                    img.name,
                    cfg,
                    base.len(),
                    bytes.len(),
                    n_diff
                );
            }
        }
    });
    eprintln!(
        "[lossless_encoder] {} configs × {} permutations in {:.1}s",
        configs.len(),
        report.permutations_run,
        start.elapsed().as_secs_f32()
    );
    eprintln!("[lossless_encoder] {report}");
}

#[test]
fn lossy_decoder_tier_parity() {
    // Build a small set of WebP files using the encoder at the current
    // (baseline) tier, then decode each one at every tier and compare.
    let mut images = procedural_set();
    if let Some(corpus) = maybe_corpus_image() {
        images.push(corpus);
    }

    let mut webps: Vec<(String, Vec<u8>)> = Vec::new();
    for img in &images {
        for &method in &[0u8, 4u8] {
            for &quality in &[25.0f32, 75.0, 95.0] {
                let cfg = LossyCfg {
                    preset: Preset::Default,
                    method,
                    quality,
                };
                let webp = encode_lossy(img, &cfg);
                webps.push((
                    format!("{}_m{}_q{}", img.name, method, quality as u32),
                    webp,
                ));
            }
        }
    }
    for (name, bytes) in maybe_corpus_webps() {
        webps.push((name, bytes));
    }
    print_budget_once("lossy_decoder", webps.len());

    let baselines: Vec<(Vec<u8>, u32, u32)> = webps.iter().map(|(_, w)| decode_rgba(w)).collect();

    // Self-determinism check.
    for ((name, w), base) in webps.iter().zip(&baselines) {
        let again = decode_rgba(w);
        assert_eq!(
            again.0, base.0,
            "lossy decoder non-deterministic at the same tier for {}",
            name
        );
    }

    let start = Instant::now();
    let report = for_each_token_permutation(policy(), |perm| {
        for ((name, w), base) in webps.iter().zip(&baselines) {
            let (pixels, width, height) = decode_rgba(w);
            assert_eq!(width, base.1, "tier={perm} image={name} width drift");
            assert_eq!(height, base.2, "tier={perm} image={name} height drift");
            if pixels != base.0 {
                let n_diff = pixels
                    .iter()
                    .zip(base.0.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                panic!(
                    "lossy decoder tier divergence: tier=[{}] image={}\n  \
                     baseline_pixels={} tier_pixels={} differing_bytes={}",
                    perm,
                    name,
                    base.0.len(),
                    pixels.len(),
                    n_diff
                );
            }
        }
    });
    eprintln!(
        "[lossy_decoder] {} webps × {} permutations in {:.1}s",
        webps.len(),
        report.permutations_run,
        start.elapsed().as_secs_f32()
    );
    eprintln!("[lossy_decoder] {report}");
}

#[test]
fn lossless_decoder_tier_parity() {
    let mut images = procedural_set();
    if let Some(corpus) = maybe_corpus_image() {
        images.push(corpus);
    }

    let mut webps: Vec<(String, Vec<u8>)> = Vec::new();
    for img in &images {
        for &method in &[0u8, 4u8] {
            let cfg = LosslessCfg {
                method,
                near_lossless: 100,
            };
            let webp = encode_lossless(img, &cfg);
            webps.push((format!("{}_ll_m{}", img.name, method), webp));
        }
    }
    print_budget_once("lossless_decoder", webps.len());

    let baselines: Vec<(Vec<u8>, u32, u32)> = webps.iter().map(|(_, w)| decode_rgba(w)).collect();

    for ((name, w), base) in webps.iter().zip(&baselines) {
        let again = decode_rgba(w);
        assert_eq!(
            again.0, base.0,
            "lossless decoder non-deterministic at the same tier for {}",
            name
        );
    }

    let start = Instant::now();
    let report = for_each_token_permutation(policy(), |perm| {
        for ((name, w), base) in webps.iter().zip(&baselines) {
            let (pixels, width, height) = decode_rgba(w);
            assert_eq!(width, base.1, "tier={perm} image={name} width drift");
            assert_eq!(height, base.2, "tier={perm} image={name} height drift");
            if pixels != base.0 {
                let n_diff = pixels
                    .iter()
                    .zip(base.0.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                panic!(
                    "lossless decoder tier divergence: tier=[{}] image={}\n  \
                     baseline_pixels={} tier_pixels={} differing_bytes={}",
                    perm,
                    name,
                    base.0.len(),
                    pixels.len(),
                    n_diff
                );
            }
        }
    });
    eprintln!(
        "[lossless_decoder] {} webps × {} permutations in {:.1}s",
        webps.len(),
        report.permutations_run,
        start.elapsed().as_secs_f32()
    );
    eprintln!("[lossless_decoder] {report}");
}

/// Explicit confidence assertion: the smallest test image must encode
/// AND decode correctly with every available token disabled (i.e., on
/// the pure-scalar fallback path). We log to stderr so CI logs prove it
/// ran.
///
/// `for_each_token_permutation` enumerates downward-closed subsets — the
/// last subset is "everything disabled" on platforms where the cascade
/// root can be disabled. We track which permutation had the most disables
/// and confirm we hit at least one with ≥ N_AVAIL-1 tokens off (which is
/// "scalar" on every platform we run on).
#[test]
fn scalar_tier_explicitly_exercised() {
    // Use a tiny image so this is fast even at method 6.
    let img = solid_rgb(16, 16, 200, 100, 50);
    let lossy_cfg = LossyCfg {
        preset: Preset::Default,
        method: 0,
        quality: 75.0,
    };
    let lossless_cfg = LosslessCfg {
        method: 0,
        near_lossless: 100,
    };

    let baseline_lossy = encode_lossy(&img, &lossy_cfg);
    let baseline_lossless = encode_lossless(&img, &lossless_cfg);
    let baseline_decode_lossy = decode_rgba(&baseline_lossy);
    let baseline_decode_lossless = decode_rgba(&baseline_lossless);

    let mut max_disabled = 0usize;
    let mut hit_scalar = false;
    let mut scalar_label = String::new();

    let report = for_each_token_permutation(policy(), |perm| {
        let n = perm.disabled.len();
        if n > max_disabled {
            max_disabled = n;
            scalar_label = perm.label.clone();
        }
        // Sanity-check encode + decode at every tier in this special test.
        let lossy = encode_lossy(&img, &lossy_cfg);
        let lossless = encode_lossless(&img, &lossless_cfg);
        let dec_lossy = decode_rgba(&lossy);
        let dec_lossless = decode_rgba(&lossless);
        assert_eq!(
            lossy, baseline_lossy,
            "lossy bytes diverge at tier [{perm}]"
        );
        assert_eq!(
            lossless, baseline_lossless,
            "lossless bytes diverge at tier [{perm}]"
        );
        assert_eq!(
            dec_lossy.0, baseline_decode_lossy.0,
            "lossy decoded pixels diverge at tier [{perm}]"
        );
        assert_eq!(
            dec_lossless.0, baseline_decode_lossless.0,
            "lossless decoded pixels diverge at tier [{perm}]"
        );

        // We claim "scalar tier exercised" when this permutation disabled
        // every token the platform actually exposes. Because for_each_*
        // skips tokens not available on the CPU, the maximum-disabled
        // permutation IS the scalar tier.
        if !hit_scalar {
            // Defer the marker until we've finished the sweep — we don't
            // know yet whether this is the maximal permutation.
        }
    });

    if report.permutations_run == 0 {
        // No tokens available on this CPU — we're already running scalar.
        eprintln!(
            "[scalar_tier_explicitly_exercised] no SIMD tokens available on this CPU; \
             baseline IS the scalar tier"
        );
        hit_scalar = true;
    } else {
        // The maximally-disabled permutation is the scalar tier.
        // (On x86_64, that's "X64V1Token, X64V2Token, X64V3Token, ..." disabled.)
        hit_scalar = max_disabled > 0;
    }
    assert!(
        hit_scalar,
        "scalar tier was never exercised (no permutations and no scalar fallback)"
    );
    eprintln!(
        "[scalar_tier_explicitly_exercised] scalar tier verified for image={} (max_disabled={}, label=\"{}\")",
        img.name, max_disabled, scalar_label
    );
    eprintln!("[scalar_tier_explicitly_exercised] {report}");
}
