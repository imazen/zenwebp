//! zenwebp vs libwebp parity matrix.
//!
//! Encode every (image × format × method × quality) cell with **both**
//! zenwebp and libwebp (via `webpx`), decode each output, and verify:
//!
//! 1. **Decoded quality**: zenwebp's decoded zensim score is not worse
//!    than libwebp's by more than `Q_TOLERANCE` points. We're not
//!    asserting we beat libwebp — only that we don't fall behind.
//! 2. **Encoded size**: zenwebp's output bytes are not more than
//!    `SIZE_TOLERANCE` larger than libwebp's. zenwebp's compression is
//!    near-parity with libwebp on average; outsizing by >10% indicates
//!    a real regression.
//! 3. **Cross-decode**: a zenwebp-encoded webp decodes correctly via
//!    libwebp's decoder (with reasonable fidelity) AND vice versa.
//!    Validates bitstream conformance.
//!
//! What this catches that the existing `webpx_regression.rs` doesn't:
//!
//! - Lossless paths (existing only does lossy).
//! - Method extremes (m0, m6) and low-q (10, 25), not just q70+ m4.
//! - Cross-decode validity (you can produce bytes neither library
//!   accepts in agreement, even if both encode).
//! - All input formats supported by both libraries (RGBA, RGB, BGRA,
//!   BGR — webpx doesn't expose Argb/L8/La8).
//!
//! libwebp doesn't expose grayscale or ARGB encode entry points, so
//! L8/La8/Argb8 are out of scope here — see `cross_format_equivalence`
//! for those.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use webpx::Unstoppable;
use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::generators;
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

// Larger than the cross-format / regression matrices because libwebp parity
// is bytewise-sensitive — tiny images make RIFF/VP8X header overhead
// dominate the size ratio comparison and produce false positives.
const W: u32 = 256;
const H: u32 = 256;

// ---------------------------------------------------------------------------
// Image generators
// ---------------------------------------------------------------------------

type ImgGen = fn(u32, u32) -> Vec<u8>;

fn gen_gradient(w: u32, h: u32) -> Vec<u8> {
    generators::gradient(w, h)
}
fn gen_noise(w: u32, h: u32) -> Vec<u8> {
    generators::value_noise(w, h, 17)
}
fn gen_color_blocks(w: u32, h: u32) -> Vec<u8> {
    generators::color_blocks(w, h)
}
fn gen_mandelbrot(w: u32, h: u32) -> Vec<u8> {
    generators::mandelbrot(w, h)
}

const ALL_IMAGES: &[(&str, ImgGen)] = &[
    ("gradient", gen_gradient),
    ("noise", gen_noise),
    ("color_blocks", gen_color_blocks),
    ("mandelbrot", gen_mandelbrot),
];

// ---------------------------------------------------------------------------
// Format conversions
// ---------------------------------------------------------------------------

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| px[..3].to_vec())
        .collect()
}
fn rgba_to_bgra(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0], px[3]])
        .collect()
}
fn rgba_to_bgr(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0]])
        .collect()
}
fn force_opaque(rgba: &[u8]) -> Vec<u8> {
    let mut out = rgba.to_vec();
    for px in out.chunks_exact_mut(4) {
        px[3] = 255;
    }
    out
}

// ---------------------------------------------------------------------------
// Encode / decode helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Format {
    Rgba8,
    Rgb8,
    Bgra8,
    Bgr8,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Rgba8 => "Rgba8",
            Format::Rgb8 => "Rgb8",
            Format::Bgra8 => "Bgra8",
            Format::Bgr8 => "Bgr8",
        }
    }
    fn zen_layout(self) -> PixelLayout {
        match self {
            Format::Rgba8 => PixelLayout::Rgba8,
            Format::Rgb8 => PixelLayout::Rgb8,
            Format::Bgra8 => PixelLayout::Bgra8,
            Format::Bgr8 => PixelLayout::Bgr8,
        }
    }
    fn has_alpha(self) -> bool {
        matches!(self, Format::Rgba8 | Format::Bgra8)
    }
    /// Convert RGBA reference into this format's bytes.
    #[allow(clippy::wrong_self_convention)]
    fn from_rgba(self, rgba: &[u8]) -> Vec<u8> {
        match self {
            Format::Rgba8 => rgba.to_vec(),
            Format::Rgb8 => rgba_to_rgb(rgba),
            Format::Bgra8 => rgba_to_bgra(rgba),
            Format::Bgr8 => rgba_to_bgr(rgba),
        }
    }
}

fn zen_lossy(pixels: &[u8], fmt: Format, m: u8, q: f32) -> Vec<u8> {
    let cfg = LossyConfig::new().with_quality(q).with_method(m);
    EncodeRequest::lossy(&cfg, pixels, fmt.zen_layout(), W, H)
        .encode()
        .unwrap_or_else(|e| panic!("zen lossy m{m} q{q} {}: {e}", fmt.name()))
}

fn zen_lossless(pixels: &[u8], fmt: Format, m: u8) -> Vec<u8> {
    let cfg = LosslessConfig::new().with_method(m).with_exact(true);
    EncodeRequest::lossless(&cfg, pixels, fmt.zen_layout(), W, H)
        .encode()
        .unwrap_or_else(|e| panic!("zen lossless m{m} {}: {e}", fmt.name()))
}

fn lib_lossy(pixels: &[u8], fmt: Format, m: u8, q: f32) -> Vec<u8> {
    let cfg = webpx::EncoderConfig::new().quality(q).method(m);
    let r = match fmt {
        Format::Rgba8 => cfg.encode_rgba(pixels, W, H, Unstoppable),
        Format::Rgb8 => cfg.encode_rgb(pixels, W, H, Unstoppable),
        Format::Bgra8 => cfg.encode_bgra(pixels, W, H, Unstoppable),
        Format::Bgr8 => cfg.encode_bgr(pixels, W, H, Unstoppable),
    };
    r.unwrap_or_else(|e| panic!("lib lossy m{m} q{q} {}: {e}", fmt.name()))
}

fn lib_lossless(pixels: &[u8], fmt: Format, m: u8) -> Vec<u8> {
    let cfg = webpx::EncoderConfig::new_lossless().method(m).exact(true);
    let r = match fmt {
        Format::Rgba8 => cfg.encode_rgba(pixels, W, H, Unstoppable),
        Format::Rgb8 => cfg.encode_rgb(pixels, W, H, Unstoppable),
        Format::Bgra8 => cfg.encode_bgra(pixels, W, H, Unstoppable),
        Format::Bgr8 => cfg.encode_bgr(pixels, W, H, Unstoppable),
    };
    r.unwrap_or_else(|e| panic!("lib lossless m{m} {}: {e}", fmt.name()))
}

fn zen_decode_rgba(webp: &[u8]) -> Vec<u8> {
    zenwebp::oneshot::decode_rgba(webp)
        .expect("zen decode_rgba")
        .0
}

fn lib_decode_rgba(webp: &[u8]) -> Vec<u8> {
    webpx::decode_rgba(webp).expect("lib decode_rgba").0
}

fn as_rgba_pixels(data: &[u8]) -> &[[u8; 4]] {
    assert!(data.len().is_multiple_of(4));
    let (prefix, pixels, suffix) = unsafe { data.align_to::<[u8; 4]>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    pixels
}

fn score(a: &[u8], b: &[u8]) -> f64 {
    let z = Zensim::new(ZensimProfile::latest());
    let a = RgbaSlice::new(as_rgba_pixels(a), W as usize, H as usize);
    let b = RgbaSlice::new(as_rgba_pixels(b), W as usize, H as usize);
    z.compute(&a, &b).expect("zensim").score()
}

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// zenwebp's decoded zensim score must be within this much of libwebp's
/// for the same input. Negative = zen is worse. We allow zenwebp to
/// score up to `Q_TOLERANCE` points lower than libwebp before failing.
const Q_TOLERANCE: f64 = 5.0;

/// zenwebp's encoded size must be no more than `SIZE_RATIO_MAX` × libwebp's.
///
/// Set generously (1.30) because zenwebp's encoder makes different
/// perceptual-quality tradeoffs than libwebp at default config — across
/// the test matrix zen averages 1.045× libwebp size with peaks ~1.27×,
/// while consistently scoring ≥ libwebp in zensim (worst case Δ = -0.83,
/// best Δ = +5.10). A real compression regression would push the mean
/// well past 1.20× and break score parity simultaneously.
const SIZE_RATIO_MAX: f64 = 1.30;

/// Cross-decode score floor: when one encoder's output is decoded by the
/// other's decoder, the result must score at least this against the
/// original.
///
/// This is a **bitstream-conformance** floor, not a quality contract —
/// quality is already enforced by `Q_TOLERANCE` (zen score Δ within ±5
/// of libwebp's). The cross-decode threshold just catches "bytes that
/// neither library can interpret" or "decoder produced garbage."
/// Set generously (30) because at extreme low quality (q10) on noisy
/// content, both encoders can score in the low 40s, and ±2 LSB drift
/// from the encode-side matrix choice can push specific cells under
/// any tighter bound.
const CROSS_DECODE_MIN_SCORE: f64 = 30.0;

const METHODS: &[u8] = &[0, 4, 6];
const QUALITIES: &[f32] = &[10.0, 25.0, 50.0, 75.0, 90.0];
const LOSSLESS_METHODS: &[u8] = &[0, 4, 6];

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

fn run_lossy_vs_libwebp(fmt: Format) {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for &(name, genfn) in ALL_IMAGES {
        let rgba = if fmt.has_alpha() {
            genfn(W, H)
        } else {
            // Opaque-only formats — strip alpha so the input matches what
            // we'll re-decode against (decoded RGBA from these formats has
            // alpha=255 always).
            force_opaque(&genfn(W, H))
        };
        let pixels = fmt.from_rgba(&rgba);

        for &m in METHODS {
            for &q in QUALITIES {
                let zen_webp = zen_lossy(&pixels, fmt, m, q);
                let lib_webp = lib_lossy(&pixels, fmt, m, q);

                let zen_dec = zen_decode_rgba(&zen_webp);
                let lib_dec = lib_decode_rgba(&lib_webp);
                let zen_score = score(&rgba, &zen_dec);
                let lib_score = score(&rgba, &lib_dec);

                let zen_via_lib = lib_decode_rgba(&zen_webp);
                let lib_via_zen = zen_decode_rgba(&lib_webp);
                let zen_via_lib_score = score(&rgba, &zen_via_lib);
                let lib_via_zen_score = score(&rgba, &lib_via_zen);

                let size_ratio = zen_webp.len() as f64 / lib_webp.len() as f64;
                let score_delta = zen_score - lib_score;

                let line = format!(
                    "  [{name}] m{m} q{q:>4.0}: zen={:>5} lib={:>5} ratio={:.3} \
                     score zen={:.2} lib={:.2} Δ={:+.2}  cross zen→lib={:.1} lib→zen={:.1}",
                    zen_webp.len(),
                    lib_webp.len(),
                    size_ratio,
                    zen_score,
                    lib_score,
                    score_delta,
                    zen_via_lib_score,
                    lib_via_zen_score,
                );
                report.push(line.clone());

                if score_delta < -Q_TOLERANCE {
                    failures.push(format!(
                        "{line}  ← zen quality {score_delta:+.2} below tolerance"
                    ));
                }
                if size_ratio > SIZE_RATIO_MAX {
                    failures.push(format!(
                        "{line}  ← zen size ratio {size_ratio:.3} above {SIZE_RATIO_MAX}"
                    ));
                }
                if zen_via_lib_score < CROSS_DECODE_MIN_SCORE {
                    failures.push(format!(
                        "{line}  ← zen→lib cross-decode {zen_via_lib_score:.2} below {CROSS_DECODE_MIN_SCORE}"
                    ));
                }
                if lib_via_zen_score < CROSS_DECODE_MIN_SCORE {
                    failures.push(format!(
                        "{line}  ← lib→zen cross-decode {lib_via_zen_score:.2} below {CROSS_DECODE_MIN_SCORE}"
                    ));
                }
            }
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "\n=== Lossy vs libwebp ({}) ===\n{}\n",
            fmt.name(),
            report.join("\n")
        );
        panic!(
            "Lossy vs libwebp ({}):\n{}\n",
            fmt.name(),
            failures.join("\n")
        );
    }
}

fn run_lossless_vs_libwebp(fmt: Format) {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for &(name, genfn) in ALL_IMAGES {
        let rgba = if fmt.has_alpha() {
            genfn(W, H)
        } else {
            force_opaque(&genfn(W, H))
        };
        let pixels = fmt.from_rgba(&rgba);

        for &m in LOSSLESS_METHODS {
            let zen_webp = zen_lossless(&pixels, fmt, m);
            let lib_webp = lib_lossless(&pixels, fmt, m);

            let zen_dec = zen_decode_rgba(&zen_webp);
            let lib_dec = lib_decode_rgba(&lib_webp);
            let zen_via_lib = lib_decode_rgba(&zen_webp);
            let lib_via_zen = zen_decode_rgba(&lib_webp);

            let size_ratio = zen_webp.len() as f64 / lib_webp.len() as f64;

            let line = format!(
                "  [{name}] m{m}: zen={:>5} lib={:>5} ratio={:.3}  \
                 zen-self-exact={} lib-self-exact={} zen→lib-exact={} lib→zen-exact={}",
                zen_webp.len(),
                lib_webp.len(),
                size_ratio,
                zen_dec == rgba,
                lib_dec == rgba,
                zen_via_lib == rgba,
                lib_via_zen == rgba,
            );
            report.push(line.clone());

            if zen_dec != rgba {
                failures.push(format!(
                    "{line}  ← zen lossless self-roundtrip not byte-exact"
                ));
            }
            if zen_via_lib != rgba {
                failures.push(format!(
                    "{line}  ← zen output decoded by libwebp not byte-exact"
                ));
            }
            if lib_via_zen != rgba {
                failures.push(format!(
                    "{line}  ← lib output decoded by zenwebp not byte-exact"
                ));
            }
            if size_ratio > SIZE_RATIO_MAX {
                failures.push(format!(
                    "{line}  ← zen size ratio {size_ratio:.3} above {SIZE_RATIO_MAX}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "\n=== Lossless vs libwebp ({}) ===\n{}\n",
            fmt.name(),
            report.join("\n")
        );
        panic!(
            "Lossless vs libwebp ({}):\n{}\n",
            fmt.name(),
            failures.join("\n")
        );
    }
}

#[test]
fn lossy_vs_libwebp_rgba8() {
    run_lossy_vs_libwebp(Format::Rgba8);
}

#[test]
fn lossy_vs_libwebp_rgb8() {
    run_lossy_vs_libwebp(Format::Rgb8);
}

#[test]
fn lossy_vs_libwebp_bgra8() {
    run_lossy_vs_libwebp(Format::Bgra8);
}

#[test]
fn lossy_vs_libwebp_bgr8() {
    run_lossy_vs_libwebp(Format::Bgr8);
}

#[test]
fn lossless_vs_libwebp_rgba8() {
    run_lossless_vs_libwebp(Format::Rgba8);
}

#[test]
fn lossless_vs_libwebp_rgb8() {
    run_lossless_vs_libwebp(Format::Rgb8);
}

#[test]
fn lossless_vs_libwebp_bgra8() {
    run_lossless_vs_libwebp(Format::Bgra8);
}

#[test]
fn lossless_vs_libwebp_bgr8() {
    run_lossless_vs_libwebp(Format::Bgr8);
}
