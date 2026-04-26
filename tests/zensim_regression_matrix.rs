//! Encode→decode roundtrip score floor across the full
//! (image × format × method × quality) matrix.
//!
//! For each (image, format, config), encode through zenwebp and decode
//! back to RGBA, then score the decoded output against the original with
//! zensim. Each cell has a pinned **floor**; falling below it is a
//! regression. Floors are deliberately conservative — they're not a
//! quality target, they're a tripwire for "the encoder regressed and
//! produces visibly worse output than it did when these tests were
//! pinned."
//!
//! What this catches that the existing roundtrip tests don't:
//!
//! - Lossy quality at low q (10–25) — the prior tests skewed high (q70+).
//! - All input formats (Rgb8, Rgba8, Bgra8, Bgr8, Argb8, L8, La8), not
//!   just Rgba8/Rgb8.
//! - Method extremes (m0 + m6) for every (image, format).
//! - Lossless byte-exact roundtrip, every format.
//!
//! Floors are hit-or-miss for synthetic content (the gradient + checker
//! are easier than real photos), so we set them per-image based on
//! observed scores plus a 5-point margin. Adjust upward over time as
//! the encoder improves.
//!
//! Failures print every (image, config) cell so a single regression
//! shows you exactly where it landed.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::generators;
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

const W: u32 = 96;
const H: u32 = 96;

// ---------------------------------------------------------------------------
// Image generators (RGBA)
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
fn gen_checker(w: u32, h: u32) -> Vec<u8> {
    generators::checkerboard(w, h, 8)
}

const ALL_IMAGES: &[(&str, ImgGen)] = &[
    ("gradient", gen_gradient),
    ("noise", gen_noise),
    ("color_blocks", gen_color_blocks),
    ("mandelbrot", gen_mandelbrot),
    ("checker", gen_checker),
];

// Each image gets an opaque variant (alpha=255) for formats that don't
// carry alpha, and a gray variant for L8/La8.

fn rgba_to_l8(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|px| {
            // Project to luminance via standard rec601 weights so we get a
            // genuine grayscale image rather than collapsing only the R
            // channel — keeps L8 testing meaningful for non-gray inputs.
            (0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32) as u8
        })
        .collect()
}

fn l8_to_rgba(l8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(l8.len() * 4);
    for &g in l8 {
        out.extend_from_slice(&[g, g, g, 255]);
    }
    out
}

fn rgba_to_la8(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| {
            let y = (0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32) as u8;
            [y, px[3]]
        })
        .collect()
}

fn la8_to_rgba(la8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(la8.len() * 2);
    for px in la8.chunks_exact(2) {
        out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
    }
    out
}

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

fn rgba_to_argb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[3], px[0], px[1], px[2]])
        .collect()
}

fn rgba_to_bgr(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0]])
        .collect()
}

// ---------------------------------------------------------------------------
// Encode / decode / score helpers
// ---------------------------------------------------------------------------

fn enc_lossy(pixels: &[u8], layout: PixelLayout, w: u32, h: u32, m: u8, q: f32) -> Vec<u8> {
    let cfg = LossyConfig::new().with_quality(q).with_method(m);
    EncodeRequest::lossy(&cfg, pixels, layout, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("lossy m{m} q{q} {layout}: {e}"))
}

fn enc_lossless(pixels: &[u8], layout: PixelLayout, w: u32, h: u32, m: u8) -> Vec<u8> {
    let cfg = LosslessConfig::new().with_method(m).with_exact(true);
    EncodeRequest::lossless(&cfg, pixels, layout, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("lossless m{m} {layout}: {e}"))
}

fn dec_rgba(webp: &[u8]) -> Vec<u8> {
    zenwebp::oneshot::decode_rgba(webp).expect("decode_rgba").0
}

fn as_rgba_pixels(data: &[u8]) -> &[[u8; 4]] {
    assert!(data.len().is_multiple_of(4));
    let (prefix, pixels, suffix) = unsafe { data.align_to::<[u8; 4]>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    pixels
}

fn score(a: &[u8], b: &[u8], w: u32, h: u32) -> f64 {
    let z = Zensim::new(ZensimProfile::latest());
    let a = RgbaSlice::new(as_rgba_pixels(a), w as usize, h as usize);
    let b = RgbaSlice::new(as_rgba_pixels(b), w as usize, h as usize);
    z.compute(&a, &b).expect("zensim compute").score()
}

// ---------------------------------------------------------------------------
// Floors
// ---------------------------------------------------------------------------

const METHODS: &[u8] = &[0, 4, 6];
const LOSSLESS_METHODS: &[u8] = &[0, 4, 6];

/// Quality bands. Per-image floors below set the actual tripwire — these
/// are just the q values we sweep.
const LOSSY_QUALITIES: &[f32] = &[10.0, 25.0, 50.0, 75.0, 90.0];

/// Per-image, per-quality minimum zensim score. Initial values come from
/// running the matrix once and recording the worst score across all
/// formats and methods, minus a 3-point margin. Adjust upward as the
/// encoder improves; do **not** lower without explicit reason — every
/// reduction is a documented regression.
///
/// Floor table layout: `(image_name, [(q, min_score), ...])`.
const FLOORS: &[(&str, &[(f32, f64)])] = &[
    (
        "gradient",
        // Smooth diagonal gradient — easy for VP8 to compress, high scores.
        &[
            (10.0, 67.0),
            (25.0, 75.0),
            (50.0, 71.0),
            (75.0, 79.0),
            (90.0, 83.0),
        ],
    ),
    (
        "noise",
        // Value noise — moderate frequency, scores climb steadily with q.
        &[
            (10.0, 39.0),
            (25.0, 56.0),
            (50.0, 67.0),
            (75.0, 73.0),
            (90.0, 84.0),
        ],
    ),
    (
        "color_blocks",
        // Sharp color edges between flat regions — hard for VP8 lossy:
        // chroma quant blurs edges and the score plateaus around 50.
        &[
            (10.0, 40.0),
            (25.0, 43.0),
            (50.0, 46.0),
            (75.0, 47.0),
            (90.0, 47.0),
        ],
    ),
    (
        "mandelbrot",
        // High-frequency fractal detail with smooth interior.
        &[
            (10.0, 40.0),
            (25.0, 52.0),
            (50.0, 58.0),
            (75.0, 64.0),
            (90.0, 68.0),
        ],
    ),
    (
        "checker",
        // Binary high-frequency — adversarial for lossy. Scores never
        // climb past ~60 even at q90 because chroma subsampling
        // fundamentally can't represent 1-pixel checkerboards exactly.
        &[
            (10.0, 50.0),
            (25.0, 53.0),
            (50.0, 53.0),
            (75.0, 55.0),
            (90.0, 57.0),
        ],
    ),
];

fn floor_for(image: &str, q: f32) -> f64 {
    let (_, qs) = FLOORS
        .iter()
        .find(|(name, _)| *name == image)
        .unwrap_or_else(|| panic!("no floor table for image {image}"));
    qs.iter()
        .find(|(qq, _)| (*qq - q).abs() < 0.01)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("no floor for image {image} at q={q}"))
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

struct LossyMatrixCase<'a> {
    name: &'a str,
    /// Generator (returns RGBA reference); converter (RGBA → encoder bytes).
    /// `decoded_to_rgba` undoes the layout for scoring against the original
    /// reference (which is always evaluated in RGBA space).
    layout: PixelLayout,
    rgba_to_layout: fn(&[u8]) -> Vec<u8>,
    /// Returns the "ideal lossless decode" for this layout in RGBA so we
    /// can score against the same reference for any input format.
    /// (Identity for Rgba8; gray-replicate for L8; force-opaque for Rgb8.)
    layout_to_reference_rgba: fn(&[u8]) -> Vec<u8>,
}

fn run_lossy_matrix(case: &LossyMatrixCase) {
    let mut failures = Vec::new();
    let mut report = Vec::new();
    for &(name, genfn) in ALL_IMAGES {
        let rgba_orig = genfn(W, H);
        let layout_pixels = (case.rgba_to_layout)(&rgba_orig);
        let reference = (case.layout_to_reference_rgba)(&layout_pixels);

        for &m in METHODS {
            for &q in LOSSY_QUALITIES {
                let min_score = floor_for(name, q);
                let webp = enc_lossy(&layout_pixels, case.layout, W, H, m, q);
                let decoded = dec_rgba(&webp);
                let s = score(&reference, &decoded, W, H);
                let line = format!(
                    "  [{name}] m{m} q{q:>4.0}: {:>5} bytes, score={:>5.2} (floor {:.0})",
                    webp.len(),
                    s,
                    min_score
                );
                report.push(line.clone());
                if s < min_score {
                    failures.push(format!("{line}  ← REGRESSION"));
                }
            }
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "\n=== Lossy roundtrip matrix: {} ===\n{}\n",
            case.name,
            report.join("\n")
        );
        panic!(
            "Lossy roundtrip regression: {}\n{}\n",
            case.name,
            failures.join("\n")
        );
    }
}

fn run_lossless_matrix(case: &LossyMatrixCase) {
    // Lossless is byte-exact: decoded RGBA must equal the reference exactly.
    let mut failures = Vec::new();
    for &(name, genfn) in ALL_IMAGES {
        let rgba_orig = genfn(W, H);
        let layout_pixels = (case.rgba_to_layout)(&rgba_orig);
        let reference = (case.layout_to_reference_rgba)(&layout_pixels);

        for &m in LOSSLESS_METHODS {
            let webp = enc_lossless(&layout_pixels, case.layout, W, H, m);
            let decoded = dec_rgba(&webp);
            if decoded != reference {
                let mut diff = 0u32;
                let mut maxd = [0u16; 4];
                for (a, b) in reference.chunks_exact(4).zip(decoded.chunks_exact(4)) {
                    if a != b {
                        diff += 1;
                    }
                    for c in 0..4 {
                        maxd[c] = maxd[c].max((a[c] as i16 - b[c] as i16).unsigned_abs());
                    }
                }
                failures.push(format!(
                    "  [{name}] m{m}: {diff} pixels differ, max ΔR={} ΔG={} ΔB={} ΔA={} ({} bytes)",
                    maxd[0],
                    maxd[1],
                    maxd[2],
                    maxd[3],
                    webp.len()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n=== Lossless roundtrip not byte-exact: {} ===\n{}\n",
        case.name,
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Helper layout converters that round-trip through "ideal lossless" reference
// ---------------------------------------------------------------------------

fn to_rgba_id(p: &[u8]) -> Vec<u8> {
    p.to_vec()
}
fn rgb_to_opaque_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}
fn bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgra.len());
    for px in bgra.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    out
}
fn argb_to_rgba(argb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(argb.len());
    for px in argb.chunks_exact(4) {
        out.extend_from_slice(&[px[1], px[2], px[3], px[0]]);
    }
    out
}
fn bgr_to_opaque_rgba(bgr: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgr.len() / 3 * 4);
    for px in bgr.chunks_exact(3) {
        out.extend_from_slice(&[px[2], px[1], px[0], 255]);
    }
    out
}

// ---------------------------------------------------------------------------
// Lossy matrix tests — one per format
// ---------------------------------------------------------------------------

#[test]
fn lossy_matrix_rgba8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "Rgba8",
        layout: PixelLayout::Rgba8,
        rgba_to_layout: to_rgba_id,
        layout_to_reference_rgba: to_rgba_id,
    });
}

#[test]
fn lossy_matrix_bgra8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "Bgra8",
        layout: PixelLayout::Bgra8,
        rgba_to_layout: rgba_to_bgra,
        layout_to_reference_rgba: bgra_to_rgba,
    });
}

#[test]
fn lossy_matrix_argb8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "Argb8",
        layout: PixelLayout::Argb8,
        rgba_to_layout: rgba_to_argb,
        layout_to_reference_rgba: argb_to_rgba,
    });
}

#[test]
fn lossy_matrix_rgb8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "Rgb8",
        layout: PixelLayout::Rgb8,
        rgba_to_layout: rgba_to_rgb,
        layout_to_reference_rgba: rgb_to_opaque_rgba,
    });
}

#[test]
fn lossy_matrix_bgr8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "Bgr8",
        layout: PixelLayout::Bgr8,
        rgba_to_layout: rgba_to_bgr,
        layout_to_reference_rgba: bgr_to_opaque_rgba,
    });
}

#[test]
fn lossy_matrix_l8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "L8",
        layout: PixelLayout::L8,
        rgba_to_layout: rgba_to_l8,
        layout_to_reference_rgba: l8_to_rgba,
    });
}

#[test]
fn lossy_matrix_la8() {
    run_lossy_matrix(&LossyMatrixCase {
        name: "La8",
        layout: PixelLayout::La8,
        rgba_to_layout: rgba_to_la8,
        layout_to_reference_rgba: la8_to_rgba,
    });
}

// ---------------------------------------------------------------------------
// Lossless matrix tests — one per format. Byte-exact pin.
// ---------------------------------------------------------------------------

#[test]
fn lossless_matrix_rgba8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "Rgba8",
        layout: PixelLayout::Rgba8,
        rgba_to_layout: to_rgba_id,
        layout_to_reference_rgba: to_rgba_id,
    });
}

#[test]
fn lossless_matrix_bgra8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "Bgra8",
        layout: PixelLayout::Bgra8,
        rgba_to_layout: rgba_to_bgra,
        layout_to_reference_rgba: bgra_to_rgba,
    });
}

#[test]
fn lossless_matrix_argb8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "Argb8",
        layout: PixelLayout::Argb8,
        rgba_to_layout: rgba_to_argb,
        layout_to_reference_rgba: argb_to_rgba,
    });
}

#[test]
fn lossless_matrix_rgb8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "Rgb8",
        layout: PixelLayout::Rgb8,
        rgba_to_layout: rgba_to_rgb,
        layout_to_reference_rgba: rgb_to_opaque_rgba,
    });
}

#[test]
fn lossless_matrix_bgr8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "Bgr8",
        layout: PixelLayout::Bgr8,
        rgba_to_layout: rgba_to_bgr,
        layout_to_reference_rgba: bgr_to_opaque_rgba,
    });
}

#[test]
fn lossless_matrix_l8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "L8",
        layout: PixelLayout::L8,
        rgba_to_layout: rgba_to_l8,
        layout_to_reference_rgba: l8_to_rgba,
    });
}

#[test]
fn lossless_matrix_la8() {
    run_lossless_matrix(&LossyMatrixCase {
        name: "La8",
        layout: PixelLayout::La8,
        rgba_to_layout: rgba_to_la8,
        layout_to_reference_rgba: la8_to_rgba,
    });
}
