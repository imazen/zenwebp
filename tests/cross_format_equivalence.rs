//! Cross-format input equivalence — the gap that hid the L8 lossy bug.
//!
//! Two `PixelDescriptor` / `PixelLayout` choices that describe the **same
//! image semantically** (e.g. `L8(g)` and `Rgba8(g, g, g, 255)`, or
//! `Rgba8(r, g, b, a)` and `Bgra8(b, g, r, a)`) must produce **decoded
//! outputs that match**:
//!
//! - **Lossless** path: byte-identical decoded RGBA.
//! - **Lossy** path: perceptually close (zensim ≥ 95) — small differences
//!   from chroma subsampling are normal but the two formats must not pick
//!   visibly different reconstructions of the same content.
//!
//! Coverage:
//!
//! | Pair | What it pins |
//! |---|---|
//! | `Rgba8` ↔ `Bgra8` ↔ `Argb8` | Per-channel order doesn't change encoding |
//! | `Rgb8` ↔ `Bgr8` | Same, alpha-less |
//! | `Rgba8` ↔ `Rgb8` (alpha=255) | Alpha plane handling for opaque input |
//! | `L8` ↔ `Rgba8(g, g, g, 255)` | sRGB-gray L8 path matches RGB-replicated gray |
//! | `La8` ↔ `Rgba8(g, g, g, a)` | sRGB-gray + alpha L8 path matches RGB-replicated |
//!
//! For each pair we sweep methods {0, 4, 6} × qualities {25, 50, 75, 90}
//! lossy plus lossless. Failures print every (config, generator) cell so
//! you can see exactly which combination diverged.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::generators;
use zenwebp::{EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

const W: u32 = 96;
const H: u32 = 96;

// ---------------------------------------------------------------------------
// Image generators (all return RGBA; gray-only generator returns gray-as-RGBA)
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

/// Diagonal gray gradient + checker noise stored as RGBA(g, g, g, 255).
fn gen_gray_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let g = ((x + y) * 255 / (w + h).max(1)) as u8;
            let n = if (x ^ y) & 1 == 0 { 0 } else { 8 };
            let g = g.saturating_add(n);
            let i = ((y * w + x) * 4) as usize;
            out[i] = g;
            out[i + 1] = g;
            out[i + 2] = g;
            out[i + 3] = 255;
        }
    }
    out
}

/// Same as `gen_gray_rgba` but with a varying alpha plane (for La8 ↔ RGBA).
fn gen_gray_alpha_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let g = ((x + y) * 255 / (w + h).max(1)) as u8;
            let a = if x > w / 2 {
                255
            } else {
                (x * 255 / (w / 2).max(1)) as u8
            };
            let i = ((y * w + x) * 4) as usize;
            out[i] = g;
            out[i + 1] = g;
            out[i + 2] = g;
            out[i + 3] = a;
        }
    }
    out
}

const COLOR_GENS: &[(&str, ImgGen)] = &[
    ("gradient", gen_gradient),
    ("noise", gen_noise),
    ("color_blocks", gen_color_blocks),
    ("mandelbrot", gen_mandelbrot),
    ("checker", gen_checker),
];

const GRAY_GENS: &[(&str, ImgGen)] = &[("gray_grad", gen_gray_rgba)];
const GRAY_ALPHA_GENS: &[(&str, ImgGen)] = &[("gray_alpha", gen_gray_alpha_rgba)];

// ---------------------------------------------------------------------------
// Format conversions (input = Rgba8 reference, output = alt layout's bytes)
// ---------------------------------------------------------------------------

fn rgba_to_bgra(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len());
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    out
}

fn rgba_to_argb(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len());
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&[px[3], px[0], px[1], px[2]]);
    }
    out
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

fn rgba_to_bgr(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
    }
    out
}

fn rgba_to_l8(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4);
    for px in rgba.chunks_exact(4) {
        debug_assert!(
            px[0] == px[1] && px[1] == px[2],
            "rgba_to_l8 requires R=G=B input"
        );
        out.push(px[0]);
    }
    out
}

fn rgba_to_la8(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 2);
    for px in rgba.chunks_exact(4) {
        debug_assert!(
            px[0] == px[1] && px[1] == px[2],
            "rgba_to_la8 requires R=G=B input"
        );
        out.push(px[0]);
        out.push(px[3]);
    }
    out
}

/// Strip alpha (force opaque) — used for opaque-only pair tests.
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

fn enc_lossless(pixels: &[u8], layout: PixelLayout, w: u32, h: u32) -> Vec<u8> {
    // exact=true so RGB under α=0 is preserved — required for the
    // pair-equivalence assertion to be meaningful when the input has any
    // transparent pixels.
    let cfg = LosslessConfig::new().with_exact(true);
    EncodeRequest::lossless(&cfg, pixels, layout, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("lossless {layout} {w}x{h}: {e}"))
}

fn enc_lossy(pixels: &[u8], layout: PixelLayout, w: u32, h: u32, method: u8, q: f32) -> Vec<u8> {
    let cfg = LossyConfig::new().with_quality(q).with_method(method);
    EncodeRequest::lossy(&cfg, pixels, layout, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("lossy m{method} q{q} {layout} {w}x{h}: {e}"))
}

fn dec_rgba(webp: &[u8]) -> (Vec<u8>, u32, u32) {
    zenwebp::oneshot::decode_rgba(webp).expect("decode_rgba")
}

/// View `&[u8]` as `&[[u8; 4]]` for zensim. Length must be a multiple of 4.
fn as_rgba_pixels(data: &[u8]) -> &[[u8; 4]] {
    assert!(data.len().is_multiple_of(4));
    // SAFETY: a `[u8]` whose length is a multiple of 4 is a valid `[[u8; 4]]`.
    let (prefix, pixels, suffix) = unsafe { data.align_to::<[u8; 4]>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    pixels
}

fn zensim_score(a: &[u8], b: &[u8], w: u32, h: u32) -> f64 {
    let z = Zensim::new(ZensimProfile::latest());
    let a = RgbaSlice::new(as_rgba_pixels(a), w as usize, h as usize);
    let b = RgbaSlice::new(as_rgba_pixels(b), w as usize, h as usize);
    z.compute(&a, &b).expect("zensim compute").score()
}

// Compare two decoded RGBA buffers; return (count_diff, max_per_channel).
fn rgba_diff(a: &[u8], b: &[u8]) -> (u32, [u16; 4]) {
    assert_eq!(a.len(), b.len());
    let mut count = 0u32;
    let mut max = [0u16; 4];
    for (ap, bp) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if ap != bp {
            count += 1;
        }
        for c in 0..4 {
            max[c] = max[c].max((ap[c] as i16 - bp[c] as i16).unsigned_abs());
        }
    }
    (count, max)
}

// ---------------------------------------------------------------------------
// Configuration matrix
// ---------------------------------------------------------------------------

const METHODS: &[u8] = &[0, 4, 6];
const QUALITIES: &[f32] = &[25.0, 50.0, 75.0, 90.0];

/// Lower bound on the cross-format zensim score for **same-precision** pairs
/// (RGBA/BGRA/ARGB/RGB/BGR). These pairs hit the same YUV pipeline (zenyuv)
/// after a constant-time channel reorder, so divergence should be tiny —
/// 95 leaves headroom only for chroma-subsampling rounding that can land
/// differently across SIMD codepaths.
const LOSSY_PAIR_MIN_SCORE_TIGHT: f64 = 95.0;

/// Lower bound for L8/La8 lossy pairs. Matches `_TIGHT` because
/// `convert_image_y` now routes through the same zenyuv kernel the RGB
/// path uses (gray-replicated to RGB transient → zenyuv → take Y). The
/// two paths produce a bit-identical Y plane and the chroma fill is the
/// same constant 128, so the encoder makes identical decisions and the
/// decoded outputs match within chroma-subsampling rounding.
const LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY: f64 = LOSSY_PAIR_MIN_SCORE_TIGHT;

// ---------------------------------------------------------------------------
// Pair runners
// ---------------------------------------------------------------------------

/// Convert reference RGBA to alt layout.
type Conv = fn(&[u8]) -> Vec<u8>;

struct PairCase<'a> {
    name: &'a str,
    gens: &'a [(&'a str, ImgGen)],
    /// `(reference layout, conv from RGBA)`. RGBA → reference bytes.
    /// (Identity is the common case but some tests use a non-RGBA reference.)
    reference: (PixelLayout, Conv),
    /// `(alt layout, conv from RGBA)`. RGBA → alt bytes.
    alt: (PixelLayout, Conv),
    /// If true, only run on opaque inputs (alpha=255 everywhere).
    opaque_only: bool,
    /// Lossy zensim score floor for this pair (lossless ignores it).
    lossy_min_score: f64,
}

fn id_rgba(rgba: &[u8]) -> Vec<u8> {
    rgba.to_vec()
}

fn run_pair_lossless(case: &PairCase) {
    let mut failures = Vec::new();
    for &(gname, genfn) in case.gens {
        let mut rgba = genfn(W, H);
        if case.opaque_only {
            rgba = force_opaque(&rgba);
        }
        let ref_pixels = (case.reference.1)(&rgba);
        let alt_pixels = (case.alt.1)(&rgba);

        let ref_webp = enc_lossless(&ref_pixels, case.reference.0, W, H);
        let alt_webp = enc_lossless(&alt_pixels, case.alt.0, W, H);
        let (ref_dec, _, _) = dec_rgba(&ref_webp);
        let (alt_dec, _, _) = dec_rgba(&alt_webp);

        let (n, max) = rgba_diff(&ref_dec, &alt_dec);
        if n != 0 {
            failures.push(format!(
                "  [{gname}] LOSSLESS {} vs {}: {} pixels differ, max delta R={} G={} B={} A={} \
                 (ref={} bytes, alt={} bytes)",
                case.reference.0,
                case.alt.0,
                n,
                max[0],
                max[1],
                max[2],
                max[3],
                ref_webp.len(),
                alt_webp.len(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\n=== Lossless cross-format divergence: {} ===\n{}\n",
        case.name,
        failures.join("\n")
    );
}

fn run_pair_lossy(case: &PairCase) {
    let mut failures = Vec::new();
    let mut report = Vec::new();
    for &(gname, genfn) in case.gens {
        let mut rgba = genfn(W, H);
        if case.opaque_only {
            rgba = force_opaque(&rgba);
        }
        let ref_pixels = (case.reference.1)(&rgba);
        let alt_pixels = (case.alt.1)(&rgba);

        for &m in METHODS {
            for &q in QUALITIES {
                let ref_webp = enc_lossy(&ref_pixels, case.reference.0, W, H, m, q);
                let alt_webp = enc_lossy(&alt_pixels, case.alt.0, W, H, m, q);
                let (ref_dec, _, _) = dec_rgba(&ref_webp);
                let (alt_dec, _, _) = dec_rgba(&alt_webp);
                let score = zensim_score(&ref_dec, &alt_dec, W, H);
                let line = format!(
                    "  [{gname}] m{m} q{q:>4.0}: {:>5} vs {:>5} bytes, score={:.2}",
                    ref_webp.len(),
                    alt_webp.len(),
                    score
                );
                report.push(line.clone());
                if score < case.lossy_min_score {
                    failures.push(format!("{line}  ← below {}", case.lossy_min_score));
                }
            }
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "\n=== Lossy cross-format report: {} ===\n{}\n",
            case.name,
            report.join("\n")
        );
        panic!(
            "Lossy cross-format divergence: {}\n{}\n",
            case.name,
            failures.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — color-channel-order pairs (full alpha)
// ---------------------------------------------------------------------------

#[test]
fn lossless_rgba_vs_bgra() {
    run_pair_lossless(&PairCase {
        name: "Rgba8 ↔ Bgra8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Bgra8, rgba_to_bgra),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

#[test]
fn lossless_rgba_vs_argb() {
    run_pair_lossless(&PairCase {
        name: "Rgba8 ↔ Argb8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Argb8, rgba_to_argb),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

#[test]
fn lossy_rgba_vs_bgra() {
    run_pair_lossy(&PairCase {
        name: "Rgba8 ↔ Bgra8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Bgra8, rgba_to_bgra),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

#[test]
fn lossy_rgba_vs_argb() {
    run_pair_lossy(&PairCase {
        name: "Rgba8 ↔ Argb8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Argb8, rgba_to_argb),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

// ---------------------------------------------------------------------------
// Tests — alpha-less pairs (RGB ↔ BGR)
// ---------------------------------------------------------------------------

#[test]
fn lossless_rgb_vs_bgr() {
    run_pair_lossless(&PairCase {
        name: "Rgb8 ↔ Bgr8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgb8, rgba_to_rgb),
        alt: (PixelLayout::Bgr8, rgba_to_bgr),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

#[test]
fn lossy_rgb_vs_bgr() {
    run_pair_lossy(&PairCase {
        name: "Rgb8 ↔ Bgr8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgb8, rgba_to_rgb),
        alt: (PixelLayout::Bgr8, rgba_to_bgr),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

// ---------------------------------------------------------------------------
// Tests — alpha vs opaque (RGBA(opaque) ↔ RGB)
// ---------------------------------------------------------------------------

#[test]
fn lossless_rgba_opaque_vs_rgb() {
    run_pair_lossless(&PairCase {
        name: "Rgba8(α=255) ↔ Rgb8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Rgb8, rgba_to_rgb),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

#[test]
fn lossy_rgba_opaque_vs_rgb() {
    run_pair_lossy(&PairCase {
        name: "Rgba8(α=255) ↔ Rgb8",
        gens: COLOR_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::Rgb8, rgba_to_rgb),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_TIGHT,
    });
}

// ---------------------------------------------------------------------------
// Tests — gray pairs (the original bug-class)
// ---------------------------------------------------------------------------

#[test]
fn lossless_l8_vs_rgba_gray() {
    run_pair_lossless(&PairCase {
        name: "L8(g) ↔ Rgba8(g, g, g, 255)",
        gens: GRAY_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::L8, rgba_to_l8),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}

#[test]
fn lossy_l8_vs_rgba_gray() {
    run_pair_lossy(&PairCase {
        name: "L8(g) ↔ Rgba8(g, g, g, 255)",
        gens: GRAY_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::L8, rgba_to_l8),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}

#[test]
fn lossless_l8_vs_rgb_gray() {
    run_pair_lossless(&PairCase {
        name: "L8(g) ↔ Rgb8(g, g, g)",
        gens: GRAY_GENS,
        reference: (PixelLayout::Rgb8, rgba_to_rgb),
        alt: (PixelLayout::L8, rgba_to_l8),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}

#[test]
fn lossy_l8_vs_rgb_gray() {
    run_pair_lossy(&PairCase {
        name: "L8(g) ↔ Rgb8(g, g, g)",
        gens: GRAY_GENS,
        reference: (PixelLayout::Rgb8, rgba_to_rgb),
        alt: (PixelLayout::L8, rgba_to_l8),
        opaque_only: true,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}

#[test]
fn lossless_la8_vs_rgba_gray_alpha() {
    run_pair_lossless(&PairCase {
        name: "La8(g, a) ↔ Rgba8(g, g, g, a)",
        gens: GRAY_ALPHA_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::La8, rgba_to_la8),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}

#[test]
fn lossy_la8_vs_rgba_gray_alpha() {
    run_pair_lossy(&PairCase {
        name: "La8(g, a) ↔ Rgba8(g, g, g, a)",
        gens: GRAY_ALPHA_GENS,
        reference: (PixelLayout::Rgba8, id_rgba),
        alt: (PixelLayout::La8, rgba_to_la8),
        opaque_only: false,
        lossy_min_score: LOSSY_PAIR_MIN_SCORE_GRAY_LOSSY,
    });
}
