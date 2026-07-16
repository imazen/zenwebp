//! Byte-exactness gate for `CostModel::StrictLibwebpParity` (#38).
//!
//! This is the test behind any claim that zenwebp has a bit-exact libwebp
//! encode mode. Until it existed, that claim rested on an ad-hoc scratchpad
//! tool (`methodcmp`) which lived in `/tmp` and was destroyed — i.e. the
//! headline property had no gate in the repo at all, and a regression would
//! have shipped silently.
//!
//! ## What is and is not claimed
//!
//! As of 2026-07-16 the full committed grid — 13 images (3 CID22 512² photos
//! plus 10 synthetics incl. 1×1/2×2/3×3/odd-chroma/edge-partial MBs) × q ∈
//! {5..95} × 4 configs {(sns,flt,segs) = (0,0,1),(50,60,4),(0,0,4),(30,20,2)}
//! × m0-m6 = 4004 cells — is **4004/4004 byte-identical** (measured by
//! `dev/byteparity_sweep.rs`, which is the score; this test is the gate).
//! The honest claim is "byte-exact across that grid": settings outside it
//! (filter_sharpness ≠ 0, partitions > 1, alpha, target_size, other content)
//! have not been swept — widen the grid before widening the claim.
//!
//! Provenance: `benchmarks/byteparity_scope_2026-07-14.md`.
//!
//! ## Why compare against webpx
//!
//! `webpx` links the real libwebp C library, so this compares our encoder to
//! the reference implementation itself rather than to a Rust re-derivation of
//! it. A byte difference here means we diverged from libwebp, full stop.

#![cfg(all(feature = "std", feature = "__expert", not(target_arch = "wasm32")))]

use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

/// Deterministic gradient+noise source. Self-contained on purpose: a gate that
/// silently skips when a corpus file is missing is not a gate.
fn synth(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 24) as u8 / 8;
            px.extend_from_slice(&[
                ((x * 255 / w.max(1)) as u8).wrapping_add(n),
                ((y * 255 / h.max(1)) as u8).wrapping_add(n),
                (((x + y) * 255 / (w + h).max(1)) as u8).wrapping_add(n),
            ]);
        }
    }
    px
}

/// One cell of the grid: the encoder settings both sides are given.
#[derive(Clone, Copy)]
struct Cell {
    q: u8,
    m: u8,
    sns: u8,
    flt: u8,
    segs: u8,
    sharp: u8,
    plim: Option<u8>,
}

impl Cell {
    /// The cleanest config: no segmentation, no filter, no SNS.
    const fn plain(q: u8, m: u8) -> Self {
        Self {
            q,
            m,
            sns: 0,
            flt: 0,
            segs: 1,
            sharp: 0,
            plim: None,
        }
    }
}

/// Encode the same source through zenwebp (parity mode) and real libwebp at
/// matched settings and assert the bitstreams are byte-identical.
fn assert_byte_identical(name: &str, rgb: &[u8], w: u32, h: u32, c: Cell) {
    let Cell {
        q,
        m,
        sns,
        flt,
        segs,
        sharp,
        plim,
    } = c;
    let cfg = LossyConfig::new()
        .with_quality(f32::from(q))
        .with_method(m)
        .with_segments(segs)
        .with_sns_strength(sns)
        .with_filter_strength(flt)
        .with_filter_sharpness(sharp)
        .with_cost_model(CostModel::StrictLibwebpParity);
    let cfg = if let Some(p) = plim {
        cfg.with_partition_limit(p)
    } else {
        cfg
    };
    let zen = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("{name}: zenwebp encode failed: {e:?}"));

    let lib = webpx::EncoderConfig::new()
        .quality(f32::from(q))
        .method(m)
        .segments(segs)
        .sns_strength(sns)
        .filter_strength(flt)
        .filter_sharpness(sharp)
        .partition_limit(plim.unwrap_or(0))
        .encode_rgb(rgb, w, h, webpx::Unstoppable)
        .unwrap_or_else(|e| panic!("{name}: libwebp encode failed: {e:?}"));

    if zen != lib {
        let common = zen.iter().zip(&lib).take_while(|(a, b)| a == b).count();
        panic!(
            "{name} q{q} m{m} sns{sns} flt{flt} segs{segs}: NOT byte-identical to libwebp \
             — zen={} bytes, lib={} bytes, first difference at offset {common}. \
             (StrictLibwebpParity regression, #38. A difference at offset 4 alone means \
             only the RIFF size field diverged so far, i.e. the payloads differ in content.)",
            zen.len(),
            lib.len()
        );
    }
}

/// The pinned operating point: q75, every method m0-m6, at the two configs that
/// are byte-exact today. This is the 14-cell grid the bit-exactness claim rests
/// on; it was previously only checked by a throwaway tool.
#[test]
fn q75_all_methods_byte_identical() {
    let (w, h) = (512u32, 512u32);
    let rgb = synth(w, h, 41);
    for (sns, flt, segs) in [(0u8, 0u8, 1u8), (50, 60, 4)] {
        for m in 0u8..=6 {
            let c = Cell {
                q: 75,
                m,
                sns,
                flt,
                segs,
                sharp: 0,
                plim: None,
            };
            assert_byte_identical(&format!("synth_{w}x{h}"), &rgb, w, h, c);
        }
    }
}

/// Tiny and odd-dimension sources exercise partial-MB edges and the fixed
/// header overhead, where parity bugs cluster. Kept at the sns0/segs1 config,
/// which is the cleanest (no segmentation, no filter).
#[test]
fn tiny_and_odd_dimensions_byte_identical() {
    for (w, h, seed) in [
        (1u32, 1u32, 7u32),
        (2, 2, 11),
        (3, 3, 13),
        (16, 16, 17),
        (17, 17, 19),
        (33, 17, 23),
    ] {
        let rgb = synth(w, h, seed);
        for m in 0u8..=6 {
            assert_byte_identical(&format!("synth_{w}x{h}"), &rgb, w, h, Cell::plain(75, m));
        }
    }
}

/// The Cat5/Cat6 stat-node accounting (#38) is method-dependent: libwebp routes
/// m3-m6 through `VP8RecordCoeffTokens` (statistic -> node 9) and m0-m2 through
/// `VP8RecordCoeffs` (node 10). Both sides of that split must hold, so pin a
/// quality where large (category-coded) levels actually occur — at high q there
/// are more and larger coefficients, which is exactly where the misfiling showed.
#[test]
fn high_quality_spans_both_recorder_paths() {
    let (w, h) = (256u32, 255u32);
    let rgb = synth(w, h, 41);
    // m0-m2 take libwebp's non-token path; m3-m6 take the token path.
    for m in 0u8..=6 {
        assert_byte_identical(&format!("synth_{w}x{h}"), &rgb, w, h, Cell::plain(90, m));
    }
}

/// Regression anchors for the four roots that completed the 4004-cell grid
/// (2026-07-16, #38): the m0-m2 skip-proba StatLoop count (fires at low q
/// with SNS + multi-segment configs), the m5/m6 skip-from-trellis-levels
/// decision (mid q), the segment-quant libm-pow truncation boundary, and the
/// I4 tie-break in libwebp's enum order (high q, where exact RD ties
/// concentrate). Sweeps the two SNS configs the q75 pin never covered at
/// low/mid/high q, plus the exact odd-dimension cell that exposed the pow
/// boundary.
#[test]
fn sns_configs_low_mid_high_q_byte_identical() {
    let (w, h) = (512u32, 512u32);
    let rgb = synth(w, h, 41);
    for (sns, flt, segs) in [(50u8, 60u8, 4u8), (30, 20, 2)] {
        for q in [5u8, 20, 80] {
            for m in 0u8..=6 {
                let c = Cell {
                    q,
                    m,
                    sns,
                    flt,
                    segs,
                    sharp: 0,
                    plim: None,
                };
                assert_byte_identical(&format!("synth_{w}x{h}"), &rgb, w, h, c);
            }
        }
    }
    // The segment-quant pow-boundary cell: synth 33x17 q90 sns50/flt60/segs4
    // (seg1 quant index flipped 12 vs 11 under the fast-pow approximation).
    let rgb = synth(33, 17, 23);
    for m in 0u8..=6 {
        let c = Cell {
            q: 90,
            m,
            sns: 50,
            flt: 60,
            segs: 4,
            sharp: 0,
            plim: None,
        };
        assert_byte_identical("synth_33x17", &rgb, 33, 17, c);
    }
}

/// Regression anchors for the setting-permutation roots (2026-07-16, #38):
/// filter_sharpness (libwebp derives strengths with the PREVIOUS call's
/// sharpness — the read-before-assign in `SetupFilterStrength`), quality
/// edges (UV-DC index clip at 117; error diffusion off above q98; the
/// flat-latch-doubled D in the StoreMaxDelta gate), pinned partition_limit
/// (libwebp's limit feeds only `max_i4_header_bits`), and single-segment
/// SNS (uv_alpha-derived UV quant deltas apply at every segment count).
/// Full sweep: `dev/byteparity_sweep.rs` phase 2.
#[test]
fn permutation_axis_anchors_byte_identical() {
    let (w, h) = (512u32, 512u32);
    let rgb = synth(w, h, 41);
    let cells = [
        // sharpness axis (sh in Cell::sharp)
        (5u8, 5u8, 50u8, 60u8, 4u8, 3u8, None),
        (75, 2, 30, 20, 2, 7, None),
        // quality edges
        (0, 5, 0, 0, 1, 0, None),
        (0, 4, 50, 60, 4, 0, None),
        (99, 4, 50, 60, 4, 0, None),
        (100, 6, 0, 0, 1, 0, None),
        (1, 5, 50, 60, 4, 0, None),
        // pinned partition_limit
        (5, 3, 50, 60, 4, 0, Some(30u8)),
        (50, 1, 50, 60, 4, 0, Some(100)),
        // single-segment SNS (uv quant deltas at segs1)
        (50, 4, 80, 60, 1, 0, None),
    ];
    for (q, m, sns, flt, segs, sharp, plim) in cells {
        let c = Cell {
            q,
            m,
            sns,
            flt,
            segs,
            sharp,
            plim,
        };
        assert_byte_identical(&format!("synth_{w}x{h}"), &rgb, w, h, c);
    }
}

/// Opaque RGBA input must produce a bare `VP8 ` file byte-identical to
/// libwebp's — no VP8X/ALPH wrapper (libwebp scans the pixels via
/// `WebPPictureHasTransparency`; zenwebp now does the same). (#38)
#[test]
fn opaque_rgba_matches_libwebp_bare_vp8() {
    let (w, h) = (64u32, 64u32);
    let rgb = synth(w, h, 31);
    let rgba: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    for &(q, m) in &[(5u8, 0u8), (50, 2), (75, 4), (90, 6)] {
        let cfg = LossyConfig::new()
            .with_quality(f32::from(q))
            .with_method(m)
            .with_segments(4)
            .with_sns_strength(50)
            .with_filter_strength(60)
            .with_filter_sharpness(0)
            .with_cost_model(CostModel::StrictLibwebpParity);
        let zen = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h)
            .encode()
            .unwrap();
        let lib = webpx::EncoderConfig::new()
            .quality(f32::from(q))
            .method(m)
            .segments(4)
            .sns_strength(50)
            .filter_strength(60)
            .filter_sharpness(0)
            .encode_rgba(&rgba, w, h, webpx::Unstoppable)
            .unwrap();
        assert_eq!(
            zen,
            lib,
            "opaque RGBA q{q} m{m}: zen={} lib={} — expected bare VP8, byte-identical",
            zen.len(),
            lib.len()
        );
    }
}
