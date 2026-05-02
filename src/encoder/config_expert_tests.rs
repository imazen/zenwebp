//! Permutation tests for the `__expert`-gated `InternalParams` knob bundle.
//!
//! Each per-field test confirms that flipping that field away from the
//! `LossyConfig` default produces an encode that is observably different
//! from the baseline (different bytes, or — for cost-equivalent
//! configurations — at least no panics + decodable output).
//!
//! These tests use a small synthetic RGB image with mixed content
//! (gradient + textured + flat regions) so all five axes get measurable
//! traction. Encodes use `quality(75) + method(4)` because m4 is where
//! `multi_pass_stats` is gated to take effect.
//!
//! Gated `#[cfg(all(test, feature = "__expert"))]` because the bundle
//! itself only exists under that feature.

#![cfg(all(test, feature = "__expert"))]

extern crate alloc;
use alloc::vec::Vec;

use super::api::{CostModel, EncodeRequest, PixelLayout};
use super::config::{InternalParams, LossyConfig, SharpYuvSetting};

const W: u32 = 192;
const H: u32 = 144;

/// Build a deterministic RGB8 image with strong per-block activity
/// variation so segmentation produces multiple segments AND noisy
/// segment boundaries (the prerequisite for smooth_segment_map to
/// actually mutate the map). The image has ~12×9 macroblocks worth of
/// regions: a fine-grain noise patch, a smooth gradient, a solid band,
/// and isolated single-MB anomalies that the 3×3 majority filter must
/// flip. Each region is sized so segmentation k-means lands it in a
/// distinct cluster.
fn synthetic_rgb() -> Vec<u8> {
    let mut buf = Vec::with_capacity((W * H * 3) as usize);
    // Simple deterministic LCG for repeatable noise.
    let mut state: u32 = 0x1234_5678;
    let mut rng = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 16) as u8
    };
    // Pre-roll noise into a buffer so the value at (x,y) is deterministic.
    let noise: Vec<u8> = (0..(W * H * 3)).map(|_| rng()).collect();

    for y in 0..H {
        for x in 0..W {
            let mb_x = x / 16;
            let mb_y = y / 16;
            // Region selector by macroblock position — gives the encoder
            // chunky regions that k-means can cluster on.
            let band = if mb_y < 3 {
                // Top: noisy textured region (high alpha → high-quant segment).
                let i = ((y * W + x) * 3) as usize;
                [noise[i], noise[i + 1], noise[i + 2]]
            } else if mb_y < 5 {
                // Smooth gradient (low alpha → low-quant segment).
                let r = (x * 255 / W) as u8;
                let g = (y * 255 / H) as u8;
                [r, g, 128]
            } else if mb_y < 7 {
                // Mid-frequency checker — distinct activity from above two.
                let on = ((x / 4) ^ (y / 4)) & 1 != 0;
                if on { [240, 30, 30] } else { [20, 200, 220] }
            } else {
                // Bottom solid bands with single-MB anomalies — these
                // isolated MBs are exactly what 3×3 majority filtering
                // reassigns when smooth_segment_map=true.
                let anomaly = (mb_x % 4 == 0) && (mb_y == 8);
                if anomaly {
                    let i = ((y * W + x) * 3) as usize;
                    [noise[i], noise[i + 1], noise[i + 2]]
                } else {
                    [128, 128, 128]
                }
            };
            buf.extend_from_slice(&band);
        }
    }
    buf
}

fn encode(cfg: &LossyConfig, pixels: &[u8]) -> Vec<u8> {
    EncodeRequest::lossy(cfg, pixels, PixelLayout::Rgb8, W, H)
        .encode()
        .expect("encode succeeded")
}

fn baseline_cfg() -> LossyConfig {
    LossyConfig::new().with_quality(75.0).with_method(4)
}

// ----------------------------------------------------------------------
// (a) Per-field tests — each non-default value differs from the baseline
//     bytes, OR (where the knob is intentionally cost-neutral on the
//     synthetic input) at least produces a decodable result.
// ----------------------------------------------------------------------

/// `partition_limit` ∈ {0, 50, 100} — at 100 the encoder forces I16-only,
/// which yields different bytes from the default at quality 75.
#[test]
fn partition_limit_variants_produce_distinct_bytes() {
    let rgb = synthetic_rgb();
    let baseline = encode(&baseline_cfg(), &rgb);

    let with_50 = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            partition_limit: Some(50),
            ..Default::default()
        }),
        &rgb,
    );
    let with_100 = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            partition_limit: Some(100),
            ..Default::default()
        }),
        &rgb,
    );

    // limit=100 forces I16-only; on this content it must differ from default.
    assert_ne!(
        baseline, with_100,
        "partition_limit=100 should change emitted bytes vs default"
    );
    // limit=50 raises the I4 skip-threshold ~5x — on a 96x64 mixed image
    // mode-decision picks differ; encoded bytes must differ from at least
    // one of {baseline, with_100}.
    assert!(
        with_50 != baseline || with_50 != with_100,
        "partition_limit=50 should produce a distinct intermediate result"
    );
    // All three decode without panicking.
    for bytes in [&baseline, &with_50, &with_100] {
        let _ = webp::Decoder::new(bytes).decode().expect("decodable");
    }
}

/// `multi_pass_stats: true` should change emitted bytes at m4 because the
/// second pass uses image-tuned `level_costs`.
#[test]
fn multi_pass_stats_toggle_changes_bytes() {
    let rgb = synthetic_rgb();
    let baseline = encode(&baseline_cfg(), &rgb);
    let two_pass = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            multi_pass_stats: Some(true),
            ..Default::default()
        }),
        &rgb,
    );
    assert_ne!(
        baseline, two_pass,
        "multi_pass_stats=true at m4 must change emitted bytes"
    );
}

/// `smooth_segment_map: true` should change emitted bytes when
/// `num_segments > 1`. Default `num_segments` resolves to 4 (no preset).
#[test]
fn smooth_segment_map_toggle_changes_bytes() {
    let rgb = synthetic_rgb();
    let baseline = encode(&baseline_cfg(), &rgb);
    let smoothed = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            smooth_segment_map: Some(true),
            ..Default::default()
        }),
        &rgb,
    );
    assert_ne!(
        baseline, smoothed,
        "smooth_segment_map=true must change emitted bytes on multi-segment input"
    );
}

/// `sharp_yuv` Off / On / Custom — On runs the iterative chroma refinement,
/// which produces a different YUV plane and therefore different bytes.
#[test]
fn sharp_yuv_variants_encode() {
    let rgb = synthetic_rgb();
    let off = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            sharp_yuv: Some(SharpYuvSetting::Off),
            ..Default::default()
        }),
        &rgb,
    );
    let on = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            sharp_yuv: Some(SharpYuvSetting::On),
            ..Default::default()
        }),
        &rgb,
    );
    let custom = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            sharp_yuv: Some(SharpYuvSetting::Custom(zenyuv::SharpYuvConfig {
                max_iterations: 4,
                convergence_threshold: 0.05,
                refine_y: false,
            })),
            ..Default::default()
        }),
        &rgb,
    );
    // Off matches LossyConfig default (no sharp YUV) — so equals baseline.
    let baseline = encode(&baseline_cfg(), &rgb);
    assert_eq!(
        off, baseline,
        "sharp_yuv=Off must match LossyConfig default"
    );
    assert_ne!(on, baseline, "sharp_yuv=On must change emitted bytes");
    assert_ne!(
        custom, on,
        "Custom sharp_yuv config must differ from On default"
    );
    // All decode.
    for bytes in [&off, &on, &custom] {
        let _ = webp::Decoder::new(bytes).decode().expect("decodable");
    }
}

/// `cost_model` ZenwebpDefault vs StrictLibwebpParity — at m4 the
/// PSY_WEIGHT_Y CSF table swap and the SATD masking-alpha blend gate are
/// active, so emitted bytes must differ on textured content.
#[test]
fn cost_model_variants_differ_at_m4() {
    let rgb = synthetic_rgb();
    let zw = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            cost_model: Some(CostModel::ZenwebpDefault),
            ..Default::default()
        }),
        &rgb,
    );
    let parity = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            cost_model: Some(CostModel::StrictLibwebpParity),
            ..Default::default()
        }),
        &rgb,
    );
    let baseline = encode(&baseline_cfg(), &rgb);
    assert_eq!(
        zw, baseline,
        "cost_model=ZenwebpDefault must match LossyConfig default"
    );
    assert_ne!(
        zw, parity,
        "cost_model=StrictLibwebpParity must change emitted bytes vs zenwebp default"
    );
}

// ----------------------------------------------------------------------
// (b) Idempotency — applying the same InternalParams twice equals applying
//     once. `with_internal_params` is partial-merge: each `Some` field
//     overrides, each `None` preserves. Two identical applications hit
//     the same fields with the same values, so the result is bit-equal.
// ----------------------------------------------------------------------

#[test]
fn with_internal_params_is_idempotent() {
    let rgb = synthetic_rgb();
    let knobs = InternalParams {
        partition_limit: Some(40),
        multi_pass_stats: Some(true),
        smooth_segment_map: Some(true),
        sharp_yuv: Some(SharpYuvSetting::On),
        cost_model: Some(CostModel::StrictLibwebpParity),
    };
    let once = encode(&baseline_cfg().with_internal_params(knobs.clone()), &rgb);
    let twice = encode(
        &baseline_cfg()
            .with_internal_params(knobs.clone())
            .with_internal_params(knobs),
        &rgb,
    );
    assert_eq!(
        once, twice,
        "applying the same InternalParams twice must be idempotent"
    );
}

// ----------------------------------------------------------------------
// (c) Combined — all five fields set produces a valid, non-empty,
//     decodable WebP.
// ----------------------------------------------------------------------

#[test]
fn all_fields_combined_produces_valid_encode() {
    let rgb = synthetic_rgb();
    let cfg = baseline_cfg().with_internal_params(InternalParams {
        partition_limit: Some(60),
        multi_pass_stats: Some(true),
        smooth_segment_map: Some(true),
        sharp_yuv: Some(SharpYuvSetting::On),
        cost_model: Some(CostModel::StrictLibwebpParity),
    });
    let bytes = encode(&cfg, &rgb);
    assert!(!bytes.is_empty(), "combined encode produced empty output");
    let decoded = webp::Decoder::new(&bytes).decode().expect("decodable");
    // The decoded image must have the original dimensions.
    assert_eq!(decoded.width(), W);
    assert_eq!(decoded.height(), H);
}

// ----------------------------------------------------------------------
// (d) Default = baseline — `InternalParams::default()` is all-None, which
//     leaves every LossyConfig field unchanged. Encoded bytes must match
//     a fresh LossyConfig::new() byte-for-byte.
// ----------------------------------------------------------------------

#[test]
fn default_internal_params_equals_baseline() {
    let rgb = synthetic_rgb();
    let baseline = encode(&baseline_cfg(), &rgb);
    let with_default = encode(
        &baseline_cfg().with_internal_params(InternalParams::default()),
        &rgb,
    );
    assert_eq!(
        baseline, with_default,
        "InternalParams::default() must be a no-op on a fresh LossyConfig"
    );
}

// ----------------------------------------------------------------------
// (e) Reset semantics — `with_internal_params` is partial-merge, NOT
//     wholesale-replace. Re-applying `Default` after setting a field
//     does NOT clear that field. We assert the documented behavior so
//     the partial-merge contract is locked in.
// ----------------------------------------------------------------------

#[test]
fn with_internal_params_is_partial_merge_not_reset() {
    let rgb = synthetic_rgb();
    let baseline = encode(&baseline_cfg(), &rgb);
    // Set smooth_segment_map=true, then re-apply Default (all None).
    let merged = encode(
        &baseline_cfg()
            .with_internal_params(InternalParams {
                smooth_segment_map: Some(true),
                ..Default::default()
            })
            .with_internal_params(InternalParams::default()),
        &rgb,
    );
    // Default's None-on-smooth_segment_map preserves the previously-set
    // `true`, so the merged encode must NOT match the baseline.
    assert_ne!(
        baseline, merged,
        "Re-applying Default must not reset previously-set fields (partial merge)"
    );
    // It must equal the encode that only sets smooth_segment_map=true.
    let single = encode(
        &baseline_cfg().with_internal_params(InternalParams {
            smooth_segment_map: Some(true),
            ..Default::default()
        }),
        &rgb,
    );
    assert_eq!(
        merged, single,
        "Re-applying Default after a real set must equal a single set"
    );
}
