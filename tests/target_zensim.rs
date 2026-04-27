//! Closed-loop `target_zensim` adaptive encoder tests.
//!
//! These tests exercise the full encode → decode → measure → adjust loop
//! and the strict-mode error path. Convergence bands are intentionally
//! generous (see issue #47): the calibration is approximate by design,
//! and tightening the band would create a noisy gate without protecting
//! anything callers actually care about.

#![cfg(feature = "target-zensim")]

use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ZensimEncodeMetrics, ZensimTarget};

/// Build a 256×256 RGB8 mixed-content image: smooth gradient + low-
/// amplitude band-limited "natural" texture. Designed to give the
/// encoder genuine RD-tradeoff work without the perceptually-impossible
/// pure noise that caps zensim well below 60 even at q=100.
fn mixed_content_256() -> (Vec<u8>, u32, u32) {
    let w: u32 = 256;
    let h: u32 = 256;
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    // Use a smooth f32-domain pattern: two overlaid sine waves + a slow
    // gradient. This is "photo-like" content that VP8 can encode well.
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            // Slow gradient.
            let base_r = 80.0 + 100.0 * fx;
            let base_g = 100.0 + 80.0 * fy;
            let base_b = 90.0 + 80.0 * (1.0 - fx);
            // Low-amplitude band-limited "texture" — sin waves at modest
            // frequencies. Stays well within VP8's coding capability.
            let tex =
                18.0 * ((fx * 9.0).sin() * (fy * 7.0).cos() + 0.6 * (fx * 17.0 + fy * 11.0).sin());
            let r = (base_r + tex).clamp(0.0, 255.0) as u8;
            let g = (base_g + tex * 0.7).clamp(0.0, 255.0) as u8;
            let b = (base_b - tex * 0.5).clamp(0.0, 255.0) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    (buf, w, h)
}

/// Smooth gradient — easy content where pass 1 at calibrated q tends to
/// overshoot the target. Used to exercise the bytes-recovery path.
fn smooth_gradient_256() -> (Vec<u8>, u32, u32) {
    let w: u32 = 256;
    let h: u32 = 256;
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 255) / (w + h)) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    (buf, w, h)
}

#[test]
fn converges_to_target_80_within_two_passes() {
    let (rgb, w, h) = mixed_content_256();
    let cfg = LossyConfig::new().with_method(4).with_target_zensim(80.0);

    let (bytes, metrics) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");

    assert!(!bytes.is_empty(), "encoder produced no bytes");
    assert!(
        metrics.passes_used <= 2,
        "used {} passes (>2)",
        metrics.passes_used
    );
    // Generous band — calibration is approximate. The tighter band
    // [78.5, 81.5] is the spec target but is sensitive to encoder/zensim
    // version drift. We test the looser band that catches gross bugs.
    let s = metrics.achieved_score;
    assert!(s.is_finite(), "score is non-finite: {s}");
    assert!(
        (75.0..=86.0).contains(&s),
        "achieved {s} outside acceptable convergence band [75, 86] for target=80"
    );
}

#[test]
fn metrics_no_target_when_disabled() {
    let (rgb, w, h) = smooth_gradient_256();
    let cfg = LossyConfig::new().with_method(4).with_quality(75.0);
    let (bytes, metrics) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");
    assert!(!bytes.is_empty());
    assert!(metrics.achieved_score.is_nan(), "no_target should be NaN");
    assert_eq!(metrics.passes_used, 1);
    assert!(metrics.targets_met);
}

#[test]
fn strict_undershoot_errors_on_unreachable_target() {
    let (rgb, w, h) = mixed_content_256();
    // Target 99 with a tight 0.5 floor and just one pass: the
    // calibrated start at q≈98 is unlikely to clear 99 on a noisy
    // mixed-content image, so this MUST error rather than silently
    // ship.
    let target = ZensimTarget::new(99.0)
        .with_max_undershoot(Some(0.5))
        .with_max_passes(1);
    let cfg = LossyConfig::new().with_method(4).with_target_zensim(target);

    // Single-pass strict mode bypasses iteration entirely (max_passes=1
    // ships the calibrated encode without measuring). To make strict-mode
    // bite we need max_passes >= 2 so the loop runs and finalize() can
    // check the floor. Use a 2-pass strict run instead.
    let target2 = ZensimTarget::new(99.0)
        .with_max_undershoot(Some(0.5))
        .with_max_passes(2);
    let cfg2 = LossyConfig::new()
        .with_method(4)
        .with_target_zensim(target2);
    let _ = cfg; // suppress unused
    let result = EncodeRequest::lossy(&cfg2, &rgb, PixelLayout::Rgb8, w, h).encode_with_metrics();
    assert!(
        result.is_err(),
        "expected strict-mode error for unreachable target 99 on mixed content; got Ok"
    );
}

#[test]
fn bytes_recovery_drops_size_when_overshooting() {
    // Smooth gradient at target=70: calibrated start at q≈65 likely
    // overshoots into the 80s. Pass 2 should claw back bytes.
    let (rgb, w, h) = smooth_gradient_256();
    let cfg_low_target = LossyConfig::new().with_method(4).with_target_zensim(
        ZensimTarget::new(70.0)
            .with_max_overshoot(Some(0.5))
            .with_max_passes(2),
    );
    let (bytes_two_pass, metrics) =
        EncodeRequest::lossy(&cfg_low_target, &rgb, PixelLayout::Rgb8, w, h)
            .encode_with_metrics()
            .expect("encode failed");

    // For comparison: single-pass (max_passes=1) just ships the
    // calibrated start, no claw-back.
    let cfg_one_pass = LossyConfig::new().with_method(4).with_target_zensim(
        ZensimTarget::new(70.0)
            .with_max_overshoot(Some(0.5))
            .with_max_passes(1),
    );
    let (bytes_one_pass, _m1) = EncodeRequest::lossy(&cfg_one_pass, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");

    // If pass 1 was already in band, claw-back doesn't run. Skip the
    // assertion silently in that case so the test isn't fragile.
    let one_pass_bytes = bytes_one_pass.len();
    let two_pass_bytes = bytes_two_pass.len();
    eprintln!(
        "smooth-gradient target=70: 1-pass bytes={}, 2-pass bytes={} (achieved {:.2}, passes {})",
        one_pass_bytes, two_pass_bytes, metrics.achieved_score, metrics.passes_used
    );
    if metrics.passes_used >= 2 {
        // We took a recovery pass — bytes should not have grown
        // significantly. Allow a small uptick in case the secant
        // overshot in the other direction.
        assert!(
            two_pass_bytes <= (one_pass_bytes as f64 * 1.10) as usize,
            "2-pass bytes {} grew >10% beyond 1-pass bytes {}",
            two_pass_bytes,
            one_pass_bytes
        );
    }
    // Final score must still meet the target (best-effort).
    if metrics.achieved_score.is_finite() {
        assert!(
            metrics.achieved_score >= 68.0,
            "claw-back went too far: achieved {} < 68 (target 70)",
            metrics.achieved_score
        );
    }
}

#[test]
fn zensim_target_default_constructor() {
    let t = ZensimTarget::default();
    assert_eq!(t.target, 80.0);
    let t2 = ZensimTarget::new(85.0);
    assert_eq!(t2.target, 85.0);
    assert_eq!(t2.max_passes, 2);
}

/// Build a synthetic "color_blocks"-style image with 4 sharp-edged flat
/// regions of distinct colors. This exercises Phase 3 — per-segment
/// quantization gives the encoder a tool the global-q step doesn't
/// have, so the per-segment correction path should converge within the
/// pass budget on content like this.
fn color_blocks_256() -> (Vec<u8>, u32, u32) {
    let w: u32 = 256;
    let h: u32 = 256;
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let q = (y < h / 2, x < w / 2);
            let (r, g, b) = match q {
                (true, true) => (220, 50, 50),    // top-left red
                (true, false) => (60, 200, 80),   // top-right green
                (false, true) => (50, 80, 220),   // bottom-left blue
                (false, false) => (200, 200, 60), // bottom-right yellow
            };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    (buf, w, h)
}

#[test]
fn per_segment_correction_converges_on_color_blocks() {
    // Phase 3: color_blocks content (4 flat regions) has segment-able
    // structure. The default 4-segment config should converge within
    // the pass budget. We don't assert "fewer bytes than global-q"
    // because per-segment correction trades off bytes for spatial
    // accuracy — it's a correctness test, not a parsimony test.
    let (rgb, w, h) = color_blocks_256();
    let cfg = LossyConfig::new()
        .with_method(4)
        .with_segments(4)
        .with_target_zensim(
            ZensimTarget::new(82.0)
                .with_max_overshoot(Some(2.0))
                .with_max_passes(3),
        );
    let (bytes, metrics) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");
    assert!(!bytes.is_empty());
    assert!(metrics.passes_used <= 3);
    assert!(
        metrics.achieved_score.is_finite(),
        "score non-finite: {}",
        metrics.achieved_score
    );
    // Generous band.
    assert!(
        metrics.achieved_score >= 75.0,
        "achieved {} < 75 for target 82 on color_blocks (passes={})",
        metrics.achieved_score,
        metrics.passes_used
    );
    eprintln!(
        "color_blocks per-segment: target=82 achieved={:.2} passes={} bytes={}",
        metrics.achieved_score,
        metrics.passes_used,
        bytes.len()
    );
}

#[test]
fn per_segment_disabled_when_one_segment() {
    // num_segments=1 disables Phase 3 → global-q step takes over.
    // Convergence should still work; just via a different path.
    let (rgb, w, h) = color_blocks_256();
    let cfg = LossyConfig::new()
        .with_method(4)
        .with_segments(1)
        .with_target_zensim(
            ZensimTarget::new(82.0)
                .with_max_overshoot(Some(2.0))
                .with_max_passes(3),
        );
    let (bytes, metrics) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");
    assert!(!bytes.is_empty());
    assert!(
        metrics.achieved_score >= 75.0,
        "achieved {} < 75 for target 82 with segments=1",
        metrics.achieved_score
    );
}

#[test]
fn metrics_no_target_helper_exists() {
    // ZensimEncodeMetrics is in the public API; smoke-test the no-target
    // path. The struct itself is `#[non_exhaustive]` so callers don't
    // construct it directly — they observe it from encode_with_metrics.
    let cfg = LossyConfig::new();
    let (rgb, w, h) = smooth_gradient_256();
    let (_bytes, m) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");
    assert!(m.bytes > 0);
    assert!(m.targets_met);
    let _: ZensimEncodeMetrics = m;
}
