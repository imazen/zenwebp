//! Paired wall-time RATIO canaries for operating-point cliffs (class E in
//! docs/BUG_RETROSPECTIVE_2026-08.md).
//!
//! Absolute timings are useless on shared CI runners (machine generations
//! vary 2-3x day to day), but a RATIO of two arms measured back-to-back in
//! the same process on the same input is stable — profile changes (debug vs
//! release, coverage instrumentation) slow both arms together.
//!
//! The m6-alpha canary guards the #75 class: the tuned default escalating
//! the alpha plane's VP8L to quality 100 at m6 made lossy-RGBA m6 ~40x
//! slower than m4 (558ms vs ~14ms on 150x150). Healthy m6/m4 sits around
//! 2-8x across debug/release on this shape; the threshold of 18x splits the
//! two distributions with a wide margin on both sides. The PRIMARY gate for
//! the specific #75 regression is `alpha_vp8l_operating_point_pinned` (a
//! pure unit pin); this canary exists to catch the rest of the class —
//! any change that quietly makes one method tier disproportionately
//! expensive on the alpha path.
//!
//! If this test ever flakes, the fix is to investigate the ratio — not to
//! raise the threshold without a measurement (CLAUDE.md: never relax a test
//! without user confirmation).

use std::time::Instant;
use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};

/// 96x96 antialiased-disc RGBA: many distinct alpha levels, the shape that
/// drives the ALPH coder through its full pipeline (same family as #75's
/// repro).
fn alpha_disc_rgba(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 4];
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let r_out = w.min(h) as f32 * 0.45;
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let a = ((r_out - d + 1.5) / 3.0).clamp(0.0, 1.0);
            let i = (y * w + x) * 4;
            px[i] = (x * 255 / w) as u8;
            px[i + 1] = (y * 255 / h) as u8;
            px[i + 2] = ((x ^ y) & 0xff) as u8;
            px[i + 3] = (a * 255.0) as u8;
        }
    }
    px
}

/// Min-of-N wall time for one encode configuration. Min (not mean) is the
/// robust statistic against scheduler noise on shared runners.
fn min_encode_time(cfg: &LossyConfig, rgba: &[u8], w: u32, h: u32, runs: u32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let out = EncodeRequest::new(
            &EncoderConfig::Lossy(cfg.clone()),
            rgba,
            PixelLayout::Rgba8,
            w,
            h,
        )
        .encode()
        .expect("encode must succeed");
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        best = best.min(dt);
    }
    best
}

#[test]
fn m6_alpha_encode_is_not_a_cliff_over_m4() {
    let (w, h) = (96usize, 96usize);
    let rgba = alpha_disc_rgba(w, h);

    // Warm-up once per arm (page-in, lazy statics), then min-of-3.
    let m4 = LossyConfig::new().with_quality(80.0).with_method(4);
    let m6 = LossyConfig::new().with_quality(80.0).with_method(6);
    let _ = min_encode_time(&m4, &rgba, w as u32, h as u32, 1);
    let _ = min_encode_time(&m6, &rgba, w as u32, h as u32, 1);
    let t4 = min_encode_time(&m4, &rgba, w as u32, h as u32, 3);
    let t6 = min_encode_time(&m6, &rgba, w as u32, h as u32, 3);

    let ratio = t6 / t4;
    eprintln!(
        "[perf_cliff_canary] m4={:.1}ms m6={:.1}ms ratio={ratio:.2}x",
        t4 * 1e3,
        t6 * 1e3
    );
    assert!(
        ratio < 18.0,
        "m6/m4 lossy-alpha encode ratio {ratio:.1}x (m4 {:.1}ms, m6 {:.1}ms) — \
         an operating-point cliff is back (the #75 escalation measured ~40x). \
         See docs/BUG_RETROSPECTIVE_2026-08.md class E.",
        t4 * 1e3,
        t6 * 1e3
    );
}
