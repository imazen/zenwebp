//! Robustness gate for the project's core requirement:
//!
//! > must work well for images already encoded between quality 20 and 100,
//! > step 2, for any 0 to 100 zensim target.
//!
//! For every source quality in `20..=100 step 2` and a spread of zensim
//! targets, recompress must:
//!
//! 1. Never error.
//! 2. Always produce a valid WebP (or a clean NoOp).
//! 3. Never ship a strictly-dominated result: a `Recompressed` output must
//!    actually shrink the file (`source_to_output_ratio < 1.0`). If it
//!    can't shrink, the router must fall back to `LosslessOnly` / `NoOp`,
//!    not ship a larger file.
//! 4. Never report the unimplemented `CoeffEdit` or the falsified
//!    `DeblockReencode` strategy.

use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp_recompress::{Budget, RecompressOptions, RecompressResult, StrategyKind, recompress};

/// A photo-ish gradient + noise so the encoder produces a realistic
/// bitstream at every quality (not a degenerate flat image).
fn make_photo_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            // Smooth gradient + a pseudo-random high-freq term.
            let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 13) & 0x1f;
            let r = (((x * 200) / w.max(1)) + n) as u8;
            let g = (((y * 200) / h.max(1)) + n) as u8;
            let b = ((((x + y) * 200) / (w + h).max(1)) + n) as u8;
            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
            buf[i + 3] = 255;
        }
    }
    buf
}

fn encode_at(q: f32, rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let cfg = LossyConfig::new().with_quality(q).with_method(4);
    EncodeRequest::lossy(&cfg, rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .expect("encode source")
}

fn is_valid_webp(bytes: &[u8]) -> bool {
    bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

#[test]
fn quality_20_to_100_step_2_all_targets_hold_invariants() {
    let (w, h) = (128u32, 128u32);
    let rgba = make_photo_rgba(w, h);

    let mut source_q = 20u32;
    let mut cells = 0u32;
    while source_q <= 100 {
        let src = encode_at(source_q as f32, &rgba, w, h);
        assert!(is_valid_webp(&src), "source encode q{source_q} invalid");

        for target in [10.0f32, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
            let opts = RecompressOptions {
                target_zensim_a: target,
                budget: Budget::OneShot,
                ..Default::default()
            };
            let res = recompress(&src, &opts).unwrap_or_else(|e| {
                panic!("recompress errored at q{source_q} target{target}: {e}")
            });

            match res {
                RecompressResult::Recompressed {
                    bytes,
                    strategy,
                    source_to_output_ratio,
                    ..
                } => {
                    assert!(
                        is_valid_webp(&bytes),
                        "invalid output q{source_q} target{target}"
                    );
                    assert!(
                        !matches!(
                            strategy,
                            StrategyKind::CoeffEdit | StrategyKind::DeblockReencode
                        ),
                        "router selected de-selected strategy {strategy:?} at q{source_q} target{target}"
                    );
                    // A shipped recompression must actually shrink — never
                    // a strictly-dominated larger-at-lower-quality result.
                    assert!(
                        source_to_output_ratio < 1.0 + 1e-3,
                        "Recompressed grew the file ({source_to_output_ratio:.4}) at q{source_q} target{target} via {strategy:?}"
                    );
                }
                RecompressResult::LosslessOnly { bytes, .. } => {
                    assert!(
                        is_valid_webp(&bytes),
                        "invalid lossless-only output q{source_q} target{target}"
                    );
                }
                RecompressResult::NoOp { .. } => {}
                _ => {}
            }
            cells += 1;
        }
        source_q += 2;
    }
    // 41 source qualities × 8 targets.
    assert_eq!(cells, 41 * 8, "expected full grid coverage");
}

#[test]
fn measured_budget_never_overshoots_into_larger_file() {
    // Under MaxIterations the secant must not converge to a result that's
    // both larger than the source AND a recompression. (It may legitimately
    // fall back to LosslessOnly.)
    let (w, h) = (96u32, 96u32);
    let rgba = make_photo_rgba(w, h);
    for source_q in [30u32, 50, 70, 90] {
        let src = encode_at(source_q as f32, &rgba, w, h);
        for target in [50.0f32, 70.0, 85.0] {
            let opts = RecompressOptions {
                target_zensim_a: target,
                budget: Budget::MaxIterations(6),
                ..Default::default()
            };
            let res = recompress(&src, &opts).expect("recompress");
            if let RecompressResult::Recompressed {
                source_to_output_ratio,
                bytes,
                ..
            } = res
            {
                assert!(is_valid_webp(&bytes));
                assert!(
                    source_to_output_ratio < 1.0 + 1e-3,
                    "iterated recompression grew file at q{source_q} target{target}: {source_to_output_ratio:.4}"
                );
            }
        }
    }
}
