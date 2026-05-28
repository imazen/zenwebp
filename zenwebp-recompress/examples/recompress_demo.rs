//! End-to-end demonstration of the zenwebp-recompress public API.
//!
//! Run with:
//!
//! ```text
//! cargo run --example recompress_demo
//! ```
//!
//! This uses only the frozen public API (no `expert` feature). It
//! synthesizes a lossy WebP source in memory, then shows:
//!
//! 1. `plan()` — preview the router's decision without running the strategy.
//! 2. `recompress()` with `Budget::OneShot` — one decode + encode.
//! 3. `recompress()` with `Budget::MaxIterations` — measured size search.
//! 4. Inspecting every `RecompressResult` arm, including the
//!    `better_handled_by_jxl` hint and the chosen `StrategyKind`.

use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp_recompress::{Budget, Plan, RecompressOptions, RecompressResult, plan, recompress};

fn main() {
    // --- Make a source WebP to recompress (stands in for a user's file). ---
    // A higher-quality source (q90) has headroom: recompressing toward a
    // lower perceptual target genuinely saves bytes.
    let (w, h) = (384u32, 384u32);
    let rgba = synthetic_photo(w, h);
    let source = EncodeRequest::lossy(
        &LossyConfig::new().with_quality(90.0).with_method(4),
        &rgba,
        PixelLayout::Rgba8,
        w,
        h,
    )
    .encode()
    .expect("encode source");
    println!("source: {} bytes ({}x{}, lossy q90)\n", source.len(), w, h);

    // --- 1. Preview the decision (decodes + estimates, but does not run
    //        the chosen strategy — cheaper than a full recompress). ---
    let target = 65.0;
    let opts = RecompressOptions {
        target_zensim_a: target,
        budget: Budget::OneShot,
        ..Default::default()
    };
    match plan(&source, &opts).expect("plan") {
        Plan::Recompress {
            strategy,
            projected_zensim_a,
            projected_size_ratio,
            better_handled_by_jxl,
        } => println!(
            "plan @ target {target}: Recompress via {strategy:?}\n  \
             projected zensim-A {projected_zensim_a:.1}, size ratio {projected_size_ratio:.3}, \
             jxl-hint {better_handled_by_jxl}"
        ),
        Plan::LosslessOnly {
            reason,
            better_handled_by_jxl,
        } => println!(
            "plan @ target {target}: LosslessOnly ({reason:?}), jxl-hint {better_handled_by_jxl}"
        ),
        Plan::NoOp { reason } => println!("plan @ target {target}: NoOp ({reason:?})"),
        _ => {}
    }

    // --- 2. One-shot recompress (production default). ---
    println!("\n-- OneShot --");
    report(&source, recompress(&source, &opts).expect("recompress"));

    // --- 3. Measured, iterative recompress (offline / best-effort). ---
    println!("\n-- MaxIterations(6) --");
    let iter_opts = RecompressOptions {
        target_zensim_a: target,
        budget: Budget::MaxIterations(6),
        ..Default::default()
    };
    report(
        &source,
        recompress(&source, &iter_opts).expect("recompress"),
    );

    // --- 4. A target the source can't reach → lossless-only fallback. ---
    println!("\n-- target 99 (unreachable from a q75 source) --");
    let high = RecompressOptions {
        target_zensim_a: 99.0,
        budget: Budget::OneShot,
        ..Default::default()
    };
    report(&source, recompress(&source, &high).expect("recompress"));
}

fn report(source: &[u8], result: RecompressResult) {
    // `into_bytes` returns the optimized bytes for any arm; here we destructure
    // to show the telemetry the API exposes.
    match result {
        RecompressResult::Recompressed {
            bytes,
            strategy,
            projected_zensim_a,
            measured_zensim_a,
            source_to_output_ratio,
            better_handled_by_jxl,
        } => {
            println!(
                "Recompressed via {strategy:?}: {} -> {} bytes ({:.1}% of source)",
                source.len(),
                bytes.len(),
                source_to_output_ratio * 100.0
            );
            println!(
                "  projected zensim-A {projected_zensim_a:.1}, measured {}",
                measured_zensim_a
                    .map(|m| format!("{m:.1}"))
                    .unwrap_or_else(|| "n/a (one-shot)".into())
            );
            if better_handled_by_jxl {
                println!("  hint: a JXL transcode would likely serve this image better");
            }
        }
        RecompressResult::LosslessOnly {
            bytes,
            reason,
            better_handled_by_jxl,
        } => {
            println!(
                "LosslessOnly ({reason:?}): {} -> {} bytes; recompression would not have helped",
                source.len(),
                bytes.len()
            );
            if better_handled_by_jxl {
                println!("  hint: try JXL instead of WebP for this image + target");
            }
        }
        RecompressResult::NoOp { reason } => {
            println!("NoOp ({reason:?}): source already meets target");
        }
        _ => {}
    }
}

/// Smooth gradients with gentle structured detail — compresses like a
/// photo (low high-frequency noise so a q90 encode has real headroom).
fn synthetic_photo(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            // Low-amplitude texture (≤7) on top of smooth gradients.
            let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 13) & 0x07;
            buf[i] = (((x * 200) / w) + n) as u8;
            buf[i + 1] = (((y * 200) / h) + n) as u8;
            buf[i + 2] = ((((x + y) * 200) / (w + h)) + n) as u8;
            buf[i + 3] = 255;
        }
    }
    buf
}
