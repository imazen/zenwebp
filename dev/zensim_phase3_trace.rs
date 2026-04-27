//! Single-image Phase 3 tracer. Runs target_zensim on one (or a few)
//! PNG files at a chosen target/max_overshoot/max_passes and prints the
//! `PHASE3_TRACE` lines to stderr (the iteration-loop instrumentation
//! is enabled here via [`zenwebp::AblationToggles::trace_phase3`]).
//!
//! Usage:
//!   zensim_phase3_trace --target 80 \
//!     --max-passes 3 --max-overshoot 1.5 \
//!     --variants baseline,noA \
//!     --image /path/to/foo.png \
//!     [--image /path/to/bar.png ...]
//!
//! For each (variant, image) the binary prints:
//!   ===VARIANT===<variant>===
//!   ===IMAGE===<path>===
//!   <PHASE3_TRACE lines for this run>
//!   ===RESULT=== achieved=... bytes=... passes=... met=...

use std::env;
use std::fs;
use std::path::PathBuf;

use zenwebp::AblationToggles;
use zenwebp::EncodeRequest;
use zenwebp::LossyConfig;
use zenwebp::PixelLayout;
use zenwebp::ZensimTarget;
use zenwebp::set_ablation_toggles;

fn apply_variant(name: &str) {
    // Always force trace_phase3=true for this binary's purpose.
    let mut t = AblationToggles {
        trace_phase3: true,
        ..AblationToggles::default()
    };
    for tok in name.split('_') {
        match tok {
            "baseline" => {}
            "noA" => t.disable_phase3 = true,
            "noB" => t.use_quadrant_proxy = true,
            "noC" => t.naive_starting_q = true,
            "noD" => t.no_multi_pass_stats = true,
            "noE" => t.pre_phase2_anchors = true,
            "noF" => t.no_secant = true,
            "alwaysOn" => t.phase3_fine_gap = Some(1000.0),
            other => panic!("unknown variant token: {other}"),
        }
    }
    set_ablation_toggles(t);
}

fn decode_png_rgb(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = fs::read(path).ok()?;
    let cur = std::io::Cursor::new(bytes);
    let mut decoder = png::Decoder::new(cur);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    let w = info.width;
    let h = info.height;
    let color = info.color_type;
    let bytes_per_pixel = match color {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => 3,
    };
    let mut buf = vec![0u8; (w as usize * h as usize) * bytes_per_pixel];
    reader.next_frame(&mut buf).ok()?;
    let rgb = match color {
        png::ColorType::Rgb | png::ColorType::Indexed => buf,
        png::ColorType::Rgba => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(4) {
                out.extend_from_slice(&[px[0], px[1], px[2]]);
            }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf {
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            out
        }
    };
    Some((rgb, w, h))
}

fn main() {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut target = 80.0f32;
    let mut max_overshoot = 1.5f32;
    let mut max_passes = 3u8;
    let mut method = 4u8;
    let mut variants: Vec<String> = vec!["baseline".into(), "noA".into()];
    let mut args = env::args().skip(1).collect::<Vec<_>>().into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--image" => images.push(PathBuf::from(args.next().unwrap())),
            "--target" => target = args.next().unwrap().parse().unwrap(),
            "--max-overshoot" => max_overshoot = args.next().unwrap().parse().unwrap(),
            "--max-passes" => max_passes = args.next().unwrap().parse().unwrap(),
            "--method" => method = args.next().unwrap().parse().unwrap(),
            "--variants" => {
                variants = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.to_string())
                    .collect();
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if images.is_empty() {
        eprintln!(
            "usage: --image <path> [--image ...] [--target 80] [--max-passes 3] [--variants baseline,noA]"
        );
        std::process::exit(2);
    }
    eprintln!(
        "zensim_phase3_trace: target={target} max_overshoot={max_overshoot} max_passes={max_passes} method={method}"
    );
    eprintln!("variants: {:?}", variants);

    for variant in &variants {
        eprintln!("\n===VARIANT==={variant}===");
        for img_path in &images {
            apply_variant(variant);
            let (rgb, w, h) = match decode_png_rgb(img_path) {
                Some(t) => t,
                None => {
                    eprintln!("skip: cannot decode {}", img_path.display());
                    continue;
                }
            };
            eprintln!("===IMAGE==={}===", img_path.display());
            let cfg = LossyConfig::new()
                .with_method(method)
                .with_segments(4)
                .with_target_zensim(
                    ZensimTarget::new(target)
                        .with_max_overshoot(Some(max_overshoot))
                        .with_max_passes(max_passes),
                );
            let (b, m) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
                .encode_with_metrics()
                .expect("encode");
            eprintln!(
                "===RESULT=== variant={variant} img={} achieved={:.4} bytes={} passes={} met={}",
                img_path.display(),
                m.achieved_score,
                b.len(),
                m.passes_used,
                m.targets_met,
            );
        }
    }
}
