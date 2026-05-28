//! `zwr` — demo CLI for zenwebp-recompress.
//!
//! ```text
//! zwr <INPUT.webp> [--target 80] [--out OUT.webp] [--plan] [--analyze] [--iterations N]
//! ```
//!
//! - `--plan` prints the router decision without running any strategy.
//! - `--analyze` prints a full source-analysis dump (dimensions, qi,
//!   encoder family, content class, projected achievable target).
//! - `--iterations N` runs measured budget with up to N secant passes.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use zenwebp_recompress::expert::{analyze_source, classify};
use zenwebp_recompress::{Budget, Plan, RecompressOptions, RecompressResult, plan, recompress};

fn usage() -> ExitCode {
    eprintln!(
        "usage: zwr <INPUT.webp> [--target N] [--out OUT.webp] [--plan] [--analyze]\n\n\
         --target N      Target zensim Profile A score, 0..100. Default: 80.\n\
         --out PATH      Output WebP path. Default: <INPUT>.opt.webp.\n\
         --plan          Print router decision only; do not run any strategy.\n\
         --analyze       Print full source analysis (no recompression).\n\
         --iterations N  Run measured budget with up to N iterations.\n"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut target: f32 = 80.0;
    let mut plan_only = false;
    let mut analyze_only = false;
    let mut iterations: Option<u32> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--target" => {
                target = match args.next().and_then(|s| s.parse().ok()) {
                    Some(v) => v,
                    None => return usage(),
                };
            }
            "--out" => {
                out = args.next().map(PathBuf::from);
            }
            "--plan" => plan_only = true,
            "--analyze" => analyze_only = true,
            "--iterations" => {
                iterations = args.next().and_then(|s| s.parse().ok());
            }
            "-h" | "--help" => return usage(),
            other if !other.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(other));
            }
            _ => return usage(),
        }
    }
    let Some(input) = input else {
        return usage();
    };
    let out = out.unwrap_or_else(|| {
        let mut p = input.clone();
        let stem = p.file_stem().unwrap_or_default().to_owned();
        p.set_file_name(format!("{}.opt.webp", stem.to_string_lossy()));
        p
    });

    let bytes = match fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: read {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    let opts = RecompressOptions {
        target_zensim_a: target,
        budget: iterations
            .map(Budget::MaxIterations)
            .unwrap_or(Budget::OneShot),
        ..Default::default()
    };

    if analyze_only {
        match analyze_source(&bytes) {
            Ok(a) => {
                println!("dimensions   = {}×{}", a.width, a.height);
                println!("kind         = {:?}", a.kind);
                println!(
                    "source_q     = {:.2} (libwebp-equivalent estimate)",
                    a.source_q
                );
                println!(
                    "quantizer_idx= {} (VP8 base quantizer, 0-127 lower=better)",
                    a.vp8_quantizer_index
                );
                println!("filter_level = {}", a.vp8_filter_level);
                println!("sharpness    = {}", a.vp8_sharpness_level);
                println!("has_segment_quant = {}", a.has_segment_quant);
                println!("has_alpha    = {}", a.has_alpha);
                println!("has_icc      = {}", a.has_icc);
                println!("encoder      = {:?}", a.encoder_family);
                println!("default content_class = {:?}", a.content_class);
                // Decode once for the heuristic classifier + the reliable
                // recompression-self-consistency quality estimate.
                if let Ok((rgba, w, h)) = zenwebp::oneshot::decode_rgba(&bytes) {
                    let cls = classify(&rgba, w as usize, h as usize);
                    println!("decoded content_class = {:?}", cls);
                    if let Ok(eq) = zenwebp_recompress::expert::estimate_quality_by_recompression(
                        &rgba,
                        w,
                        h,
                        bytes.len(),
                    ) {
                        println!(
                            "estimated_quality = {eq:.1} (decode-based; the reliable calibration key)"
                        );
                    }
                }
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("error: analyze: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if plan_only {
        match plan(&bytes, &opts) {
            Ok(Plan::Recompress {
                strategy,
                projected_zensim_a,
                projected_size_ratio,
                better_handled_by_jxl,
            }) => {
                println!(
                    "plan=Recompress strategy={strategy:?} projected_zensim_a={projected_zensim_a:.2} \
                     projected_size_ratio={projected_size_ratio:.4} better_handled_by_jxl={better_handled_by_jxl}"
                );
            }
            Ok(Plan::LosslessOnly {
                reason,
                better_handled_by_jxl,
            }) => {
                println!(
                    "plan=LosslessOnly reason={reason:?} better_handled_by_jxl={better_handled_by_jxl}"
                );
            }
            Ok(Plan::NoOp { reason }) => println!("plan=NoOp reason={reason:?}"),
            Ok(_) => println!("plan=Other"),
            Err(e) => {
                eprintln!("error: plan: {e}");
                return ExitCode::from(1);
            }
        }
        return ExitCode::SUCCESS;
    }

    match recompress(&bytes, &opts) {
        Ok(RecompressResult::Recompressed {
            bytes: out_bytes,
            strategy,
            projected_zensim_a,
            measured_zensim_a,
            source_to_output_ratio,
            better_handled_by_jxl,
            ..
        }) => {
            if let Err(e) = fs::write(&out, &out_bytes) {
                eprintln!("error: write {}: {e}", out.display());
                return ExitCode::from(1);
            }
            println!(
                "wrote {} ({} bytes -> {} bytes, ratio {:.4}, strategy={strategy:?}, \
                 projected_zensim_a={projected_zensim_a:.2}, measured_zensim_a={}, \
                 better_handled_by_jxl={better_handled_by_jxl})",
                out.display(),
                bytes.len(),
                out_bytes.len(),
                source_to_output_ratio,
                measured_zensim_a
                    .map(|m| format!("{m:.2}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            );
        }
        Ok(RecompressResult::LosslessOnly {
            bytes: out_bytes,
            reason,
            better_handled_by_jxl,
            ..
        }) => {
            if let Err(e) = fs::write(&out, &out_bytes) {
                eprintln!("error: write {}: {e}", out.display());
                return ExitCode::from(1);
            }
            println!(
                "wrote {} ({} bytes -> {} bytes, LosslessOnly reason={reason:?}, \
                 better_handled_by_jxl={better_handled_by_jxl})",
                out.display(),
                bytes.len(),
                out_bytes.len(),
            );
        }
        Ok(RecompressResult::NoOp { reason }) => {
            println!("no-op (source already meets target): {reason:?}");
        }
        Ok(_) => println!("other result"),
        Err(e) => {
            eprintln!("error: recompress: {e}");
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
