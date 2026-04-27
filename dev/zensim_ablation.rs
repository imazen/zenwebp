//! Ablation harness — measure each chunk of the target_zensim PR
//! independently against a real-world corpus.
//!
//! The full PR (commits aefc345..fac6fb5 on `feat/target-zensim`) ships:
//!
//!   - Chunk A: Phase 3 per-segment correction (~250 LOC + EncodeDiagnostics)
//!   - Chunk B: real `segment_map` vs 2x2 quadrant proxy (~80 LOC plumbing)
//!   - Chunk C: per-bucket starting-q calibration (~3 anchor tables + classifier)
//!   - Chunk D: forced `multi_pass_stats=true` inside the loop (~0 LOC, 1 flag)
//!   - Chunk E: Phase 2 fitted anchors (replaced hand-distilled values)
//!   - Chunk F: secant step in pass 2+ (~20 LOC of state)
//!
//! This binary toggles each chunk off via env vars
//! (`ZENWEBP_ABLATE_NO_PHASE3`, `ZENWEBP_ABLATE_NAIVE_Q`,
//! `ZENWEBP_ABLATE_NO_MULTI_PASS_STATS`, `ZENWEBP_ABLATE_PRE_PHASE2_ANCHORS`,
//! `ZENWEBP_ABLATE_NO_SECANT`) and the existing `ZENWEBP_PHASE3_QUADRANT`
//! for chunk B. It runs the same 76-image x 3-target sweep used in the
//! original A/B (CID22 + gb82 + gb82-sc, targets {75, 80, 85},
//! max_overshoot=1.5, max_passes=3, m4) for the BASELINE plus each
//! variant, writes a per-image TSV, and prints a per-variant summary
//! table to stdout.
//!
//! Output TSV format (one row per (variant, image, target)):
//!   variant  corpus  image  width  height  target  passes  achieved
//!   bytes  targets_met
//!
//! Single-threaded (env vars must be flipped between encodes).
//!
//! Usage:
//!   zensim_ablation \
//!     --corpus CID22=/path \
//!     --corpus gb82=/path \
//!     --corpus gb82-sc=/path \
//!     --targets 75,80,85 \
//!     --max-overshoot 1.5 \
//!     --max-passes 3 \
//!     --variants baseline,noA,noB,noC,noD,noE,noF \
//!     --out /mnt/v/output/zenwebp/zensim-ab/ablations/run_2026-04-27.tsv

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use zenwebp::EncodeRequest;
use zenwebp::LossyConfig;
use zenwebp::PixelLayout;
use zenwebp::ZensimTarget;

#[derive(Clone)]
struct Cli {
    corpora: Vec<(String, PathBuf)>,
    targets: Vec<f32>,
    max_overshoot: f32,
    max_passes: u8,
    method: u8,
    variants: Vec<String>,
    out: Option<PathBuf>,
    limit: Option<usize>,
}

fn parse_args() -> Cli {
    let mut corpora: Vec<(String, PathBuf)> = Vec::new();
    let mut targets: Vec<f32> = vec![80.0];
    let mut max_overshoot = 1.5f32;
    let mut max_passes = 3u8;
    let mut method = 4u8;
    let mut variants: Vec<String> = vec![
        "baseline".into(),
        "noA".into(),
        "noB".into(),
        "noC".into(),
        "noD".into(),
        "noE".into(),
        "noF".into(),
        "noA_noD".into(),
    ];
    let mut out: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut args = env::args().skip(1).collect::<Vec<_>>().into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--corpus" => {
                let v = args.next().expect("--corpus needs name=path");
                let (name, path) = v.split_once('=').expect("--corpus expects name=path");
                corpora.push((name.to_string(), PathBuf::from(path)));
            }
            "--targets" => {
                let v = args.next().expect("--targets needs CSV");
                targets = v
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().expect("bad --targets value"))
                    .collect();
            }
            "--max-overshoot" => {
                max_overshoot = args.next().unwrap().parse().unwrap();
            }
            "--max-passes" => {
                max_passes = args.next().unwrap().parse().unwrap();
            }
            "--method" => {
                method = args.next().unwrap().parse().unwrap();
            }
            "--variants" => {
                let v = args.next().expect("--variants needs CSV");
                variants = v.split(',').map(|s| s.trim().to_string()).collect();
            }
            "--out" => {
                out = Some(PathBuf::from(args.next().unwrap()));
            }
            "--limit" => {
                limit = Some(args.next().unwrap().parse().unwrap());
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if corpora.is_empty() {
        eprintln!(
            "usage: --corpus name=/path [--corpus ...] [--targets 75,80,85] [--variants baseline,noA,noB,...] [--out /mnt/v/.../ablation.tsv]"
        );
        std::process::exit(2);
    }
    Cli {
        corpora,
        targets,
        max_overshoot,
        max_passes,
        method,
        variants,
        out,
        limit,
    }
}

fn list_pngs(dir: &PathBuf) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "png" || e == "PNG"))
        .collect();
    v.sort();
    v
}

#[derive(Default, Clone)]
struct Agg {
    n: u32,
    passes_sum: u32,
    dist_sum: f64,
    met: u32,
    undershoot: u32,
    bytes: Vec<u64>,
    // Head-to-head against baseline (not populated for baseline itself).
    // wins = variant strictly better |achieved-target| (margin >= 0.05)
    // losses = baseline strictly better; ties = within 0.05.
    wins_vs_base: u32,
    losses_vs_base: u32,
    ties_vs_base: u32,
}

fn fold(into: &mut Agg, src: &Agg) {
    into.n += src.n;
    into.passes_sum += src.passes_sum;
    into.dist_sum += src.dist_sum;
    into.met += src.met;
    into.undershoot += src.undershoot;
    into.bytes.extend(src.bytes.iter().copied());
    into.wins_vs_base += src.wins_vs_base;
    into.losses_vs_base += src.losses_vs_base;
    into.ties_vs_base += src.ties_vs_base;
}

fn median_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let mut s: Vec<u64> = v.to_vec();
    s.sort_unstable();
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2
    }
}

/// Reset all ablation env vars to off.
fn reset_env() {
    let keys = [
        "ZENWEBP_PHASE3_QUADRANT",
        "ZENWEBP_ABLATE_NO_PHASE3",
        "ZENWEBP_ABLATE_NAIVE_Q",
        "ZENWEBP_ABLATE_NO_MULTI_PASS_STATS",
        "ZENWEBP_ABLATE_PRE_PHASE2_ANCHORS",
        "ZENWEBP_ABLATE_NO_SECANT",
        "ZENWEBP_PHASE3_FINE_GAP",
    ];
    for k in &keys {
        env_set::set(k, None);
    }
}

/// Apply env-var state for a named variant.
fn apply_variant(name: &str) {
    reset_env();
    for tok in name.split('_') {
        match tok {
            "baseline" => {} // all defaults
            "noA" => env_set::set("ZENWEBP_ABLATE_NO_PHASE3", Some("1")),
            "noB" => env_set::set("ZENWEBP_PHASE3_QUADRANT", Some("1")),
            "noC" => env_set::set("ZENWEBP_ABLATE_NAIVE_Q", Some("1")),
            "noD" => env_set::set("ZENWEBP_ABLATE_NO_MULTI_PASS_STATS", Some("1")),
            "noE" => env_set::set("ZENWEBP_ABLATE_PRE_PHASE2_ANCHORS", Some("1")),
            "noF" => env_set::set("ZENWEBP_ABLATE_NO_SECANT", Some("1")),
            // `alwaysOn` recovers the pre-fix net-negative always-on Phase 3
            // (per-segment correction takes precedence over global-q secant
            // even at large gaps). Used to A/B against the post-fix
            // baseline.
            "alwaysOn" => env_set::set("ZENWEBP_PHASE3_FINE_GAP", Some("1000")),
            other => panic!("unknown variant token: {other} (valid: baseline,noA..noF,alwaysOn)"),
        }
    }
}

fn main() {
    let cli = parse_args();

    eprintln!(
        "ablation sweep (method={} max_overshoot={} max_passes={})",
        cli.method, cli.max_overshoot, cli.max_passes
    );
    eprintln!("targets: {:?}", cli.targets);
    eprintln!("variants: {:?}", cli.variants);
    eprintln!();

    // Prep TSV output if requested.
    let mut tsv: Option<std::io::BufWriter<fs::File>> = match &cli.out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let f = fs::File::create(p).expect("cannot create --out file");
            let mut w = std::io::BufWriter::new(f);
            writeln!(
                w,
                "variant\tcorpus\timage\twidth\theight\ttarget\tmax_overshoot\tmax_passes\tmethod\t\
                 passes\tachieved\tbytes\ttargets_met"
            )
            .unwrap();
            Some(w)
        }
        None => None,
    };

    // Load all images up-front so we don't redecode per variant.
    eprintln!("loading corpus...");
    let mut images: Vec<LoadedImage> = Vec::new();
    for (corpus_name, corpus_path) in &cli.corpora {
        let mut paths = list_pngs(corpus_path);
        if let Some(lim) = cli.limit {
            paths.truncate(lim);
        }
        eprintln!(
            "  corpus {corpus_name} ({} images): {}",
            paths.len(),
            corpus_path.display()
        );
        for path in paths {
            let img_name = path.file_name().unwrap().to_string_lossy().to_string();
            match decode_png_rgb(&path) {
                Some((rgb, w, h)) => images.push(LoadedImage {
                    corpus: corpus_name.clone(),
                    name: img_name,
                    rgb,
                    w,
                    h,
                }),
                None => eprintln!("    skip: cannot decode {}", path.display()),
            }
        }
    }
    eprintln!("loaded {} images.", images.len());
    eprintln!();

    // For each (variant, image, target), record the outcome.
    // Variant 0 is the baseline; we use it for head-to-head wins.
    let mut variant_aggs: BTreeMap<(String, String, u32), Agg> = BTreeMap::new();
    let mut variant_outcomes: BTreeMap<String, Vec<Outcome>> = BTreeMap::new();

    let cells = images.len() * cli.targets.len();
    for variant in &cli.variants {
        eprintln!("=== variant: {variant} ({cells} cells) ===");
        apply_variant(variant);
        let t0 = std::time::Instant::now();
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(cells);
        for img in &images {
            for &target in &cli.targets {
                let out = run(
                    &img.rgb,
                    img.w,
                    img.h,
                    target,
                    cli.max_overshoot,
                    cli.max_passes,
                    cli.method,
                );

                if let Some(out_w) = tsv.as_mut() {
                    writeln!(
                        out_w,
                        "{variant}\t{corpus}\t{name}\t{wid}\t{hgt}\t{tgt:.2}\t{mo:.2}\t{mp}\t{me}\t\
                         {p}\t{a:.4}\t{b}\t{m}",
                        variant = variant,
                        corpus = img.corpus,
                        name = img.name,
                        wid = img.w,
                        hgt = img.h,
                        tgt = target,
                        mo = cli.max_overshoot,
                        mp = cli.max_passes,
                        me = cli.method,
                        p = out.passes,
                        a = out.score,
                        b = out.bytes,
                        m = u8::from(out.met),
                    )
                    .unwrap();
                }

                let key = (variant.clone(), img.corpus.clone(), (target * 100.0) as u32);
                let a = variant_aggs.entry(key).or_default();
                a.n += 1;
                a.passes_sum += u32::from(out.passes);
                a.dist_sum += f64::from((out.score - target).abs());
                if out.met {
                    a.met += 1;
                }
                if out.score < target {
                    a.undershoot += 1;
                }
                a.bytes.push(out.bytes as u64);

                outcomes.push(out);
                eprint!(".");
            }
        }
        eprintln!();
        eprintln!(
            "  variant {variant}: {} cells in {:.1}s",
            outcomes.len(),
            t0.elapsed().as_secs_f32()
        );
        variant_outcomes.insert(variant.clone(), outcomes);
    }

    // Compute head-to-head vs baseline for each non-baseline variant.
    if let Some(base_outs) = variant_outcomes.get("baseline") {
        for variant in cli.variants.iter() {
            if variant == "baseline" {
                continue;
            }
            let v_outs = match variant_outcomes.get(variant) {
                Some(v) => v,
                None => continue,
            };
            let mut idx = 0usize;
            for img in &images {
                for &target in &cli.targets {
                    let bo = &base_outs[idx];
                    let vo = &v_outs[idx];
                    let db = (bo.score - target).abs();
                    let dv = (vo.score - target).abs();
                    let key = (variant.clone(), img.corpus.clone(), (target * 100.0) as u32);
                    let a = variant_aggs.get_mut(&key).unwrap();
                    if (dv - db).abs() < 0.05 {
                        a.ties_vs_base += 1;
                    } else if dv < db {
                        a.wins_vs_base += 1;
                    } else {
                        a.losses_vs_base += 1;
                    }
                    idx += 1;
                }
            }
        }
    }

    if let Some(mut out) = tsv {
        let _ = out.flush();
    }

    // Print summary table.
    println!();
    println!("=== summary by (variant, corpus) [aggregated across targets] ===");
    println!(
        "{:14} {:10} {:>4} | {:>5} | {:>7} | {:>9} | {:>5} | {:>9} | win/loss/tie vs base",
        "variant", "corpus", "n", "passes", "|d|", "met", "und", "med-bytes"
    );
    let mut by_variant_corpus: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut by_variant: BTreeMap<String, Agg> = BTreeMap::new();
    let mut by_variant_target: BTreeMap<(String, u32), Agg> = BTreeMap::new();
    for ((variant, corpus, target100), a) in &variant_aggs {
        fold(
            by_variant_corpus
                .entry((variant.clone(), corpus.clone()))
                .or_default(),
            a,
        );
        fold(by_variant.entry(variant.clone()).or_default(), a);
        fold(
            by_variant_target
                .entry((variant.clone(), *target100))
                .or_default(),
            a,
        );
    }
    for ((variant, corpus), a) in &by_variant_corpus {
        if a.n == 0 {
            continue;
        }
        let n = a.n as f32;
        println!(
            "{:14} {:10} {:>4} | {:>5.2} | {:>7.3} | {:>3}/{:<3}  | {:>5} | {:>9} | {}/{}/{}",
            variant,
            corpus,
            a.n,
            a.passes_sum as f32 / n,
            a.dist_sum / a.n as f64,
            a.met,
            a.n,
            a.undershoot,
            median_u64(&a.bytes),
            a.wins_vs_base,
            a.losses_vs_base,
            a.ties_vs_base,
        );
    }

    println!();
    println!("=== summary by (variant, target) [aggregated across corpora] ===");
    println!(
        "{:14} {:>6} {:>4} | {:>5} | {:>7} | {:>9} | {:>5} | {:>9} | win/loss/tie vs base",
        "variant", "target", "n", "passes", "|d|", "met", "und", "med-bytes"
    );
    for ((variant, t100), a) in &by_variant_target {
        if a.n == 0 {
            continue;
        }
        let n = a.n as f32;
        println!(
            "{:14} {:>6.2} {:>4} | {:>5.2} | {:>7.3} | {:>3}/{:<3}  | {:>5} | {:>9} | {}/{}/{}",
            variant,
            *t100 as f32 / 100.0,
            a.n,
            a.passes_sum as f32 / n,
            a.dist_sum / a.n as f64,
            a.met,
            a.n,
            a.undershoot,
            median_u64(&a.bytes),
            a.wins_vs_base,
            a.losses_vs_base,
            a.ties_vs_base,
        );
    }

    println!();
    println!("=== aggregate by variant (all corpora x all targets) ===");
    println!(
        "{:14} {:>4} | {:>5} | {:>7} | {:>9} | {:>5} | {:>9} | win/loss/tie vs base",
        "variant", "n", "passes", "|d|", "met", "und", "med-bytes"
    );
    for variant in cli.variants.iter() {
        let a = match by_variant.get(variant) {
            Some(a) => a,
            None => continue,
        };
        if a.n == 0 {
            continue;
        }
        let n = a.n as f32;
        println!(
            "{:14} {:>4} | {:>5.2} | {:>7.3} | {:>3}/{:<3}  | {:>5} | {:>9} | {}/{}/{}",
            variant,
            a.n,
            a.passes_sum as f32 / n,
            a.dist_sum / a.n as f64,
            a.met,
            a.n,
            a.undershoot,
            median_u64(&a.bytes),
            a.wins_vs_base,
            a.losses_vs_base,
            a.ties_vs_base,
        );
    }
}

struct LoadedImage {
    corpus: String,
    name: String,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

#[derive(Clone, Copy)]
struct Outcome {
    score: f32,
    passes: u8,
    bytes: usize,
    met: bool,
}

fn run(
    rgb: &[u8],
    w: u32,
    h: u32,
    target: f32,
    max_overshoot: f32,
    max_passes: u8,
    method: u8,
) -> Outcome {
    let cfg = LossyConfig::new()
        .with_method(method)
        .with_segments(4)
        .with_target_zensim_target(
            ZensimTarget::new(target)
                .with_max_overshoot(Some(max_overshoot))
                .with_max_passes(max_passes),
        );
    let (b, m) = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h)
        .encode_with_metrics()
        .expect("encode failed");
    Outcome {
        score: m.achieved_score,
        passes: m.passes_used,
        bytes: b.len(),
        met: m.targets_met,
    }
}

mod env_set {
    pub fn set(key: &str, val: Option<&str>) {
        // SAFETY: single-threaded driver binary; no other threads can
        // observe stale env pointers. Only used by `dev/` tooling.
        unsafe {
            match val {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
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
