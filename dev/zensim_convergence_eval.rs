//! Run target_zensim against one or more corpora and report convergence
//! stats (single-pass-success rate, avg passes, achieved-score
//! distribution, undershoot/overshoot/in-band counts).
//!
//! Usage (legacy positional form, single corpus + single target):
//!   zensim_convergence_eval [corpus_dir] [target] [max_passes]
//!
//! Usage (flag form, multi-corpus + multi-target + asymmetric ship-band
//! sweep):
//!   zensim_convergence_eval \
//!     --corpus CID22=/path/to/CID22/CID22-512/validation \
//!     --corpus gb82=/path/to/gb82 \
//!     --corpus gb82-sc=/path/to/gb82-sc \
//!     --targets 75,80,85 \
//!     --max-passes 2 \
//!     --max-overshoot 1.5 \
//!     [--max-undershoot-ship 0.5]   # explicit ship-band override
//!     [--strict-ship]               # equivalent to --max-undershoot-ship 0
//!     [--limit 30]
//!
//! `--strict-ship` recovers the pre-Task-2 (PR #48) iteration semantics
//! where any undershoot — even 0.001 — triggers a second pass. The
//! default ship-band (Some(0.5)) is the post-Task-2 default. Pair the
//! two runs to A/B the ship-band change.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

#[derive(Default)]
struct Cli {
    corpora: Vec<(String, PathBuf)>,
    targets: Vec<f32>,
    max_passes: u8,
    max_overshoot: f32,
    max_undershoot_ship: Option<Option<f32>>, // outer Option = "user provided?"
    method: u8,
    limit: Option<usize>,
}

fn parse_args() -> Cli {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cli = Cli {
        corpora: vec![],
        targets: vec![],
        max_passes: 2,
        max_overshoot: 1.5,
        max_undershoot_ship: None,
        method: 4,
        limit: None,
    };
    // If no flags are present, fall back to the historical positional form.
    let has_flag = argv.iter().any(|a| a.starts_with("--"));
    if !has_flag {
        let corpus_dir = argv.first().cloned().unwrap_or_else(|| {
            let home = env::var("HOME").expect("HOME not set");
            format!("{home}/work/codec-corpus/CID22/CID22-512/validation")
        });
        let target_v: f32 = argv.get(1).and_then(|s| s.parse().ok()).unwrap_or(80.0);
        let max_passes: u8 = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
        cli.corpora
            .push(("default".into(), PathBuf::from(corpus_dir)));
        cli.targets.push(target_v);
        cli.max_passes = max_passes;
        return cli;
    }
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let next = || -> &str { argv.get(i + 1).map(|s| s.as_str()).unwrap_or("") };
        match a.as_str() {
            "--corpus" => {
                let v = next();
                let (name, path) = v.split_once('=').expect("--corpus expects name=path");
                cli.corpora.push((name.to_string(), PathBuf::from(path)));
                i += 2;
            }
            "--targets" => {
                cli.targets = next()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().expect("bad --targets value"))
                    .collect();
                i += 2;
            }
            "--max-passes" => {
                cli.max_passes = next().parse().unwrap();
                i += 2;
            }
            "--max-overshoot" => {
                cli.max_overshoot = next().parse().unwrap();
                i += 2;
            }
            "--max-undershoot-ship" => {
                cli.max_undershoot_ship = Some(Some(next().parse().unwrap()));
                i += 2;
            }
            "--strict-ship" => {
                cli.max_undershoot_ship = Some(None); // means: explicitly set field to None
                i += 1;
            }
            "--method" => {
                cli.method = next().parse().unwrap();
                i += 2;
            }
            "--limit" => {
                cli.limit = Some(next().parse().unwrap());
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if cli.corpora.is_empty() {
        eprintln!(
            "usage: --corpus name=/path [--corpus ...] --targets 75,80,85 [--max-passes 2] [--strict-ship]"
        );
        std::process::exit(2);
    }
    if cli.targets.is_empty() {
        cli.targets.push(80.0);
    }
    cli
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
    passes_one: u32,
    achieved_sum: f64,
    targets_met: u32,
    undershoot: u32,
    overshoot: u32,
    bytes: Vec<u64>,
}

fn fold(into: &mut Agg, src: &Agg) {
    into.n += src.n;
    into.passes_sum += src.passes_sum;
    into.passes_one += src.passes_one;
    into.achieved_sum += src.achieved_sum;
    into.targets_met += src.targets_met;
    into.undershoot += src.undershoot;
    into.overshoot += src.overshoot;
    into.bytes.extend(src.bytes.iter().copied());
}

fn median_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2
    }
}

fn print_summary(label: &str, target: Option<f32>, a: &Agg) {
    if a.n == 0 {
        return;
    }
    let n = a.n as f32;
    let tgt_str = target.map(|t| format!("{t:.1}")).unwrap_or("ALL".into());
    println!(
        "{:24} target={:>5} n={:>4}  passes_avg={:.2}  pass1_share={:.0}%  achieved_avg={:.2}  met={}/{}  und={} ovs={}  med_bytes={}",
        label,
        tgt_str,
        a.n,
        a.passes_sum as f32 / n,
        100.0 * a.passes_one as f32 / n,
        a.achieved_sum / a.n as f64,
        a.targets_met,
        a.n,
        a.undershoot,
        a.overshoot,
        median_u64(&a.bytes),
    );
}

struct LoadedImage {
    corpus: String,
    name: String,
    rgb: Vec<u8>,
    rgba: Option<Vec<u8>>,
    w: u32,
    h: u32,
}

fn main() {
    let cli = parse_args();

    eprintln!(
        "convergence-eval: max_passes={} max_overshoot={} method={} ship_override={:?}",
        cli.max_passes, cli.max_overshoot, cli.method, cli.max_undershoot_ship
    );
    eprintln!("corpora:");
    for (n, p) in &cli.corpora {
        eprintln!("  {n} = {}", p.display());
    }
    eprintln!("targets: {:?}", cli.targets);

    // Load images.
    let mut images: Vec<LoadedImage> = Vec::new();
    for (corpus_name, corpus_path) in &cli.corpora {
        let mut paths = list_pngs(corpus_path);
        if let Some(lim) = cli.limit {
            paths.truncate(lim);
        }
        eprintln!("  {corpus_name}: {} images", paths.len());
        for path in paths {
            let img_name = path.file_name().unwrap().to_string_lossy().to_string();
            match decode_png(&path) {
                Some(decoded) => images.push(LoadedImage {
                    corpus: corpus_name.clone(),
                    name: img_name,
                    rgb: decoded.rgb,
                    rgba: decoded.rgba,
                    w: decoded.w,
                    h: decoded.h,
                }),
                None => eprintln!("    skip {}", path.display()),
            }
        }
    }
    eprintln!("loaded {} images", images.len());

    // Run.
    let mut by_target: BTreeMap<u32, Agg> = BTreeMap::new();
    let mut by_corpus: BTreeMap<String, Agg> = BTreeMap::new();
    let mut by_corpus_target: BTreeMap<(String, u32), Agg> = BTreeMap::new();
    let mut overall = Agg::default();

    for img in &images {
        let _ = &img.rgba; // silence dead-code when no rgba uses (handled by extended eval binary)
        for &target in &cli.targets {
            let mut t = ZensimTarget::new(target)
                .with_max_overshoot(Some(cli.max_overshoot))
                .with_max_passes(cli.max_passes);
            if let Some(ship) = cli.max_undershoot_ship {
                t = t.with_max_undershoot_ship(ship);
            }
            let cfg = LossyConfig::new()
                .with_method(cli.method)
                .with_target_zensim(t);
            let result = EncodeRequest::lossy(&cfg, &img.rgb, PixelLayout::Rgb8, img.w, img.h)
                .encode_with_metrics();
            match result {
                Ok((b, m)) => {
                    let a = Agg {
                        n: 1,
                        passes_sum: u32::from(m.passes_used),
                        passes_one: u32::from(m.passes_used == 1),
                        achieved_sum: f64::from(m.achieved_score),
                        targets_met: u32::from(m.targets_met),
                        undershoot: u32::from(m.achieved_score < target),
                        overshoot: u32::from(m.achieved_score > target + cli.max_overshoot),
                        bytes: vec![b.len() as u64],
                    };
                    fold(&mut overall, &a);
                    fold(by_target.entry((target * 100.0) as u32).or_default(), &a);
                    fold(by_corpus.entry(img.corpus.clone()).or_default(), &a);
                    fold(
                        by_corpus_target
                            .entry((img.corpus.clone(), (target * 100.0) as u32))
                            .or_default(),
                        &a,
                    );
                }
                Err(e) => {
                    eprintln!("ERROR {} target={}: {:?}", img.name, target, e);
                }
            }
        }
        eprint!(".");
    }
    eprintln!();

    println!("\n=== by target (across all corpora) ===");
    for (t100, a) in &by_target {
        print_summary("(all)", Some(*t100 as f32 / 100.0), a);
    }
    println!("\n=== by corpus (across all targets) ===");
    for (corpus, a) in &by_corpus {
        print_summary(corpus, None, a);
    }
    println!("\n=== by (corpus, target) ===");
    for ((corpus, t100), a) in &by_corpus_target {
        print_summary(corpus, Some(*t100 as f32 / 100.0), a);
    }
    println!("\n=== overall ===");
    print_summary("ALL", None, &overall);
}

struct DecodedPng {
    rgb: Vec<u8>,
    rgba: Option<Vec<u8>>,
    w: u32,
    h: u32,
}

fn decode_png(path: &PathBuf) -> Option<DecodedPng> {
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
    let (rgb, rgba) = match color {
        png::ColorType::Rgb | png::ColorType::Indexed => (buf, None),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&[px[0], px[1], px[2]]);
            }
            (rgb, Some(buf))
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf {
                rgb.extend_from_slice(&[g, g, g]);
            }
            (rgb, None)
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            (rgb, None)
        }
    };
    Some(DecodedPng { rgb, rgba, w, h })
}
