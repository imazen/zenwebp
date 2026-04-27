//! A/B comparison: Phase 3 per-segment correction using the 2x2
//! spatial-quadrant proxy vs the encoder's real k-means `segment_map`.
//!
//! Toggles `ZENWEBP_PHASE3_QUADRANT=1` for the A leg (proxy) and unsets
//! it for the B leg (real segment_map). Everything else — starting q,
//! pass-0 encode, score band, max_overshoot, max_passes — is identical.
//!
//! Reads corpus PNG files from the filesystem (CLI args), runs both
//! aggregators at multiple zensim targets, writes a per-image TSV plus
//! a summary table to stdout. Output TSV is intended for block storage
//! at /mnt/v/output/zenwebp/zensim-ab/ — not committed to the repo.
//!
//! Usage:
//!   zensim_ab_quadrant_vs_segmap \
//!     --corpus CID22=/path/to/CID22-512/validation \
//!     --corpus gb82=/path/to/gb82 \
//!     --corpus gb82-sc=/path/to/gb82-sc \
//!     --targets 75,80,85 \
//!     --max-overshoot 1.5 \
//!     --max-passes 3 \
//!     --out /mnt/v/output/zenwebp/zensim-ab/ab_2026-04-26.tsv
//!
//! NOTE: this binary uses `std::env::set_var` to toggle the
//! `ZENWEBP_PHASE3_QUADRANT` env var between A and B legs. That call is
//! `unsafe` in Rust 2024 (POSIX getenv races); this binary is
//! single-threaded so it's sound. Confined to dev/ tooling.

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
    out: Option<PathBuf>,
    limit: Option<usize>,
}

fn parse_args() -> Cli {
    let mut corpora: Vec<(String, PathBuf)> = Vec::new();
    let mut targets: Vec<f32> = vec![80.0];
    let mut max_overshoot = 1.5f32;
    let mut max_passes = 3u8;
    let mut method = 4u8;
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
            "usage: --corpus name=/path [--corpus ...] [--targets 75,80,85] [--out /mnt/v/.../ab.tsv]"
        );
        std::process::exit(2);
    }
    Cli {
        corpora,
        targets,
        max_overshoot,
        max_passes,
        method,
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

#[derive(Default)]
struct Agg {
    n: u32,
    passes_q_sum: u32,
    passes_s_sum: u32,
    dist_q_sum: f64,
    dist_s_sum: f64,
    met_q: u32,
    met_s: u32,
    undershoot_q: u32,
    undershoot_s: u32,
    bytes_q: Vec<u64>,
    bytes_s: Vec<u64>,
    better_seg: u32,
    better_quad: u32,
    tied: u32,
}

fn fold(into: &mut Agg, src: &Agg) {
    into.n += src.n;
    into.passes_q_sum += src.passes_q_sum;
    into.passes_s_sum += src.passes_s_sum;
    into.dist_q_sum += src.dist_q_sum;
    into.dist_s_sum += src.dist_s_sum;
    into.met_q += src.met_q;
    into.met_s += src.met_s;
    into.undershoot_q += src.undershoot_q;
    into.undershoot_s += src.undershoot_s;
    into.bytes_q.extend(src.bytes_q.iter().copied());
    into.bytes_s.extend(src.bytes_s.iter().copied());
    into.better_seg += src.better_seg;
    into.better_quad += src.better_quad;
    into.tied += src.tied;
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

fn main() {
    let cli = parse_args();

    eprintln!(
        "A/B: 2x2 quadrant proxy vs real segment_map (method={} max_overshoot={} max_passes={})",
        cli.method, cli.max_overshoot, cli.max_passes
    );
    eprintln!("targets: {:?}", cli.targets);
    eprintln!();

    let mut tsv: Option<std::io::BufWriter<fs::File>> = match &cli.out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let f = fs::File::create(p).expect("cannot create --out file");
            let mut w = std::io::BufWriter::new(f);
            writeln!(
                w,
                "corpus\timage\twidth\theight\ttarget\tmax_overshoot\tmax_passes\tmethod\t\
                 passes_quad\tachieved_quad\tbytes_quad\ttargets_met_quad\t\
                 passes_seg\tachieved_seg\tbytes_seg\ttargets_met_seg"
            )
            .unwrap();
            Some(w)
        }
        None => None,
    };

    let mut aggs: BTreeMap<(String, u32), Agg> = BTreeMap::new();

    for (corpus_name, corpus_path) in &cli.corpora {
        let mut paths = list_pngs(corpus_path);
        if let Some(lim) = cli.limit {
            paths.truncate(lim);
        }
        eprintln!(
            "corpus {corpus_name} ({} images): {}",
            paths.len(),
            corpus_path.display()
        );

        for path in &paths {
            let (rgb, w, h) = match decode_png_rgb(path) {
                Some(t) => t,
                None => {
                    eprintln!("skip: cannot decode {}", path.display());
                    continue;
                }
            };
            let img_name = path.file_name().unwrap().to_string_lossy().to_string();

            for &target in &cli.targets {
                let r_quad = run(
                    &rgb,
                    w,
                    h,
                    target,
                    cli.max_overshoot,
                    cli.max_passes,
                    cli.method,
                    true,
                );
                let r_seg = run(
                    &rgb,
                    w,
                    h,
                    target,
                    cli.max_overshoot,
                    cli.max_passes,
                    cli.method,
                    false,
                );

                if let Some(out) = tsv.as_mut() {
                    writeln!(
                        out,
                        "{corpus_name}\t{img_name}\t{wid}\t{hgt}\t{tgt:.2}\t{mo:.2}\t{mp}\t{me}\t\
                         {pq}\t{aq:.4}\t{bq}\t{mq}\t{ps}\t{a_s:.4}\t{bs}\t{ms}",
                        wid = w,
                        hgt = h,
                        tgt = target,
                        mo = cli.max_overshoot,
                        mp = cli.max_passes,
                        me = cli.method,
                        pq = r_quad.passes,
                        aq = r_quad.score,
                        bq = r_quad.bytes,
                        mq = u8::from(r_quad.met),
                        ps = r_seg.passes,
                        a_s = r_seg.score,
                        bs = r_seg.bytes,
                        ms = u8::from(r_seg.met),
                    )
                    .unwrap();
                }

                let key = (corpus_name.clone(), (target * 100.0).round() as u32);
                let a = aggs.entry(key).or_default();
                a.n += 1;
                a.passes_q_sum += u32::from(r_quad.passes);
                a.passes_s_sum += u32::from(r_seg.passes);
                let d_q = (r_quad.score - target).abs();
                let d_s = (r_seg.score - target).abs();
                a.dist_q_sum += f64::from(d_q);
                a.dist_s_sum += f64::from(d_s);
                if r_quad.met {
                    a.met_q += 1;
                }
                if r_seg.met {
                    a.met_s += 1;
                }
                if r_quad.score < target {
                    a.undershoot_q += 1;
                }
                if r_seg.score < target {
                    a.undershoot_s += 1;
                }
                a.bytes_q.push(r_quad.bytes as u64);
                a.bytes_s.push(r_seg.bytes as u64);
                if (d_s - d_q).abs() < 0.05 {
                    a.tied += 1;
                } else if d_s < d_q {
                    a.better_seg += 1;
                } else {
                    a.better_quad += 1;
                }
                eprint!(".");
            }
            eprintln!(" {} ({}x{})", img_name, w, h);
        }
    }

    if let Some(mut out) = tsv {
        let _ = out.flush();
    }

    println!();
    println!("=== summary by (corpus, target) ===");
    println!(
        "{:14} {:>6} {:>4} | {:>6} {:>6} | {:>7} {:>7} | {:>7} {:>7} | {:>5} {:>5} | {:>9} {:>9} | win:S/Q/T",
        "corpus",
        "target",
        "n",
        "p_q",
        "p_s",
        "|d|_q",
        "|d|_s",
        "met_q",
        "met_s",
        "und_q",
        "und_s",
        "med_b_q",
        "med_b_s"
    );
    let mut overall = Agg::default();
    let mut by_target: BTreeMap<u32, Agg> = BTreeMap::new();
    let mut by_corpus: BTreeMap<String, Agg> = BTreeMap::new();
    for ((corpus, t100), a) in &aggs {
        let target = *t100 as f32 / 100.0;
        let n = a.n as f32;
        let med_q = median_u64(&a.bytes_q);
        let med_s = median_u64(&a.bytes_s);
        println!(
            "{:14} {:>6.2} {:>4} | {:>6.2} {:>6.2} | {:>7.3} {:>7.3} | {:>3}/{:<3} {:>3}/{:<3} | {:>5} {:>5} | {:>9} {:>9} | {}/{}/{}",
            corpus,
            target,
            a.n,
            a.passes_q_sum as f32 / n,
            a.passes_s_sum as f32 / n,
            a.dist_q_sum / a.n as f64,
            a.dist_s_sum / a.n as f64,
            a.met_q,
            a.n,
            a.met_s,
            a.n,
            a.undershoot_q,
            a.undershoot_s,
            med_q,
            med_s,
            a.better_seg,
            a.better_quad,
            a.tied
        );
        fold(&mut overall, a);
        fold(by_target.entry(*t100).or_default(), a);
        fold(by_corpus.entry(corpus.clone()).or_default(), a);
    }

    println!();
    println!("=== by target (across all corpora) ===");
    for (t100, a) in &by_target {
        print_summary(&format!("target={:.2}", *t100 as f32 / 100.0), a);
    }
    println!();
    println!("=== by corpus (across all targets) ===");
    for (corpus, a) in &by_corpus {
        print_summary(corpus, a);
    }
    println!();
    println!("=== aggregate (all corpora x all targets) ===");
    print_summary("ALL", &overall);
}

fn print_summary(label: &str, a: &Agg) {
    if a.n == 0 {
        return;
    }
    let n = a.n as f32;
    println!(
        "{:18} n={:>4}  pass q={:.2}/s={:.2}  |d| q={:.3}/s={:.3}  met q={}/s={}  und q={}/s={}  med-bytes q={}/s={}  win S/Q/T={}/{}/{}",
        label,
        a.n,
        a.passes_q_sum as f32 / n,
        a.passes_s_sum as f32 / n,
        a.dist_q_sum / a.n as f64,
        a.dist_s_sum / a.n as f64,
        a.met_q,
        a.met_s,
        a.undershoot_q,
        a.undershoot_s,
        median_u64(&a.bytes_q),
        median_u64(&a.bytes_s),
        a.better_seg,
        a.better_quad,
        a.tied
    );
}

struct Outcome {
    score: f32,
    passes: u8,
    bytes: usize,
    met: bool,
}

#[allow(clippy::too_many_arguments)]
fn run(
    rgb: &[u8],
    w: u32,
    h: u32,
    target: f32,
    max_overshoot: f32,
    max_passes: u8,
    method: u8,
    quadrant: bool,
) -> Outcome {
    if quadrant {
        env_set::set("ZENWEBP_PHASE3_QUADRANT", Some("1"));
    } else {
        env_set::set("ZENWEBP_PHASE3_QUADRANT", None);
    }

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
