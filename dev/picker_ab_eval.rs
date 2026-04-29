//! A/B evaluator for the zenpicker spike.
//!
//! Runs the existing target_zensim convergence loop on a corpus
//! using `Preset::Auto`. When built with `--features picker`, the
//! Auto path inside the encoder routes through the baked picker;
//! without, it routes through the bucket table.
//!
//! Compared to `dev/zensim_convergence_eval.rs`, this binary
//! ALWAYS uses `Preset::Auto` (the convergence eval was built for
//! per-method comparisons) and prints a compact bucket-vs-picker
//! summary at the end. Pair two runs for the A/B comparison:
//!
//!   cargo run --release --features "target-zensim analyzer" \
//!     --example picker_ab_eval -- <args>
//!   cargo run --release --features "target-zensim picker" \
//!     --example picker_ab_eval -- <args>
//!
//! Output goes to stdout in TSV format so a third tool can join
//! the two runs by `image_path,target` and report deltas.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, Preset, ZensimTarget};

#[derive(Default)]
struct Cli {
    corpora: Vec<(String, PathBuf)>,
    targets: Vec<f32>,
    out_tsv: Option<PathBuf>,
    method: u8,
    label: String,
}

fn parse_args() -> Cli {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cli = Cli {
        corpora: vec![],
        targets: vec![],
        out_tsv: None,
        method: 4,
        label: "run".into(),
    };
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
            "--out-tsv" => {
                cli.out_tsv = Some(PathBuf::from(next()));
                i += 2;
            }
            "--method" => {
                cli.method = next().parse().unwrap();
                i += 2;
            }
            "--label" => {
                cli.label = next().into();
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if cli.corpora.is_empty() {
        eprintln!(
            "usage: picker_ab_eval --label LBL --corpus name=/path [--corpus ...] --targets 75,80,85 [--out-tsv FILE]"
        );
        std::process::exit(2);
    }
    if cli.targets.is_empty() {
        cli.targets = vec![75.0, 80.0, 85.0];
    }
    cli
}

struct DecodedPng {
    rgb: Vec<u8>,
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
    let rgb = match color {
        png::ColorType::Rgb | png::ColorType::Indexed => buf,
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&[px[0], px[1], px[2]]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            rgb
        }
    };
    Some(DecodedPng { rgb, w, h })
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

fn main() {
    let cli = parse_args();
    let mut images = Vec::new();
    for (name, dir) in &cli.corpora {
        let paths = list_pngs(dir);
        eprintln!("  {name}: {} images from {}", paths.len(), dir.display());
        for p in paths {
            if let Some(d) = decode_png(&p) {
                images.push((name.clone(), p, d));
            }
        }
    }
    eprintln!("loaded {} images", images.len());

    let mut tsv_writer: Option<fs::File> = cli.out_tsv.as_ref().map(|p| {
        let mut f = fs::File::create(p).expect("open --out-tsv");
        writeln!(
            f,
            "label\timage\tcorpus\twidth\theight\ttarget\tbytes\tachieved\tpasses\tmet\tencode_ms"
        )
        .unwrap();
        f
    });

    let t0 = Instant::now();
    let mut total_passes: u64 = 0;
    let mut total_n: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut total_met: u64 = 0;
    let mut sum_achieved: f64 = 0.0;

    for (corpus, path, d) in &images {
        for &target in &cli.targets {
            let cfg = LossyConfig::new()
                .with_method(cli.method)
                .with_preset_value(Preset::Auto)
                .with_target_zensim(
                    ZensimTarget::new(target)
                        .with_max_overshoot(Some(1.5))
                        .with_max_passes(2),
                );
            let t_one = Instant::now();
            let r = EncodeRequest::lossy(&cfg, &d.rgb, PixelLayout::Rgb8, d.w, d.h)
                .encode_with_metrics();
            let elapsed_ms = t_one.elapsed().as_secs_f64() * 1000.0;
            match r {
                Ok((bytes, m)) => {
                    total_n += 1;
                    total_passes += u64::from(m.passes_used);
                    total_bytes += bytes.len() as u64;
                    if m.targets_met {
                        total_met += 1;
                    }
                    sum_achieved += f64::from(m.achieved_score);
                    if let Some(f) = tsv_writer.as_mut() {
                        writeln!(
                            f,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}",
                            cli.label,
                            path.to_string_lossy(),
                            corpus,
                            d.w,
                            d.h,
                            target,
                            bytes.len(),
                            m.achieved_score,
                            m.passes_used,
                            m.targets_met as u8,
                            elapsed_ms,
                        )
                        .unwrap();
                    }
                }
                Err(e) => eprintln!("ERR {} t={}: {:?}", path.display(), target, e),
            }
        }
    }
    if let Some(mut f) = tsv_writer {
        f.flush().unwrap();
    }

    eprintln!(
        "\nlabel={}  n={}  passes_avg={:.2}  achieved_avg={:.2}  met={}/{} ({:.0}%)  total_bytes={}  wall={:.1}s",
        cli.label,
        total_n,
        total_passes as f64 / total_n as f64,
        sum_achieved / total_n as f64,
        total_met,
        total_n,
        100.0 * total_met as f64 / total_n as f64,
        total_bytes,
        t0.elapsed().as_secs_f64(),
    );
    println!(
        "{}\t{}\t{:.4}\t{:.4}\t{}\t{}",
        cli.label,
        total_n,
        total_passes as f64 / total_n as f64,
        sum_achieved / total_n as f64,
        total_met,
        total_bytes,
    );
}
