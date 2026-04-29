//! Picker spike sweep harness — encode each image at every cell in the
//! picker grid for each target zensim, dump TSVs of (bytes, achieved
//! zensim, passes, encode_ms) and per-image zenanalyze features. The
//! Python distillation step in `scripts/zenwebp_picker_distill.py`
//! consumes both files.
//!
//! Usage:
//!   cargo run --release --features "picker target-zensim analyzer" \
//!     --example zenwebp_picker_sweep -- \
//!     --corpus CID22=/home/lilith/work/codec-corpus/CID22/CID22-512/validation \
//!     --corpus gb82=/home/lilith/work/codec-corpus/gb82 \
//!     --corpus gb82-sc=/home/lilith/work/codec-corpus/gb82-sc \
//!     --targets 75,80,85 \
//!     --limit 50 \
//!     --out-dir /mnt/v/output/zenwebp/picker-sweep
//!
//! Outputs (under --out-dir):
//!   pareto_<UTC>.tsv     per (image, cell, target) cell
//!   features_<UTC>.tsv   per image: zenanalyze feature vector + WxH

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use zenwebp::encoder::picker::spec::{
    CONFIGS, FEAT_COLS, FIXED_METHOD, N_CELLS, SCHEMA_VERSION_TAG,
};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

#[derive(Default)]
struct Cli {
    corpora: Vec<(String, PathBuf)>,
    targets: Vec<f32>,
    limit: Option<usize>,
    out_dir: PathBuf,
    method: u8,
}

fn parse_args() -> Cli {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cli = Cli {
        corpora: vec![],
        targets: vec![],
        limit: None,
        out_dir: PathBuf::from("/mnt/v/output/zenwebp/picker-sweep"),
        method: FIXED_METHOD,
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
            "--limit" => {
                cli.limit = Some(next().parse().unwrap());
                i += 2;
            }
            "--out-dir" => {
                cli.out_dir = PathBuf::from(next());
                i += 2;
            }
            "--method" => {
                cli.method = next().parse().unwrap();
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    if cli.corpora.is_empty() {
        eprintln!(
            "usage: zenwebp_picker_sweep --corpus name=/path [--corpus ...] --targets 75,80,85 [--limit 50] [--out-dir DIR] [--method 4]"
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

/// Extract the 14-feature zenanalyze vector that `FEAT_COLS` declares.
/// Returned in the *exact* order `FEAT_COLS` lists, so the picker
/// schema_hash stays stable.
fn extract_features(rgb: &[u8], w: u32, h: u32) -> Option<Vec<f32>> {
    use zenwebp::encoder::analysis::classifier::{ZenanalyzeDiag, classify_image_type_rgb8_diag};
    if rgb.len() != (w as usize) * (h as usize) * 3 {
        return None;
    }
    let (_bucket, diag): (_, ZenanalyzeDiag) = classify_image_type_rgb8_diag(rgb, w, h);
    Some(vec![
        diag.screen_content,
        diag.text_likelihood,
        diag.natural_likelihood,
        diag.flat_color_block_ratio,
        diag.distinct_color_bins as f32,
        diag.variance,
        diag.edge_density,
        diag.uniformity,
        diag.high_freq_energy_ratio,
        if diag.palette_fits_in_256 { 1.0 } else { 0.0 },
        diag.indexed_palette_width as f32,
        diag.line_art_score,
        diag.skin_tone_fraction,
        diag.edge_slope_stdev,
    ])
}

fn timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Just YYYYMMDD-HHMMSS-ish without external deps.
    format!("{secs}")
}

fn main() {
    let cli = parse_args();
    fs::create_dir_all(&cli.out_dir).expect("create out-dir");
    eprintln!(
        "picker-sweep: schema={} method={} cells={} targets={:?}",
        SCHEMA_VERSION_TAG, cli.method, N_CELLS, cli.targets
    );
    eprintln!("out-dir: {}", cli.out_dir.display());
    eprintln!("FEAT_COLS ({}): {:?}", FEAT_COLS.len(), FEAT_COLS);

    // Load images with their corpus tag.
    struct Img {
        corpus: String,
        path: PathBuf,
        rgb: Vec<u8>,
        w: u32,
        h: u32,
    }
    let mut images: Vec<Img> = Vec::new();
    for (corpus, dir) in &cli.corpora {
        let mut paths = list_pngs(dir);
        if let Some(lim) = cli.limit {
            paths.truncate(lim);
        }
        eprintln!("  {corpus}: {} images from {}", paths.len(), dir.display());
        for p in paths {
            match decode_png(&p) {
                Some(d) => images.push(Img {
                    corpus: corpus.clone(),
                    path: p,
                    rgb: d.rgb,
                    w: d.w,
                    h: d.h,
                }),
                None => eprintln!("    skip {}", p.display()),
            }
        }
    }
    eprintln!("loaded {} images", images.len());

    let stamp = timestamp();
    let pareto_path = cli.out_dir.join(format!("pareto_{stamp}.tsv"));
    let features_path = cli.out_dir.join(format!("features_{stamp}.tsv"));

    // Features TSV header.
    let mut features_f = fs::File::create(&features_path).unwrap();
    write!(
        &mut features_f,
        "image_path\tcorpus\tsize_class\twidth\theight"
    )
    .unwrap();
    for c in FEAT_COLS {
        write!(&mut features_f, "\t{c}").unwrap();
    }
    writeln!(&mut features_f).unwrap();

    // Pareto TSV header.
    let mut pareto_f = fs::File::create(&pareto_path).unwrap();
    writeln!(
        &mut pareto_f,
        "image_path\tcorpus\tsize_class\twidth\theight\tcell_idx\tcell_name\tsns\tfilter\tsharpness\tsegments\tmethod\ttarget_zensim\tbytes\tachieved_zensim\tpasses_used\ttargets_met\tencode_ms"
    )
    .unwrap();

    let total_encodes = images.len() * N_CELLS * cli.targets.len();
    eprintln!(
        "estimated encodes: {} ({} images × {} cells × {} targets)",
        total_encodes,
        images.len(),
        N_CELLS,
        cli.targets.len()
    );

    let mut done = 0usize;
    let bench_start = Instant::now();
    for img in &images {
        // size_class buckets matching zenjpeg's distill convention.
        let pixels = (img.w as usize) * (img.h as usize);
        let size_class = if pixels < 64 * 64 {
            "tiny"
        } else if pixels < 256 * 256 {
            "small"
        } else if pixels < 1024 * 1024 {
            "medium"
        } else {
            "large"
        };
        let img_path_s = img.path.to_string_lossy();

        // Features: extracted ONCE per image, regardless of cell/target.
        let feats = match extract_features(&img.rgb, img.w, img.h) {
            Some(f) => f,
            None => {
                eprintln!("feature extract failed: {}", img_path_s);
                continue;
            }
        };
        write!(
            &mut features_f,
            "{}\t{}\t{}\t{}\t{}",
            img_path_s, img.corpus, size_class, img.w, img.h
        )
        .unwrap();
        for v in &feats {
            write!(&mut features_f, "\t{v}").unwrap();
        }
        writeln!(&mut features_f).unwrap();
        features_f.flush().unwrap();

        for (cell_idx, cell) in CONFIGS.iter().enumerate() {
            for &target_z in &cli.targets {
                let cfg = LossyConfig::new()
                    .with_method(cli.method)
                    .with_sns_strength(cell.sns_strength)
                    .with_filter_strength(cell.filter_strength)
                    .with_filter_sharpness(cell.filter_sharpness)
                    .with_segments(cell.num_segments)
                    .with_target_zensim(
                        ZensimTarget::new(target_z)
                            .with_max_overshoot(Some(1.5))
                            .with_max_passes(2),
                    );
                let t0 = Instant::now();
                let r =
                    EncodeRequest::lossy(&cfg, &img.rgb, PixelLayout::Rgb8, img.w, img.h)
                        .encode_with_metrics();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                match r {
                    Ok((bytes, m)) => {
                        writeln!(
                            &mut pareto_f,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}",
                            img_path_s,
                            img.corpus,
                            size_class,
                            img.w,
                            img.h,
                            cell_idx,
                            cell.name(),
                            cell.sns_strength,
                            cell.filter_strength,
                            cell.filter_sharpness,
                            cell.num_segments,
                            cli.method,
                            target_z,
                            bytes.len(),
                            m.achieved_score,
                            m.passes_used,
                            m.targets_met as u8,
                            elapsed_ms,
                        )
                        .unwrap();
                    }
                    Err(e) => {
                        eprintln!(
                            "ERROR {} cell={} target={}: {:?}",
                            img_path_s, cell_idx, target_z, e
                        );
                    }
                }
                done += 1;
            }
        }
        pareto_f.flush().unwrap();
        let elapsed = bench_start.elapsed().as_secs_f64();
        eprintln!(
            "  [{}/{}] {} ({:.1} encodes/s, ETA {:.0}s)",
            done,
            total_encodes,
            img.path.file_name().unwrap().to_string_lossy(),
            done as f64 / elapsed.max(1e-3),
            (total_encodes - done) as f64 / (done as f64 / elapsed.max(1e-3)).max(1e-3),
        );
    }

    eprintln!("\nDone. wrote:");
    eprintln!("  pareto:   {}", pareto_path.display());
    eprintln!("  features: {}", features_path.display());
}
