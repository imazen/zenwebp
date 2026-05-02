//! Pareto sweep harness for zenwebp picker training.
//!
//! Follows `zenanalyze/zenpicker/FOR_NEW_CODECS.md` Step 1 + Step 2:
//! emits (image, size_class, config, q) → (bytes, zensim, encode_ms,
//! total_ms) plus a sister features TSV with the full zenanalyze feature
//! vector per (image, size_class). Both files feed `train_hybrid.py`.
//!
//! Cell taxonomy (categorical axes, form cells):
//!   - method ∈ {4, 5, 6}    (RD optimization tier)
//!   - segments ∈ {1, 4}     (uniformity gating; 2/3 omitted)
//!
//! Scalar grid (per-cell prediction targets):
//!   - sns_strength ∈ {0, 50, 100}
//!   - filter_strength ∈ {0, 60}
//!   - filter_sharpness ∈ {0, 6}
//!
//! 6 cells × 12 scalar combos = 72 ConfigSpec rows.
//!
//! Q grid: zenpicker standard, 0..70 step 5 + 70..101 step 2 = 30 values.
//!
//! Image size variants (`--sizes 64,256,1024,native`): each source PNG is
//! Lanczos-resampled to {64, 256, 1024, native} on the long edge so the
//! picker sees tiny + small + medium + large content per the project-wide
//! sweep discipline.
//!
//! Naming convention (parser is `examples/zenwebp_picker_config.py`):
//!   m{method}_seg{segments}_sns{sns}_fs{filter_strength}_sh{filter_sharpness}
//! e.g. `m4_seg1_sns0_fs0_sh0`, `m6_seg4_sns100_fs60_sh6`.
//!
//! Usage:
//!   cargo run --release --features analyzer --example zenwebp_pareto -- \
//!     --corpus /path/to/CID22-512/training \
//!     --corpus /path/to/gb82 \
//!     --max-images 200 \
//!     --output benchmarks/zenwebp_pareto_$(date +%F).tsv \
//!     --features-output benchmarks/zenwebp_pareto_features_$(date +%F).tsv
//!
//! Add `--features-only` to skip the encode loop and only re-extract
//! features (cheap; ~1 second / 1000 images).

#![forbid(unsafe_code)]

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use rayon::prelude::*;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
use zensim::{RgbSlice, Zensim, ZensimProfile};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

// ---------------------------------------------------------------------
// ConfigSpec table — full Cartesian product of categorical × scalar axes
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct ConfigSpec {
    name: &'static str,
    id: u16,
    method: u8,
    segments: u8,
    sns: u8,
    filter_strength: u8,
    filter_sharpness: u8,
}

// Q grid: 0..70 step 5 + 70..101 step 2. Matches zenjpeg's training grid.
const Q_GRID: &[u8] = &[
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
    90, 92, 94, 96, 98, 100,
];

// Default size variants when --sizes is not given. 0 = native.
const DEFAULT_SIZES: &[u32] = &[64, 256, 1024, 0];

// 6 cells × 12 scalar combos = 72 configs.
// Generated via build_configs() so adding/removing scalar grid points
// stays in one place.
fn build_configs() -> Vec<ConfigSpec> {
    let methods: [u8; 3] = [4, 5, 6];
    let segments: [u8; 2] = [1, 4];
    let sns_grid: [u8; 3] = [0, 50, 100];
    let filter_grid: [u8; 2] = [0, 60];
    let sharp_grid: [u8; 2] = [0, 6];

    let mut configs = Vec::new();
    let mut id: u16 = 0;
    for &m in &methods {
        for &seg in &segments {
            for &sns in &sns_grid {
                for &fs in &filter_grid {
                    for &sh in &sharp_grid {
                        // Leak the formatted name to get &'static str.
                        let name = format!("m{m}_seg{seg}_sns{sns}_fs{fs}_sh{sh}").leak();
                        configs.push(ConfigSpec {
                            name,
                            id,
                            method: m,
                            segments: seg,
                            sns,
                            filter_strength: fs,
                            filter_sharpness: sh,
                        });
                        id += 1;
                    }
                }
            }
        }
    }
    configs
}

// ---------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------

struct Args {
    corpora: Vec<PathBuf>,
    sizes: Vec<u32>,
    output: PathBuf,
    features_output: PathBuf,
    max_images: usize,
    threads: usize,
    features_only: bool,
}

fn parse_args() -> Args {
    let mut corpora: Vec<PathBuf> = Vec::new();
    let mut sizes: Vec<u32> = Vec::new();
    let mut max_images = 1024;
    let mut threads = 0;
    let mut features_only = false;
    let date = ymd_today();
    let mut output = PathBuf::from(format!("benchmarks/zenwebp_pareto_{date}.tsv"));
    let mut features_output =
        PathBuf::from(format!("benchmarks/zenwebp_pareto_features_{date}.tsv"));

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--corpus" => corpora.push(PathBuf::from(it.next().unwrap())),
            "--sizes" => {
                let s = it.next().unwrap();
                for tok in s.split(',') {
                    if tok == "native" {
                        sizes.push(0);
                    } else {
                        sizes.push(tok.parse().expect("size must be uint or 'native'"));
                    }
                }
            }
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("threads uint"),
            "--features-only" => features_only = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if corpora.is_empty() {
        // Default mixed corpus covering screen + photo + line-art.
        for d in [
            "/home/lilith/work/codec-corpus/CID22/CID22-512/training",
            "/home/lilith/work/codec-corpus/gb82",
            "/home/lilith/work/codec-corpus/gb82-sc",
            "/home/lilith/work/codec-corpus/clic2025/training",
        ] {
            corpora.push(PathBuf::from(d));
        }
    }
    if sizes.is_empty() {
        sizes = DEFAULT_SIZES.to_vec();
    }
    Args {
        corpora,
        sizes,
        output,
        features_output,
        features_only,
        max_images,
        threads,
    }
}

fn ymd_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Howard Hinnant's algorithm: epoch days → calendar date.
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ---------------------------------------------------------------------
// Image loading + resize (zenpng decoder + zenresize Lanczos)
// ---------------------------------------------------------------------

fn load_png_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    use enough::Unstoppable;
    use zenpixels_convert::PixelBufferConvertTypedExt;
    use zenpng::PngDecodeConfig;

    let bytes = std::fs::read(path).ok()?;
    let output = zenpng::decode(&bytes, &PngDecodeConfig::default(), &Unstoppable).ok()?;
    let w = output.info.width;
    let h = output.info.height;

    // Convert to tightly-packed RGB8 regardless of source layout
    // (handles RGBA, grayscale, palette, gray+alpha, 16-bit → 8-bit).
    let rgb_buf = output.pixels.to_rgb8();
    let slice = rgb_buf.as_slice();
    if let Some(contiguous) = slice.as_contiguous_bytes() {
        return Some((contiguous.to_vec(), w, h));
    }
    // Compact row-by-row when stride > width*3 (rare but possible).
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        let row = slice.row(y);
        out.extend_from_slice(&row[..(w as usize) * 3]);
    }
    Some((out, w, h))
}

fn resize_to(rgb: &[u8], w: u32, h: u32, target_max: u32) -> (Vec<u8>, u32, u32) {
    if target_max == 0 || w.max(h) <= target_max {
        return (rgb.to_vec(), w, h);
    }
    let scale = target_max as f32 / w.max(h) as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    let cfg = zenresize::ResizeConfig::builder(w, h, new_w, new_h)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .filter(zenresize::Filter::Mitchell)
        .srgb()
        .build();
    let resized = zenresize::Resizer::new(&cfg).resize(rgb);
    (resized, new_w, new_h)
}

/// Derive size_class from actual pixel count rather than the resize
/// target. This makes the harness correct for corpora where the source
/// PNG is already at the desired size (e.g. the size-dense corpus).
fn size_class_label_from_pixels(w: u32, h: u32) -> &'static str {
    let n = (w as u64) * (h as u64);
    if n < 64 * 64 {
        "tiny"
    } else if n < 256 * 256 {
        "small"
    } else if n < 1024 * 1024 {
        "medium"
    } else {
        "large"
    }
}

// ---------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------

fn full_feature_set() -> FeatureSet {
    FeatureSet::SUPPORTED
}

fn feature_columns() -> Vec<AnalysisFeature> {
    FeatureSet::SUPPORTED.iter().collect()
}

fn feature_value_str(
    analysis: &zenanalyze::feature::AnalysisResults,
    f: AnalysisFeature,
) -> String {
    if let Some(v) = analysis.get_f32(f) {
        format!("{v:.6}")
    } else if let Some(v) = analysis.get(f) {
        match v {
            FeatureValue::F32(x) => format!("{x:.6}"),
            FeatureValue::U32(x) => format!("{x}"),
            FeatureValue::Bool(b) => format!("{}", b as u8),
            _ => String::new(),
        }
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------
// Encode + score
// ---------------------------------------------------------------------

fn build_lossy(spec: ConfigSpec, q: u8) -> LossyConfig {
    LossyConfig::new()
        .with_quality(q as f32)
        .with_method(spec.method)
        .with_segments(spec.segments)
        .with_sns_strength(spec.sns)
        .with_filter_strength(spec.filter_strength)
        .with_filter_sharpness(spec.filter_sharpness)
}

fn encode_decode_score(
    z: &Zensim,
    pre: &zensim::PrecomputedReference,
    rgb: &[u8],
    w: u32,
    h: u32,
    spec: ConfigSpec,
    q: u8,
) -> Option<(usize, f32, f64, f64)> {
    let total_start = Instant::now();
    let cfg = build_lossy(spec, q);
    let req = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h);
    let encode_start = Instant::now();
    let webp = req.encode().ok()?;
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
    let bytes = webp.len();

    let (rgb_dec, w2, h2) = zenwebp::oneshot::decode_rgb(&webp).ok()?;
    if w2 != w || h2 != h {
        return None;
    }
    let n = (w as usize) * (h as usize) * 3;
    let dec_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb_dec[..n]);
    let dec_slice = RgbSlice::new(dec_chunks, w as usize, h as usize);
    let res = z.compute_with_ref(pre, &dec_slice).ok()?;
    let zensim_score = res.score() as f32;
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, zensim_score, encode_ms, total_ms))
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let configs = build_configs();
    eprintln!(
        "[zenwebp_pareto] {} configs (6 cells × 12 scalar combos)",
        configs.len()
    );

    let z = Zensim::new(ZensimProfile::latest());

    let mut paths: Vec<PathBuf> = Vec::new();
    for corpus in &args.corpora {
        let entries = std::fs::read_dir(corpus)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", corpus.display()));
        for entry in entries.filter_map(|r| r.ok()) {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("png") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths.truncate(args.max_images);

    let cells = paths.len() * args.sizes.len() * configs.len() * Q_GRID.len();
    eprintln!(
        "[zenwebp_pareto] {} images × {} sizes × {} configs × {} q values = {} cells",
        paths.len(),
        args.sizes.len(),
        configs.len(),
        Q_GRID.len(),
        cells,
    );
    eprintln!("[zenwebp_pareto] output:    {}", args.output.display());
    eprintln!(
        "[zenwebp_pareto] features:  {}",
        args.features_output.display()
    );
    if args.features_only {
        eprintln!("[zenwebp_pareto] features-only mode — encode loop SKIPPED");
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Some(parent) = args.features_output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Open in append mode; write header if file is new.
    let main_file: Option<Mutex<std::fs::File>> = if args.features_only {
        None
    } else {
        let main_is_new = !args.output.exists();
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output");
        let main_file = Mutex::new(f);
        if main_is_new {
            writeln!(
                main_file.lock().unwrap(),
                "image_path\tsize_class\twidth\theight\tconfig_id\tconfig_name\tq\tbytes\tzensim\tencode_ms\ttotal_ms"
            )
            .ok();
        }
        Some(main_file)
    };

    let feat_is_new = !args.features_output.exists();
    let feat_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.features_output)
        .expect("open features output");
    let feat_file = Mutex::new(feat_file);
    let cols = feature_columns();
    if feat_is_new {
        let mut f = feat_file.lock().unwrap();
        write!(f, "image_path\tsize_class\twidth\theight").ok();
        for c in &cols {
            write!(f, "\tfeat_{}", c.name()).ok();
        }
        writeln!(f).ok();
    }

    let query = AnalysisQuery::new(full_feature_set());
    let started = Instant::now();
    let unit_count = paths.len() * args.sizes.len();
    let done = AtomicUsize::new(0);

    let work_units: Vec<(PathBuf, u32)> = paths
        .iter()
        .flat_map(|path| args.sizes.iter().map(move |&sz| (path.clone(), sz)))
        .collect();

    work_units.par_iter().for_each(|(path, target_size)| {
        let target_size = *target_size;
        let (rgb_native, w_native, h_native) = match load_png_rgb8(path) {
            Some(t) => t,
            None => {
                eprintln!("  load failed: {}", path.display());
                return;
            }
        };
        let (rgb, w, h) = resize_to(&rgb_native, w_native, h_native, target_size);
        let size_class = size_class_label_from_pixels(w, h);

        // Per-image features (analyzed once at this size).
        let analysis = analyze_features_rgb8(&rgb, w, h, &query);
        {
            let mut f = feat_file.lock().unwrap();
            write!(f, "{}\t{}\t{}\t{}", path.display(), size_class, w, h).ok();
            for c in &cols {
                write!(f, "\t{}", feature_value_str(&analysis, *c)).ok();
            }
            writeln!(f).ok();
            f.flush().ok();
        }

        // Skip encode loop in features-only mode.
        if let Some(main_file) = main_file.as_ref() {
            // Pre-compute zensim reference once per (image, size).
            let n = (w as usize) * (h as usize) * 3;
            let src_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb[..n]);
            let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
            let pre = match z.precompute_reference(&src_slice) {
                Ok(p) => p,
                Err(_) => {
                    eprintln!(
                        "  zensim precompute failed: {} ({size_class}, {w}x{h})",
                        path.display()
                    );
                    return;
                }
            };

            for spec in &configs {
                for &q in Q_GRID {
                    let row = encode_decode_score(&z, &pre, &rgb, w, h, *spec, q);
                    let mut f = main_file.lock().unwrap();
                    match row {
                        Some((bytes, z_score, enc_ms, tot_ms)) => {
                            writeln!(
                                f,
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}",
                                path.display(),
                                size_class,
                                w,
                                h,
                                spec.id,
                                spec.name,
                                q,
                                bytes,
                                z_score,
                                enc_ms,
                                tot_ms,
                            )
                            .ok();
                        }
                        None => {
                            // Mark a failed cell with empty fields — keeps
                            // cell index dense so the trainer can detect
                            // failures vs missing rows.
                            writeln!(
                                f,
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t\t\t",
                                path.display(),
                                size_class,
                                w,
                                h,
                                spec.id,
                                spec.name,
                                q,
                            )
                            .ok();
                        }
                    }
                }
                main_file.lock().unwrap().flush().ok();
            }
        }

        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 4 == 0 || n == unit_count {
            let dt = started.elapsed().as_secs_f64();
            let rate = n as f64 / dt;
            let eta = (unit_count - n) as f64 / rate;
            eprintln!(
                "  progress: {}/{}  ({:.1}/sec, ETA {:.0}s)",
                n, unit_count, rate, eta
            );
        }
    });

    let primary = if args.features_only {
        args.features_output.display().to_string()
    } else {
        args.output.display().to_string()
    };
    eprintln!(
        "[zenwebp_pareto] done in {:.1}s. wrote {}",
        started.elapsed().as_secs_f64(),
        primary
    );
}
