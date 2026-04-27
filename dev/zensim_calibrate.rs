//! Per-bucket starting-q calibration for `target_zensim`.
//!
//! Sweeps a corpus at a small ladder of VP8 quality values, decodes each
//! result, measures zensim vs the source, and per (content-bucket,
//! target-zensim) cell emits the median quality value that meets the
//! target. The output is an anchor table (TSV + Rust const block) ready
//! to paste into `src/encoder/zensim_target.rs`.
//!
//! Usage:
//!   cargo run --release --features target-zensim --example zensim_calibrate -- \
//!       <corpus_dir> [output_dir]
//!
//! Defaults: corpus=`~/work/codec-corpus/CID22/CID22-512/validation`,
//! output=`/mnt/v/output/zenwebp/zensim-calibrate`.
//!
//! Methodology:
//!   1. For each PNG: decode RGB, run zenwebp's `Preset::Auto` content
//!      classifier to determine the bucket (Photo / Drawing / Icon).
//!   2. Encode at q ∈ {30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94,
//!      96, 98, 100} via the standard non-target-zensim path. Decode
//!      back to RGB, measure zensim vs source.
//!   3. For each target ∈ {60, 70, 75, 80, 85, 90, 95}, find the
//!      smallest q whose decoded zensim meets target. That's the
//!      "fitted q" for this image at this target.
//!   4. Median-aggregate fitted-q values per bucket → anchor table.
//!   5. Print as Rust const blocks ready to paste.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use zenwebp::encoder::analysis::{ImageContentType, classify_image_type};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

const Q_LADDER: &[u8] = &[
    30, 40, 50, 60, 65, 70, 75, 80, 85, 88, 90, 92, 94, 96, 98, 100,
];
const TARGETS: &[f32] = &[60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0];

fn main() {
    let corpus_dir = env::args().nth(1).unwrap_or_else(|| {
        // Default corpus path.
        let home = env::var("HOME").expect("HOME not set");
        format!("{home}/work/codec-corpus/CID22/CID22-512/validation")
    });
    let output_dir = env::args()
        .nth(2)
        .unwrap_or_else(|| "/mnt/v/output/zenwebp/zensim-calibrate".to_string());

    fs::create_dir_all(&output_dir).expect("create output dir");
    eprintln!("corpus: {corpus_dir}");
    eprintln!("output: {output_dir}");

    let mut paths: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read corpus dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().is_some_and(|e| e == "png" || e == "PNG")).then_some(p)
        })
        .collect();
    paths.sort();

    // Cap at 20 images for the fit (issue #47 spec) — deterministic stride.
    let max_images = 20;
    if paths.len() > max_images {
        let stride = paths.len() / max_images;
        paths = paths
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .take(max_images)
            .map(|(_, p)| p.clone())
            .collect();
    }
    eprintln!("processing {} images", paths.len());

    let z = zensim::Zensim::new(zensim::ZensimProfile::latest());

    // (bucket, target) → Vec<fitted_q>
    let mut fits: BTreeMap<(BucketKey, OrderedF32), Vec<f32>> = BTreeMap::new();

    let tsv_path = PathBuf::from(&output_dir).join("zensim_calibrate_raw.tsv");
    let mut tsv = fs::File::create(&tsv_path).expect("create tsv");
    use std::io::Write;
    writeln!(tsv, "file\tw\th\tbucket\tq\tbytes\tscore").unwrap();

    for (i, path) in paths.iter().enumerate() {
        eprintln!(
            "[{}/{}] {}",
            i + 1,
            paths.len(),
            path.file_name().unwrap().to_string_lossy()
        );
        let (rgb, w, h) = match decode_png_rgb(path) {
            Some(t) => t,
            None => continue,
        };
        if w < 32 || h < 32 {
            continue;
        }
        let bucket = classify_via_y(&rgb, w, h);

        let pre = match build_pre(&z, &rgb, w, h) {
            Some(p) => p,
            None => continue,
        };

        // Per-q encode + measure.
        let mut probes: Vec<(u8, f32)> = Vec::with_capacity(Q_LADDER.len());
        for &q in Q_LADDER {
            let cfg = LossyConfig::new().with_quality(q as f32).with_method(4);
            let req = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h);
            let webp = match req.encode() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let (rgb_dec, w2, h2) = match zenwebp::oneshot::decode_rgb(&webp) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if w2 != w || h2 != h {
                continue;
            }
            let n = (w * h * 3) as usize;
            let dec_chunks: Vec<[u8; 3]> = rgb_dec[..n]
                .chunks_exact(3)
                .map(|p| [p[0], p[1], p[2]])
                .collect();
            let dec_slice = zensim::RgbSlice::new(&dec_chunks, w as usize, h as usize);
            let res = match z.compute_with_ref(&pre, &dec_slice) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let score = res.score() as f32;
            probes.push((q, score));
            writeln!(
                tsv,
                "{}\t{}\t{}\t{:?}\t{}\t{}\t{:.3}",
                path.file_name().unwrap().to_string_lossy(),
                w,
                h,
                bucket,
                q,
                webp.len(),
                score,
            )
            .unwrap();
        }
        // For each target, find the smallest q whose score >= target.
        for &t in TARGETS {
            let fitted = probes.iter().find(|(_, s)| *s >= t).map(|(q, _)| *q as f32);
            if let Some(q) = fitted {
                fits.entry((BucketKey::from(bucket), OrderedF32(t)))
                    .or_default()
                    .push(q);
            } else {
                // Target unreachable — record as 100 (the q ceiling) so the
                // anchor still maps to a sensible starting point.
                fits.entry((BucketKey::from(bucket), OrderedF32(t)))
                    .or_default()
                    .push(100.0);
            }
        }
    }

    drop(tsv);

    // Aggregate: per (bucket, target) → median, p25, p75, n.
    eprintln!();
    eprintln!("================================================================");
    eprintln!("Per-bucket anchor table (median fitted-q per target_zensim)");
    eprintln!("================================================================");
    let summary_path = PathBuf::from(&output_dir).join("zensim_calibrate_summary.tsv");
    let mut sum = fs::File::create(&summary_path).expect("create summary");
    writeln!(sum, "bucket\ttarget\tmedian_q\tp25_q\tp75_q\tn").unwrap();

    let mut by_bucket: BTreeMap<BucketKey, Vec<(f32, f32)>> = BTreeMap::new();
    for ((b, t), qs) in &fits {
        let med = median(qs.clone());
        let p25 = percentile(qs.clone(), 0.25);
        let p75 = percentile(qs.clone(), 0.75);
        eprintln!(
            "  {:<10} target={:>5.1}  median_q={:>5.1}  p25={:>5.1}  p75={:>5.1}  n={}",
            format!("{:?}", b),
            t.0,
            med,
            p25,
            p75,
            qs.len()
        );
        writeln!(
            sum,
            "{:?}\t{}\t{}\t{}\t{}\t{}",
            b,
            t.0,
            med,
            p25,
            p75,
            qs.len()
        )
        .unwrap();
        by_bucket.entry(*b).or_default().push((t.0, med));
    }
    drop(sum);

    eprintln!();
    eprintln!("================================================================");
    eprintln!("Rust const block (paste into zensim_target.rs):");
    eprintln!("================================================================");
    for (bucket, mut anchors) in by_bucket {
        anchors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        eprintln!("    const {}: &[(f32, f32)] = &[", bucket.const_name());
        for (t, q) in anchors {
            eprintln!("        ({:.1}, {:.1}),", t, q);
        }
        eprintln!("    ];");
    }
    eprintln!();
    eprintln!("Raw  TSV: {}", tsv_path.display());
    eprintln!("Summary : {}", summary_path.display());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BucketKey {
    Photo,
    Drawing,
    Icon,
}

impl BucketKey {
    fn from(c: ImageContentType) -> Self {
        match c {
            ImageContentType::Photo => Self::Photo,
            ImageContentType::Icon => Self::Icon,
            // Drawing, Text, and any future variants → Drawing.
            _ => Self::Drawing,
        }
    }
    fn const_name(self) -> &'static str {
        match self {
            Self::Photo => "PHOTO",
            Self::Drawing => "DRAWING",
            Self::Icon => "ICON",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OrderedF32(f32);
impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
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

fn classify_via_y(rgb: &[u8], w: u32, h: u32) -> ImageContentType {
    let w = w as usize;
    let h = h as usize;
    let mut y_plane: Vec<u8> = Vec::with_capacity(w * h);
    let mut hist = [0u32; 256];
    for px in rgb.chunks_exact(3) {
        let y =
            ((u32::from(px[0]) * 76 + u32::from(px[1]) * 150 + u32::from(px[2]) * 30) >> 8) as u8;
        y_plane.push(y);
        hist[y as usize] += 1;
    }
    classify_image_type(&y_plane, w, h, w, &hist)
}

fn build_pre(
    z: &zensim::Zensim,
    rgb: &[u8],
    w: u32,
    h: u32,
) -> Option<zensim::PrecomputedReference> {
    let chunks: Vec<[u8; 3]> = rgb[..(w as usize * h as usize * 3)]
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let slice = zensim::RgbSlice::new(&chunks, w as usize, h as usize);
    z.precompute_reference(&slice).ok()
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    }
}

fn percentile(mut v: Vec<f32>, p: f32) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return 0.0;
    }
    let idx = ((v.len() as f32 - 1.0) * p).round() as usize;
    v[idx.min(v.len() - 1)]
}
