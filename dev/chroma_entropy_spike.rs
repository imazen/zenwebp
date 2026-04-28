//! ChromaEntropy spike — research prototype.
//!
//! Hypothesis: artwork uses a chosen palette (a few prominent hue/sat
//! clusters with most of the pixels concentrated in them); photographs
//! spread pixels across the chroma gamut more uniformly. Distinct-bin
//! count (the prior spike, AUC ~0.73) failed because rich-palette art
//! blew the count to the photo-distribution high end. Shannon entropy
//! over the chroma histogram should pick up *concentration*, not just
//! *richness*: concentrated distributions have low entropy, spread
//! distributions have high entropy.
//!
//! For each labelled image we compute six metrics on the chroma plane:
//!
//!   * `chroma_entropy_ycbcr_32` — Shannon entropy of a 32x32 BT.601 (Cb,Cr) histogram (bits)
//!   * `chroma_entropy_ycbcr_64` — same at 64x64 bins
//!   * `chroma_entropy_lab_32`   — same in CIELAB a*/b* at 32x32
//!   * `chroma_entropy_normalized` — entropy / log2(non_empty_bins) (1.0 = uniform-on-occupied; 0 = Dirac)
//!   * `chroma_top1_fraction`    — fraction of pixels in the single most-populated bin
//!   * `chroma_top10_fraction`   — fraction in the 10 most-populated bins
//!
//! Output: per-image TSV + summary.md with per-bucket distributions and
//! AUC (Photo positive). See zenjpeg#123 spike thread.
//!
//! Usage (from repo root):
//!   cargo run --release --example chroma_entropy_spike -- \
//!     /home/lilith/work/coefficient/benchmarks/classifier-eval/labels.tsv

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Truth {
    Photo,
    Drawing,
    Skip,
}

fn map_gpt_category(s: &str) -> Truth {
    match s {
        "photo_natural" | "photo_detailed" | "photo_portrait" | "photo_uniform" => Truth::Photo,
        "screen_ui" | "screen_document" | "screen_chart" | "illustration" => Truth::Drawing,
        _ => Truth::Skip,
    }
}

fn corpus_root(corpus: &str) -> Option<&'static str> {
    match corpus {
        "cid22-train" => Some("/home/lilith/work/codec-corpus/CID22/CID22-512/training"),
        "cid22-val" => Some("/home/lilith/work/codec-corpus/CID22/CID22-512/validation"),
        "clic2025-1024" => Some("/home/lilith/work/codec-corpus/clic2025-1024"),
        "gb82" => Some("/home/lilith/work/codec-corpus/gb82"),
        "gb82-sc" => Some("/home/lilith/work/codec-corpus/gb82-sc"),
        "imageflow" => Some("/home/lilith/work/codec-corpus/imageflow/test_inputs"),
        "kadid10k" => Some("/home/lilith/work/codec-corpus/kadid10k"),
        "qoi-benchmark" => Some("/home/lilith/work/codec-corpus/qoi-benchmark/screenshot_web"),
        _ => None,
    }
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let w = info.width;
    let h = info.height;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        _ => return None,
    };
    Some((rgb, w, h))
}

fn load_image(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png") => load_png(path),
        _ => None,
    }
}

/// BT.601 RGB->YCbCr (gamma-encoded, full-range Cb/Cr in [0,255]).
#[inline]
fn rgb_to_cbcr_bt601(r: u8, g: u8, b: u8) -> (u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let cb = (-43 * r - 85 * g + 128 * b + 128 * 256 + 128) >> 8;
    let cr = (128 * r - 107 * g - 21 * b + 128 * 256 + 128) >> 8;
    let cb = cb.clamp(0, 255) as u8;
    let cr = cr.clamp(0, 255) as u8;
    (cb, cr)
}

fn srgb_to_linear_lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        t[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }
    t
}

/// Linear sRGB -> CIELAB (D65). Returns (a*, b*) clamped to u8.
#[inline]
fn linear_rgb_to_ab_u8(r: f32, g: f32, b: f32) -> (u8, u8) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let xn = x / 0.95047;
    let yn = y;
    let zn = z / 1.08883;
    fn f(t: f32) -> f32 {
        const D: f32 = 6.0 / 29.0;
        if t > D * D * D {
            t.cbrt()
        } else {
            t / (3.0 * D * D) + 4.0 / 29.0
        }
    }
    let fx = f(xn);
    let fy = f(yn);
    let fz = f(zn);
    let a = 500.0 * (fx - fy);
    let b_ = 200.0 * (fy - fz);
    let a_u = ((a + 128.0).clamp(0.0, 255.999)) as u8;
    let b_u = ((b_ + 128.0).clamp(0.0, 255.999)) as u8;
    (a_u, b_u)
}

#[derive(Default, Clone, Debug)]
struct EntropyMetrics {
    entropy_y32: f64,
    entropy_y64: f64,
    entropy_l32: f64,
    entropy_normalized: f64, // entropy_y32 / log2(non_empty_bins_y32)
    top1_fraction: f64,      // computed on y32 histogram
    top10_fraction: f64,     // computed on y32 histogram
    distinct_bins_y32: u32,
    total_pixels: u32,
}

fn shannon_entropy(counts: &[u32]) -> f64 {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut h = 0.0f64;
    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total_f;
        h -= p * p.log2();
    }
    h
}

fn compute_metrics(rgb: &[u8]) -> EntropyMetrics {
    let lut = srgb_to_linear_lut();
    let mut hist_y32 = vec![0u32; 32 * 32];
    let mut hist_y64 = vec![0u32; 64 * 64];
    let mut hist_l32 = vec![0u32; 32 * 32];

    let mut total_pixels: u32 = 0;
    for px in rgb.chunks_exact(3) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        let (cb, cr) = rgb_to_cbcr_bt601(r, g, b);
        let i32y = (cb >> 3) as usize * 32 + (cr >> 3) as usize;
        let i64y = (cb >> 2) as usize * 64 + (cr >> 2) as usize;
        hist_y32[i32y] = hist_y32[i32y].saturating_add(1);
        hist_y64[i64y] = hist_y64[i64y].saturating_add(1);

        let rl = lut[r as usize];
        let gl = lut[g as usize];
        let bl = lut[b as usize];
        let (a, b_) = linear_rgb_to_ab_u8(rl, gl, bl);
        let i32l = (a >> 3) as usize * 32 + (b_ >> 3) as usize;
        hist_l32[i32l] = hist_l32[i32l].saturating_add(1);

        total_pixels = total_pixels.saturating_add(1);
    }

    let entropy_y32 = shannon_entropy(&hist_y32);
    let entropy_y64 = shannon_entropy(&hist_y64);
    let entropy_l32 = shannon_entropy(&hist_l32);

    let distinct_y32 = hist_y32.iter().filter(|&&c| c > 0).count() as u32;
    let entropy_normalized = if distinct_y32 > 1 {
        entropy_y32 / (distinct_y32 as f64).log2()
    } else {
        0.0
    };

    // top1 / top10 on the y32 histogram.
    let mut sorted = hist_y32.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let total_f = total_pixels.max(1) as f64;
    let top1 = sorted[0] as f64 / total_f;
    let top10: u64 = sorted.iter().take(10).map(|&c| c as u64).sum();
    let top10_f = top10 as f64 / total_f;

    EntropyMetrics {
        entropy_y32,
        entropy_y64,
        entropy_l32,
        entropy_normalized,
        top1_fraction: top1,
        top10_fraction: top10_f,
        distinct_bins_y32: distinct_y32,
        total_pixels,
    }
}

#[derive(Clone, Debug)]
struct Row {
    corpus: String,
    image: String,
    width: u32,
    height: u32,
    truth_bucket: Truth,
    primary_category: String,
    metrics: EntropyMetrics,
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let r = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[r.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn median_sorted(sorted: &[f64]) -> f64 {
    percentile_sorted(sorted, 0.5)
}

/// Mann-Whitney U based AUC, sweeping for best accuracy threshold.
/// `direction = +1` means "higher value -> Photo" (e.g. entropy).
/// `direction = -1` means "lower value -> Photo" (e.g. top1_fraction).
fn auc_and_best_threshold(
    values: &[(f64, Truth)],
    direction: i32,
) -> (f64, f64, f64) {
    let mut pos: Vec<f64> = values
        .iter()
        .filter(|(_, t)| *t == Truth::Photo)
        .map(|(v, _)| *v)
        .collect();
    let mut neg: Vec<f64> = values
        .iter()
        .filter(|(_, t)| *t == Truth::Drawing)
        .map(|(v, _)| *v)
        .collect();
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    neg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let np = pos.len() as f64;
    let nn = neg.len() as f64;
    if np == 0.0 || nn == 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut all: Vec<(f64, u8)> = Vec::with_capacity(pos.len() + neg.len());
    for &v in &pos {
        all.push((v, 1));
    }
    for &v in &neg {
        all.push((v, 0));
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut sum_rank_pos = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            if all[k].1 == 1 {
                sum_rank_pos += avg_rank;
            }
        }
        i = j;
    }
    let u = sum_rank_pos - np * (np + 1.0) / 2.0;
    let auc_high_is_photo = u / (np * nn);
    let auc = if direction >= 0 {
        auc_high_is_photo
    } else {
        1.0 - auc_high_is_photo
    };

    let mut candidates: BTreeSet<u64> = BTreeSet::new();
    for (v, _) in values {
        candidates.insert(v.to_bits());
    }
    let mut best_acc = 0.0;
    let mut best_thr = f64::NAN;
    let total = values.len() as f64;
    for bits in &candidates {
        let thr = f64::from_bits(*bits);
        let mut correct = 0.0;
        for (v, t) in values {
            let pred = if direction >= 0 {
                if *v > thr {
                    Truth::Photo
                } else {
                    Truth::Drawing
                }
            } else {
                if *v < thr {
                    Truth::Photo
                } else {
                    Truth::Drawing
                }
            };
            if pred == *t {
                correct += 1.0;
            }
        }
        let acc = correct / total;
        if acc > best_acc {
            best_acc = acc;
            best_thr = thr;
        }
    }
    (auc, best_thr, best_acc)
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        return f64::NAN;
    }
    num / (dx2.sqrt() * dy2.sqrt())
}

/// Combined linear classifier: predict Photo if (a * x + b * y) > thr.
/// Sweeps a wide grid of (a,b) on unit circle plus thr. Returns
/// (best_auc, best_acc, a, b). Used to test orthogonality of two metrics.
fn best_linear_combo(
    xs: &[f64],
    ys: &[f64],
    truths: &[Truth],
) -> (f64, f64, f64, f64) {
    // Z-score normalize first so the (a,b) sweep is meaningful regardless of scale.
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sx = (xs.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / n).sqrt().max(1e-12);
    let sy = (ys.iter().map(|v| (v - my).powi(2)).sum::<f64>() / n).sqrt().max(1e-12);
    let zx: Vec<f64> = xs.iter().map(|v| (v - mx) / sx).collect();
    let zy: Vec<f64> = ys.iter().map(|v| (v - my) / sy).collect();

    let mut best_auc = 0.0;
    let mut best_acc = 0.0;
    let mut best_a = 0.0;
    let mut best_b = 0.0;
    let n_angles = 36;
    for k in 0..n_angles {
        let theta = std::f64::consts::PI * (k as f64) / (n_angles as f64);
        let a = theta.cos();
        let b = theta.sin();
        let combo: Vec<(f64, Truth)> = (0..xs.len())
            .map(|i| (a * zx[i] + b * zy[i], truths[i]))
            .collect();
        let (auc, _, acc) = auc_and_best_threshold(&combo, 1);
        // Try both directions (the angle sweep covers half the unit circle).
        let auc_eff = auc.max(1.0 - auc);
        if auc_eff > best_auc {
            best_auc = auc_eff;
            best_acc = acc.max(1.0 - acc);
            best_a = a;
            best_b = b;
        }
    }
    (best_auc, best_acc, best_a, best_b)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let labels_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/lilith/work/coefficient/benchmarks/classifier-eval/labels.tsv")
        });

    let f = fs::File::open(&labels_path).expect("open labels.tsv");
    let r = BufReader::new(f);

    let out_dir = PathBuf::from("/mnt/v/output/zenwebp/chroma-entropy-spike");
    fs::create_dir_all(&out_dir).expect("mkdir output");
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let tsv_path = out_dir.join(format!("run_{}.tsv", ts));
    let summary_path = out_dir.join("summary.md");

    let mut tsv = BufWriter::new(fs::File::create(&tsv_path).expect("create tsv"));
    writeln!(
        tsv,
        "corpus\timage\twidth\theight\ttruth_bucket\tprimary_category\t\
         chroma_entropy_ycbcr_32\tchroma_entropy_ycbcr_64\tchroma_entropy_lab_32\t\
         chroma_entropy_normalized\tchroma_top1_fraction\tchroma_top10_fraction\t\
         distinct_bins_y32\ttotal_pixels"
    )
    .unwrap();

    let mut rows: Vec<Row> = Vec::new();
    let mut lines = r.lines();
    let header = lines.next().expect("header").expect("header read");
    let cols: Vec<&str> = header.split('\t').collect();
    let idx = |name: &str| cols.iter().position(|c| *c == name).expect(name);
    let i_corpus = idx("corpus");
    let i_image = idx("image");
    let i_w = idx("width");
    let i_h = idx("height");
    let i_cat = idx("primary_category");

    let mut total = 0;
    let mut skipped_unmapped = 0;
    let mut skipped_missing = 0;
    let mut skipped_load = 0;
    for line in lines {
        let line = line.expect("read line");
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < cols.len() {
            continue;
        }
        total += 1;
        let corpus = f[i_corpus].to_string();
        let image = f[i_image].to_string();
        let _width: u32 = f[i_w].parse().unwrap_or(0);
        let _height: u32 = f[i_h].parse().unwrap_or(0);
        let cat = f[i_cat].to_string();
        let truth = map_gpt_category(&cat);
        if truth == Truth::Skip {
            skipped_unmapped += 1;
            continue;
        }
        let Some(root) = corpus_root(&corpus) else {
            skipped_missing += 1;
            continue;
        };
        let path = Path::new(root).join(&image);
        if !path.exists() {
            skipped_missing += 1;
            continue;
        }
        let Some((rgb, w, h)) = load_image(&path) else {
            skipped_load += 1;
            continue;
        };
        let metrics = compute_metrics(&rgb);
        // Sanity: no NaN in any metric.
        let any_nan = metrics.entropy_y32.is_nan()
            || metrics.entropy_y64.is_nan()
            || metrics.entropy_l32.is_nan()
            || metrics.entropy_normalized.is_nan()
            || metrics.top1_fraction.is_nan()
            || metrics.top10_fraction.is_nan();
        if any_nan {
            eprintln!(
                "NaN metric for {}/{} — skipping",
                corpus, image
            );
            continue;
        }
        let row = Row {
            corpus,
            image,
            width: w,
            height: h,
            truth_bucket: truth,
            primary_category: cat,
            metrics,
        };
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}",
            row.corpus,
            row.image,
            row.width,
            row.height,
            match row.truth_bucket {
                Truth::Photo => "Photo",
                Truth::Drawing => "Drawing",
                _ => "Skip",
            },
            row.primary_category,
            row.metrics.entropy_y32,
            row.metrics.entropy_y64,
            row.metrics.entropy_l32,
            row.metrics.entropy_normalized,
            row.metrics.top1_fraction,
            row.metrics.top10_fraction,
            row.metrics.distinct_bins_y32,
            row.metrics.total_pixels,
        )
        .unwrap();
        rows.push(row);
    }
    drop(tsv);

    // (name, extractor, direction): direction = +1 means "higher -> Photo".
    type MetricFn = fn(&Row) -> f64;
    let metrics: &[(&str, MetricFn, i32)] = &[
        ("chroma_entropy_ycbcr_32", |r| r.metrics.entropy_y32, 1),
        ("chroma_entropy_ycbcr_64", |r| r.metrics.entropy_y64, 1),
        ("chroma_entropy_lab_32", |r| r.metrics.entropy_l32, 1),
        ("chroma_entropy_normalized", |r| r.metrics.entropy_normalized, 1),
        ("chroma_top1_fraction", |r| r.metrics.top1_fraction, -1),
        ("chroma_top10_fraction", |r| r.metrics.top10_fraction, -1),
    ];

    let mut summary = String::new();
    summary.push_str("# ChromaEntropy Spike — Results\n\n");
    summary.push_str(&format!("Corpus: {}\n\n", labels_path.display()));
    summary.push_str(&format!(
        "Rows total: {}, used: {}, skipped (unmapped category): {}, \
         skipped (file missing): {}, skipped (load fail): {}\n\n",
        total,
        rows.len(),
        skipped_unmapped,
        skipped_missing,
        skipped_load,
    ));

    let n_photo = rows.iter().filter(|r| r.truth_bucket == Truth::Photo).count();
    let n_draw = rows
        .iter()
        .filter(|r| r.truth_bucket == Truth::Drawing)
        .count();
    summary.push_str(&format!("Photo: {}, Drawing: {}\n\n", n_photo, n_draw));

    summary.push_str("## Per-metric distributions\n\n");
    summary.push_str(
        "| Metric | bucket | n | mean | p10 | p25 | median | p75 | p90 |\n",
    );
    summary.push_str("|---|---|---|---|---|---|---|---|---|\n");

    let mut per_metric_stats: BTreeMap<String, (f64, f64, f64, i32)> = BTreeMap::new();

    for (name, f, direction) in metrics {
        for bucket_name in &["Photo", "Drawing"] {
            let truth = if *bucket_name == "Photo" {
                Truth::Photo
            } else {
                Truth::Drawing
            };
            let mut vs: Vec<f64> = rows
                .iter()
                .filter(|r| r.truth_bucket == truth)
                .map(|r| f(r))
                .collect();
            vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = mean(&vs);
            summary.push_str(&format!(
                "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                name,
                bucket_name,
                vs.len(),
                m,
                percentile_sorted(&vs, 0.10),
                percentile_sorted(&vs, 0.25),
                median_sorted(&vs),
                percentile_sorted(&vs, 0.75),
                percentile_sorted(&vs, 0.90),
            ));
        }
        let pairs: Vec<(f64, Truth)> = rows
            .iter()
            .map(|r| (f(r), r.truth_bucket))
            .collect();
        let (auc, thr, acc) = auc_and_best_threshold(&pairs, *direction);
        per_metric_stats.insert(name.to_string(), (auc, thr, acc, *direction));
    }

    summary.push_str("\n## AUC summary (Photo = positive class)\n\n");
    summary.push_str(
        "| Metric | direction | AUC | best_threshold | best_accuracy |\n\
         |---|---|---|---|---|\n",
    );
    let mut sorted_metrics: Vec<(String, (f64, f64, f64, i32))> = per_metric_stats
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    sorted_metrics.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    for (name, (auc, thr, acc, dir)) in &sorted_metrics {
        let arrow = if *dir >= 0 { ">" } else { "<" };
        summary.push_str(&format!(
            "| {} | photo if metric {} thr | {:.4} | {:.6} | {:.4} |\n",
            name, arrow, auc, thr, acc
        ));
    }

    let best = sorted_metrics.first().cloned();

    // Hard-case illustrations
    let hard_cases = [
        "14768480444_bc0a1b4a2e_o.png",
        "1910225.png",
        "26103251787_d8635e260d_o.png",
        "2693212.png",
        "3140d643a1f57e80ff265507984d95a0.png",
        "86127fbdb368eb28c3039cf61aff1c4cfdc4ade24070c8f2389968d5ead681e1.png",
        "mies-lossless.png",
    ];

    summary.push_str("\n## Hard-case illustrations vs photo distribution\n\n");

    if let Some((best_name, _)) = best.as_ref() {
        let best_extractor: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == best_name.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let best_dir = metrics
            .iter()
            .find(|(n, _, _)| *n == best_name.as_str())
            .map(|(_, _, d)| *d)
            .unwrap();

        let mut photo_best: Vec<f64> = rows
            .iter()
            .filter(|r| r.truth_bucket == Truth::Photo)
            .map(|r| best_extractor(r))
            .collect();
        photo_best.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pb_p10 = percentile_sorted(&photo_best, 0.10);
        let pb_p25 = percentile_sorted(&photo_best, 0.25);
        let pb_p75 = percentile_sorted(&photo_best, 0.75);
        let pb_p90 = percentile_sorted(&photo_best, 0.90);
        summary.push_str(&format!(
            "Best metric: **{}** (direction: photo if metric {} threshold).\n\n",
            best_name,
            if best_dir >= 0 { ">" } else { "<" }
        ));
        summary.push_str(&format!(
            "Photo p10/p25/p75/p90 = {:.4} / {:.4} / {:.4} / {:.4}\n\n",
            pb_p10, pb_p25, pb_p75, pb_p90
        ));

        // For drawing, the "outside photo distribution" criterion depends on direction.
        // direction +1 means high-value -> Photo, so a Drawing should be LOW (< photo_p10).
        // direction -1 means low-value -> Photo, so a Drawing should be HIGH (> photo_p90).
        summary.push_str(
            "| image | corpus | category | y32_entropy | y32_entropy_norm | top1 | top10 | best_metric_value | outside_photo_dist? |\n\
             |---|---|---|---|---|---|---|---|---|\n",
        );
        let mut found_hard = 0;
        let mut outside_count = 0;
        for h in &hard_cases {
            if let Some(r) = rows.iter().find(|r| r.image == *h) {
                found_hard += 1;
                let v = best_extractor(r);
                let outside = if best_dir >= 0 { v < pb_p10 } else { v > pb_p90 };
                if outside {
                    outside_count += 1;
                }
                summary.push_str(&format!(
                    "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} |\n",
                    r.image,
                    r.corpus,
                    r.primary_category,
                    r.metrics.entropy_y32,
                    r.metrics.entropy_normalized,
                    r.metrics.top1_fraction,
                    r.metrics.top10_fraction,
                    v,
                    if outside { "YES" } else { "no" },
                ));
            } else {
                summary.push_str(&format!(
                    "| {} | (not in corpus) | | | | | | | |\n",
                    h
                ));
            }
        }
        summary.push_str(&format!(
            "\nFound {} of 7 hard cases in run; {} are outside photo distribution (clean separation).\n\n",
            found_hard, outside_count
        ));
    }

    // Correlation between metrics.
    summary.push_str("\n## Pairwise Pearson correlation\n\n");
    summary.push_str(
        "| metric A | metric B | r |\n|---|---|---|\n",
    );
    let pairs: Vec<(&str, MetricFn, &str, MetricFn)> = vec![
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "chroma_top1_fraction", |r: &Row| r.metrics.top1_fraction),
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "chroma_top10_fraction", |r: &Row| r.metrics.top10_fraction),
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "chroma_entropy_normalized", |r: &Row| r.metrics.entropy_normalized),
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "distinct_bins_y32", |r: &Row| r.metrics.distinct_bins_y32 as f64),
        ("chroma_top1_fraction", |r: &Row| r.metrics.top1_fraction,
         "distinct_bins_y32", |r: &Row| r.metrics.distinct_bins_y32 as f64),
    ];
    for (na, fa, nb, fb) in &pairs {
        let xs: Vec<f64> = rows.iter().map(|r| fa(r)).collect();
        let ys: Vec<f64> = rows.iter().map(|r| fb(r)).collect();
        let r = pearson(&xs, &ys);
        summary.push_str(&format!("| {} | {} | {:.4} |\n", na, nb, r));
    }

    // Composite: best linear combination of (entropy_y32, distinct_bins_y32).
    // Tests whether entropy + count are orthogonal enough to outperform either alone.
    summary.push_str("\n## Composite: linear combination test\n\n");
    summary.push_str(
        "Tests whether two metrics combined (z-scored, swept over angle) outperform either alone.\n\
         If best_combo_AUC >> max(individual AUC), the metrics are orthogonal and combining helps.\n\n",
    );
    let truths: Vec<Truth> = rows.iter().map(|r| r.truth_bucket).collect();
    let combo_pairs: Vec<(&str, MetricFn, &str, MetricFn)> = vec![
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "distinct_bins_y32", |r: &Row| r.metrics.distinct_bins_y32 as f64),
        ("chroma_top1_fraction", |r: &Row| r.metrics.top1_fraction,
         "distinct_bins_y32", |r: &Row| r.metrics.distinct_bins_y32 as f64),
        ("chroma_entropy_normalized", |r: &Row| r.metrics.entropy_normalized,
         "distinct_bins_y32", |r: &Row| r.metrics.distinct_bins_y32 as f64),
        ("chroma_entropy_ycbcr_32", |r: &Row| r.metrics.entropy_y32,
         "chroma_top1_fraction", |r: &Row| r.metrics.top1_fraction),
    ];
    summary.push_str(
        "| metric A | metric B | combo AUC | combo acc | a (z-A) | b (z-B) |\n\
         |---|---|---|---|---|---|\n",
    );
    for (na, fa, nb, fb) in &combo_pairs {
        let xs: Vec<f64> = rows.iter().map(|r| fa(r)).collect();
        let ys: Vec<f64> = rows.iter().map(|r| fb(r)).collect();
        let (auc, acc, a, b) = best_linear_combo(&xs, &ys, &truths);
        summary.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.3} | {:.3} |\n",
            na, nb, auc, acc, a, b
        ));
    }

    print!("{}", summary);
    fs::write(&summary_path, &summary).expect("write summary.md");
    eprintln!("\n[wrote {}]", tsv_path.display());
    eprintln!("[wrote {}]", summary_path.display());
}
