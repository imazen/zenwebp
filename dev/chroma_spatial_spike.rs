//! ChromaSpatial spike — research prototype.
//!
//! Hypothesis: the artwork-vs-photo signal lives in the *spatial structure*
//! of chroma, not in any aggregate histogram. Photographs have continuous
//! chroma gradients (illumination, perspective, atmospheric scattering,
//! sensor noise + lens shading). Digital artwork has chroma palette
//! regions: flat-fill areas bordered by sharp chroma boundaries.
//!
//! For each labelled image we compute these spatial chroma metrics:
//!
//!   * `chroma_var_mean/stddev/p10/p25/p50/p75/p90` — distribution of
//!     per-block (Cb_var + Cr_var) across 8x8 blocks.
//!   * `flat_block_fraction_t1/t4/t16` — fraction of blocks whose chroma
//!     variance is below a flatness threshold.
//!   * `chroma_var_iqr_ratio` — (p75 - p25) / (p75 + p25), bimodality
//!     proxy. Higher = more spread between concentrated and dispersed.
//!   * `chroma_var_log_entropy` — Shannon entropy over a 32-bin log-scale
//!     histogram of block variances. Bimodal → low entropy.
//!   * `chroma_luma_edge_agreement` — fraction of luma-edge pixels that
//!     also have a chroma edge. Photos high, artwork low (palette
//!     boundaries don't always align with luma edges).
//!   * `chroma_block_neighbor_delta` — sum of |Δmean_Cb| + |Δmean_Cr|
//!     across right/down neighbor pairs of 8x8 blocks, normalized.
//!     Photos: smooth → low; artwork: blocky → high.
//!
//! Output: per-image TSV + summary.md with per-bucket distributions,
//! AUC (Photo positive class), best thresholds, hard-case analysis,
//! and 2/3-way composite linear classifiers.
//!
//! Usage (from repo root):
//!   cargo run --release --example chroma_spatial_spike -- \
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

/// BT.601 RGB->YCbCr (gamma-encoded, full-range Y/Cb/Cr in [0,255]).
#[inline]
fn rgb_to_ycbcr_bt601(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    // BT.601: Y = 0.299 R + 0.587 G + 0.114 B (full range)
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = (-43 * r - 85 * g + 128 * b + 128 * 256 + 128) >> 8;
    let cr = (128 * r - 107 * g - 21 * b + 128 * 256 + 128) >> 8;
    (
        y.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
    )
}

#[derive(Default, Clone, Debug)]
struct SpatialMetrics {
    chroma_var_mean: f64,
    chroma_var_stddev: f64,
    chroma_var_p10: f64,
    chroma_var_p25: f64,
    chroma_var_p50: f64,
    chroma_var_p75: f64,
    chroma_var_p90: f64,
    flat_block_fraction_t1: f64,
    flat_block_fraction_t4: f64,
    flat_block_fraction_t16: f64,
    chroma_var_iqr_ratio: f64,
    chroma_var_log_entropy: f64,
    chroma_luma_edge_agreement: f64,
    chroma_block_neighbor_delta: f64,
    n_blocks: u32,
    block_size: u32,
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let r = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[r.min(sorted.len() - 1)]
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

fn compute_spatial(rgb: &[u8], w: u32, h: u32, block_size: u32) -> SpatialMetrics {
    let w = w as usize;
    let h = h as usize;
    let bs = block_size as usize;

    // Convert to planar Y/Cb/Cr.
    let n_pix = w * h;
    let mut y_plane = vec![0u8; n_pix];
    let mut cb_plane = vec![0u8; n_pix];
    let mut cr_plane = vec![0u8; n_pix];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        let (y, cb, cr) = rgb_to_ycbcr_bt601(px[0], px[1], px[2]);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    // ---- Per-block chroma variance ----
    let bw = w / bs;
    let bh = h / bs;
    let n_blocks = bw * bh;
    if n_blocks == 0 {
        return SpatialMetrics {
            block_size: block_size,
            ..Default::default()
        };
    }

    let mut chroma_vars: Vec<f64> = Vec::with_capacity(n_blocks);
    let mut block_mean_cb: Vec<f64> = Vec::with_capacity(n_blocks);
    let mut block_mean_cr: Vec<f64> = Vec::with_capacity(n_blocks);

    for by in 0..bh {
        for bx in 0..bw {
            let mut s_cb: u64 = 0;
            let mut s_cr: u64 = 0;
            let mut s_cb2: u64 = 0;
            let mut s_cr2: u64 = 0;
            let n = (bs * bs) as u64;
            for dy in 0..bs {
                let row = (by * bs + dy) * w + bx * bs;
                for dx in 0..bs {
                    let cb = cb_plane[row + dx] as u64;
                    let cr = cr_plane[row + dx] as u64;
                    s_cb += cb;
                    s_cr += cr;
                    s_cb2 += cb * cb;
                    s_cr2 += cr * cr;
                }
            }
            let mean_cb = s_cb as f64 / n as f64;
            let mean_cr = s_cr as f64 / n as f64;
            let var_cb = (s_cb2 as f64 / n as f64) - mean_cb * mean_cb;
            let var_cr = (s_cr2 as f64 / n as f64) - mean_cr * mean_cr;
            let var_total = var_cb.max(0.0) + var_cr.max(0.0);
            chroma_vars.push(var_total);
            block_mean_cb.push(mean_cb);
            block_mean_cr.push(mean_cr);
        }
    }

    let mut sorted_vars = chroma_vars.clone();
    sorted_vars.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_var: f64 = chroma_vars.iter().sum::<f64>() / chroma_vars.len() as f64;
    let var_of_var: f64 = chroma_vars
        .iter()
        .map(|v| (v - mean_var) * (v - mean_var))
        .sum::<f64>()
        / chroma_vars.len() as f64;
    let stddev_var = var_of_var.sqrt();

    let p10 = percentile_sorted(&sorted_vars, 0.10);
    let p25 = percentile_sorted(&sorted_vars, 0.25);
    let p50 = percentile_sorted(&sorted_vars, 0.50);
    let p75 = percentile_sorted(&sorted_vars, 0.75);
    let p90 = percentile_sorted(&sorted_vars, 0.90);

    let flat_t1 = chroma_vars.iter().filter(|&&v| v < 1.0).count() as f64 / n_blocks as f64;
    let flat_t4 = chroma_vars.iter().filter(|&&v| v < 4.0).count() as f64 / n_blocks as f64;
    let flat_t16 = chroma_vars.iter().filter(|&&v| v < 16.0).count() as f64 / n_blocks as f64;

    let iqr_ratio = if (p75 + p25) > 1e-9 {
        (p75 - p25) / (p75 + p25)
    } else {
        0.0
    };

    // 32-bin log-scale histogram of variances, range [0, 16384] log-spaced.
    // Bin = floor(log2(v + 1) * 32 / 14), clamped to [0, 31].
    let mut log_hist = vec![0u32; 32];
    for &v in &chroma_vars {
        let lv = (v + 1.0).log2();
        let bin = ((lv * 32.0 / 14.0) as i32).clamp(0, 31) as usize;
        log_hist[bin] += 1;
    }
    let log_entropy = shannon_entropy(&log_hist);

    // ---- Chroma-luma edge agreement ----
    // For each interior pixel, compute |dY| + |dY|' and |dCb| + |dCr|.
    // Among pixels with luma_grad > LUMA_EDGE_THR, what fraction have
    // chroma_grad > CHROMA_EDGE_THR?
    const LUMA_EDGE_THR: u32 = 16;
    const CHROMA_EDGE_THR: u32 = 4;
    let mut luma_edge_count: u64 = 0;
    let mut both_edge_count: u64 = 0;
    for yy in 1..(h - 1) {
        for xx in 1..(w - 1) {
            let i = yy * w + xx;
            let dy_x = (y_plane[i + 1] as i32 - y_plane[i - 1] as i32).unsigned_abs();
            let dy_y = (y_plane[i + w] as i32 - y_plane[i - w] as i32).unsigned_abs();
            let luma_grad = dy_x + dy_y;
            if luma_grad > LUMA_EDGE_THR {
                luma_edge_count += 1;
                let dcb_x = (cb_plane[i + 1] as i32 - cb_plane[i - 1] as i32).unsigned_abs();
                let dcb_y = (cb_plane[i + w] as i32 - cb_plane[i - w] as i32).unsigned_abs();
                let dcr_x = (cr_plane[i + 1] as i32 - cr_plane[i - 1] as i32).unsigned_abs();
                let dcr_y = (cr_plane[i + w] as i32 - cr_plane[i - w] as i32).unsigned_abs();
                let chroma_grad = dcb_x + dcb_y + dcr_x + dcr_y;
                if chroma_grad > CHROMA_EDGE_THR {
                    both_edge_count += 1;
                }
            }
        }
    }
    let edge_agreement = if luma_edge_count > 0 {
        both_edge_count as f64 / luma_edge_count as f64
    } else {
        0.0
    };

    // ---- Block-level chroma neighbor delta ----
    // For each block, compute |Δmean_Cb| + |Δmean_Cr| with right and down
    // neighbors. Sum and normalize by the count of neighbor pairs.
    let mut neighbor_delta_sum = 0.0f64;
    let mut neighbor_pairs = 0u64;
    for by in 0..bh {
        for bx in 0..bw {
            let i = by * bw + bx;
            // Right neighbor.
            if bx + 1 < bw {
                let j = by * bw + (bx + 1);
                neighbor_delta_sum += (block_mean_cb[i] - block_mean_cb[j]).abs()
                    + (block_mean_cr[i] - block_mean_cr[j]).abs();
                neighbor_pairs += 1;
            }
            // Down neighbor.
            if by + 1 < bh {
                let j = (by + 1) * bw + bx;
                neighbor_delta_sum += (block_mean_cb[i] - block_mean_cb[j]).abs()
                    + (block_mean_cr[i] - block_mean_cr[j]).abs();
                neighbor_pairs += 1;
            }
        }
    }
    let neighbor_delta = if neighbor_pairs > 0 {
        neighbor_delta_sum / neighbor_pairs as f64
    } else {
        0.0
    };

    SpatialMetrics {
        chroma_var_mean: mean_var,
        chroma_var_stddev: stddev_var,
        chroma_var_p10: p10,
        chroma_var_p25: p25,
        chroma_var_p50: p50,
        chroma_var_p75: p75,
        chroma_var_p90: p90,
        flat_block_fraction_t1: flat_t1,
        flat_block_fraction_t4: flat_t4,
        flat_block_fraction_t16: flat_t16,
        chroma_var_iqr_ratio: iqr_ratio,
        chroma_var_log_entropy: log_entropy,
        chroma_luma_edge_agreement: edge_agreement,
        chroma_block_neighbor_delta: neighbor_delta,
        n_blocks: n_blocks as u32,
        block_size: block_size,
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
    metrics8: SpatialMetrics,
    metrics16: SpatialMetrics,
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
fn auc_and_best_threshold(values: &[(f64, Truth)], direction: i32) -> (f64, f64, f64) {
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
                if *v > thr { Truth::Photo } else { Truth::Drawing }
            } else {
                if *v < thr { Truth::Photo } else { Truth::Drawing }
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

fn z_score(xs: &[f64]) -> Vec<f64> {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let sx = (xs.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    xs.iter().map(|v| (v - mx) / sx).collect()
}

/// Best linear combo of two z-scored metrics, sweeping unit-circle angles.
fn best_linear_combo_2(
    xs: &[f64],
    ys: &[f64],
    truths: &[Truth],
) -> (f64, f64, f64, f64) {
    let zx = z_score(xs);
    let zy = z_score(ys);
    let mut best_auc = 0.0;
    let mut best_acc = 0.0;
    let mut best_a = 0.0;
    let mut best_b = 0.0;
    let n_angles = 72;
    for k in 0..n_angles {
        let theta = std::f64::consts::PI * (k as f64) / (n_angles as f64);
        let a = theta.cos();
        let b = theta.sin();
        let combo: Vec<(f64, Truth)> = (0..xs.len())
            .map(|i| (a * zx[i] + b * zy[i], truths[i]))
            .collect();
        let (auc, _, acc) = auc_and_best_threshold(&combo, 1);
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

/// Best linear combo of three z-scored metrics, swept on hemisphere grid.
fn best_linear_combo_3(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    truths: &[Truth],
) -> (f64, f64, f64, f64, f64) {
    let zx = z_score(xs);
    let zy = z_score(ys);
    let zz = z_score(zs);
    let mut best_auc = 0.0;
    let mut best_acc = 0.0;
    let mut best_a = 0.0;
    let mut best_b = 0.0;
    let mut best_c = 0.0;
    let n_theta = 18; // polar
    let n_phi = 36; // azimuth
    for ti in 0..=n_theta {
        let theta = std::f64::consts::PI * (ti as f64) / (n_theta as f64);
        let st = theta.sin();
        let ct = theta.cos();
        for pi in 0..n_phi {
            let phi = 2.0 * std::f64::consts::PI * (pi as f64) / (n_phi as f64);
            let a = st * phi.cos();
            let b = st * phi.sin();
            let c = ct;
            let combo: Vec<(f64, Truth)> = (0..xs.len())
                .map(|i| (a * zx[i] + b * zy[i] + c * zz[i], truths[i]))
                .collect();
            let (auc, _, acc) = auc_and_best_threshold(&combo, 1);
            let auc_eff = auc.max(1.0 - auc);
            if auc_eff > best_auc {
                best_auc = auc_eff;
                best_acc = acc.max(1.0 - acc);
                best_a = a;
                best_b = b;
                best_c = c;
            }
        }
    }
    (best_auc, best_acc, best_a, best_b, best_c)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let labels_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/lilith/work/coefficient/benchmarks/classifier-eval/labels.tsv")
    });

    let f = fs::File::open(&labels_path).expect("open labels.tsv");
    let r = BufReader::new(f);

    let out_dir = PathBuf::from("/mnt/v/output/zenwebp/chroma-spatial-spike");
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
         chroma_var_mean8\tchroma_var_stddev8\tchroma_var_p10_8\tchroma_var_p50_8\tchroma_var_p90_8\t\
         flat_block_fraction_t1_8\tflat_block_fraction_t4_8\tflat_block_fraction_t16_8\t\
         chroma_var_iqr_ratio_8\tchroma_var_log_entropy_8\t\
         chroma_luma_edge_agreement\tchroma_block_neighbor_delta_8\t\
         chroma_var_mean16\tflat_block_fraction_t4_16\tchroma_var_log_entropy_16\tchroma_block_neighbor_delta_16\t\
         n_blocks_8\tn_blocks_16"
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
    let t_start = std::time::Instant::now();
    let mut total_pixels: u64 = 0;
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
        // Need at least one 16x16 block.
        if w < 16 || h < 16 {
            skipped_load += 1;
            continue;
        }
        total_pixels += (w as u64) * (h as u64);
        let metrics8 = compute_spatial(&rgb, w, h, 8);
        let metrics16 = compute_spatial(&rgb, w, h, 16);
        // NaN guard
        let any_nan = metrics8.chroma_var_mean.is_nan()
            || metrics8.chroma_var_log_entropy.is_nan()
            || metrics8.chroma_luma_edge_agreement.is_nan()
            || metrics8.chroma_block_neighbor_delta.is_nan()
            || metrics16.chroma_var_mean.is_nan();
        if any_nan {
            eprintln!("NaN metric for {}/{} — skipping", corpus, image);
            continue;
        }
        let row = Row {
            corpus,
            image,
            width: w,
            height: h,
            truth_bucket: truth,
            primary_category: cat,
            metrics8: metrics8.clone(),
            metrics16: metrics16.clone(),
        };
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t\
             {:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t\
             {:.6}\t{:.6}\t{:.6}\t\
             {:.6}\t{:.6}\t\
             {:.6}\t{:.4}\t\
             {:.4}\t{:.6}\t{:.6}\t{:.4}\t\
             {}\t{}",
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
            metrics8.chroma_var_mean,
            metrics8.chroma_var_stddev,
            metrics8.chroma_var_p10,
            metrics8.chroma_var_p50,
            metrics8.chroma_var_p90,
            metrics8.flat_block_fraction_t1,
            metrics8.flat_block_fraction_t4,
            metrics8.flat_block_fraction_t16,
            metrics8.chroma_var_iqr_ratio,
            metrics8.chroma_var_log_entropy,
            metrics8.chroma_luma_edge_agreement,
            metrics8.chroma_block_neighbor_delta,
            metrics16.chroma_var_mean,
            metrics16.flat_block_fraction_t4,
            metrics16.chroma_var_log_entropy,
            metrics16.chroma_block_neighbor_delta,
            metrics8.n_blocks,
            metrics16.n_blocks,
        )
        .unwrap();
        rows.push(row);
    }
    drop(tsv);
    let elapsed = t_start.elapsed();
    let cycles_per_pixel = if total_pixels > 0 {
        // Approximate: assume 4.5 GHz reference. Reports actual ns/pixel too.
        (elapsed.as_nanos() as f64 * 4.5) / (total_pixels as f64)
    } else {
        0.0
    };
    let ns_per_pixel = if total_pixels > 0 {
        elapsed.as_nanos() as f64 / total_pixels as f64
    } else {
        0.0
    };

    type MetricFn = fn(&Row) -> f64;
    // (name, extractor, direction): +1 = higher → Photo
    let metrics: &[(&str, MetricFn, i32)] = &[
        ("chroma_var_mean_8", |r| r.metrics8.chroma_var_mean, 1),
        ("chroma_var_stddev_8", |r| r.metrics8.chroma_var_stddev, 1),
        ("chroma_var_p10_8", |r| r.metrics8.chroma_var_p10, 1),
        ("chroma_var_p25_8", |r| r.metrics8.chroma_var_p25, 1),
        ("chroma_var_p50_8", |r| r.metrics8.chroma_var_p50, 1),
        ("chroma_var_p75_8", |r| r.metrics8.chroma_var_p75, 1),
        ("chroma_var_p90_8", |r| r.metrics8.chroma_var_p90, 1),
        ("flat_block_fraction_t1_8", |r| r.metrics8.flat_block_fraction_t1, -1),
        ("flat_block_fraction_t4_8", |r| r.metrics8.flat_block_fraction_t4, -1),
        ("flat_block_fraction_t16_8", |r| r.metrics8.flat_block_fraction_t16, -1),
        ("chroma_var_iqr_ratio_8", |r| r.metrics8.chroma_var_iqr_ratio, -1),
        ("chroma_var_log_entropy_8", |r| r.metrics8.chroma_var_log_entropy, 1),
        ("chroma_luma_edge_agreement", |r| r.metrics8.chroma_luma_edge_agreement, 1),
        ("chroma_block_neighbor_delta_8", |r| r.metrics8.chroma_block_neighbor_delta, 1),
        ("chroma_var_mean_16", |r| r.metrics16.chroma_var_mean, 1),
        ("flat_block_fraction_t4_16", |r| r.metrics16.flat_block_fraction_t4, -1),
        ("chroma_var_log_entropy_16", |r| r.metrics16.chroma_var_log_entropy, 1),
        ("chroma_block_neighbor_delta_16", |r| r.metrics16.chroma_block_neighbor_delta, 1),
    ];

    let mut summary = String::new();
    summary.push_str("# ChromaSpatial Spike — Results\n\n");
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
    summary.push_str(&format!(
        "Total pixels processed: {}; total wall-clock: {:.2}s; \
         {:.2} ns/pixel (≈ {:.2} cycles/pixel @ 4.5 GHz, single-thread, includes RGB→YCbCr + all metrics)\n\n",
        total_pixels,
        elapsed.as_secs_f64(),
        ns_per_pixel,
        cycles_per_pixel,
    ));

    let n_photo = rows
        .iter()
        .filter(|r| r.truth_bucket == Truth::Photo)
        .count();
    let n_draw = rows
        .iter()
        .filter(|r| r.truth_bucket == Truth::Drawing)
        .count();
    summary.push_str(&format!("Photo: {}, Drawing: {}\n\n", n_photo, n_draw));

    summary.push_str("## Per-metric distributions\n\n");
    summary.push_str("| Metric | bucket | n | mean | p10 | p25 | median | p75 | p90 |\n");
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

        // For direction +1: drawing should be LOW (< photo_p10 or < photo_p25)
        // For direction -1: drawing should be HIGH (> photo_p75 or > photo_p90)
        summary.push_str(
            "| image | corpus | category | best_metric_value | outside_p10/p90? | outside_p25/p75? |\n\
             |---|---|---|---|---|---|\n",
        );
        let mut found_hard = 0;
        let mut outside_extreme = 0;
        let mut outside_quartile = 0;
        for h in &hard_cases {
            if let Some(r) = rows.iter().find(|r| r.image == *h) {
                found_hard += 1;
                let v = best_extractor(r);
                let outside_x = if best_dir >= 0 { v < pb_p10 } else { v > pb_p90 };
                let outside_q = if best_dir >= 0 { v < pb_p25 } else { v > pb_p75 };
                if outside_x {
                    outside_extreme += 1;
                }
                if outside_q {
                    outside_quartile += 1;
                }
                summary.push_str(&format!(
                    "| {} | {} | {} | {:.4} | {} | {} |\n",
                    r.image,
                    r.corpus,
                    r.primary_category,
                    v,
                    if outside_x { "YES" } else { "no" },
                    if outside_q { "YES" } else { "no" },
                ));
            } else {
                summary.push_str(&format!("| {} | (not in corpus) | | | | |\n", h));
            }
        }
        summary.push_str(&format!(
            "\nFound {} of 7 hard cases; {} outside extreme (p10/p90), {} outside quartile (p25/p75).\n\n",
            found_hard, outside_extreme, outside_quartile
        ));

        // Also: for the top 4 metrics, report hard-case separation.
        summary.push_str("\n### Hard-case separation across top metrics\n\n");
        summary.push_str(
            "| metric | photo p25 | photo p75 | hard cases outside p25/p75 (of 7 found) |\n\
             |---|---|---|---|\n",
        );
        for (name, (_, _, _, _)) in sorted_metrics.iter().take(6) {
            let extractor = metrics
                .iter()
                .find(|(n, _, _)| *n == name.as_str())
                .map(|(_, f, _)| *f)
                .unwrap();
            let dir = metrics
                .iter()
                .find(|(n, _, _)| *n == name.as_str())
                .map(|(_, _, d)| *d)
                .unwrap();
            let mut p_vals: Vec<f64> = rows
                .iter()
                .filter(|r| r.truth_bucket == Truth::Photo)
                .map(|r| extractor(r))
                .collect();
            p_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p25_v = percentile_sorted(&p_vals, 0.25);
            let p75_v = percentile_sorted(&p_vals, 0.75);
            let mut hits = 0;
            let mut found = 0;
            for h in &hard_cases {
                if let Some(r) = rows.iter().find(|r| r.image == *h) {
                    found += 1;
                    let v = extractor(r);
                    let outside_q = if dir >= 0 { v < p25_v } else { v > p75_v };
                    if outside_q {
                        hits += 1;
                    }
                }
            }
            summary.push_str(&format!(
                "| {} | {:.4} | {:.4} | {}/{} |\n",
                name, p25_v, p75_v, hits, found
            ));
        }
    }

    // Pearson correlation among top 6 metrics.
    summary.push_str("\n## Pairwise Pearson correlation (top metrics)\n\n");
    let top_names: Vec<String> = sorted_metrics
        .iter()
        .take(6)
        .map(|(n, _)| n.clone())
        .collect();
    summary.push_str("| metric A | metric B | r |\n|---|---|---|\n");
    for i in 0..top_names.len() {
        for j in (i + 1)..top_names.len() {
            let na = &top_names[i];
            let nb = &top_names[j];
            let ext_a: MetricFn = metrics
                .iter()
                .find(|(n, _, _)| *n == na.as_str())
                .map(|(_, f, _)| *f)
                .unwrap();
            let ext_b: MetricFn = metrics
                .iter()
                .find(|(n, _, _)| *n == nb.as_str())
                .map(|(_, f, _)| *f)
                .unwrap();
            let xs: Vec<f64> = rows.iter().map(|r| ext_a(r)).collect();
            let ys: Vec<f64> = rows.iter().map(|r| ext_b(r)).collect();
            let r = pearson(&xs, &ys);
            summary.push_str(&format!("| {} | {} | {:.4} |\n", na, nb, r));
        }
    }

    // Composite tests.
    summary.push_str("\n## Composite linear classifiers (z-scored)\n\n");
    summary.push_str(
        "Tests whether linear combinations of top metrics outperform any single metric.\n\
         AUC is direction-invariant (we take max(auc, 1-auc)).\n\n",
    );
    let truths: Vec<Truth> = rows.iter().map(|r| r.truth_bucket).collect();

    summary.push_str("### 2-way combos\n\n");
    summary.push_str(
        "| metric A | metric B | combo AUC | combo acc | a (z-A) | b (z-B) |\n\
         |---|---|---|---|---|---|\n",
    );
    let combo_pairs: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3)];
    for (i, j) in &combo_pairs {
        if *i >= top_names.len() || *j >= top_names.len() {
            continue;
        }
        let na = &top_names[*i];
        let nb = &top_names[*j];
        let ext_a: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == na.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let ext_b: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == nb.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let xs: Vec<f64> = rows.iter().map(|r| ext_a(r)).collect();
        let ys: Vec<f64> = rows.iter().map(|r| ext_b(r)).collect();
        let (auc, acc, a, b) = best_linear_combo_2(&xs, &ys, &truths);
        summary.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.3} | {:.3} |\n",
            na, nb, auc, acc, a, b
        ));
    }

    summary.push_str("\n### 3-way combos (top 3, top 1+2+4, top 1+3+4)\n\n");
    summary.push_str(
        "| A | B | C | combo AUC | combo acc | a | b | c |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    let combo_triples: Vec<(usize, usize, usize)> = vec![(0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1, 4)];
    for (i, j, k) in &combo_triples {
        if *i >= top_names.len() || *j >= top_names.len() || *k >= top_names.len() {
            continue;
        }
        let na = &top_names[*i];
        let nb = &top_names[*j];
        let nc = &top_names[*k];
        let ext_a: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == na.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let ext_b: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == nb.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let ext_c: MetricFn = metrics
            .iter()
            .find(|(n, _, _)| *n == nc.as_str())
            .map(|(_, f, _)| *f)
            .unwrap();
        let xs: Vec<f64> = rows.iter().map(|r| ext_a(r)).collect();
        let ys: Vec<f64> = rows.iter().map(|r| ext_b(r)).collect();
        let zs: Vec<f64> = rows.iter().map(|r| ext_c(r)).collect();
        let (auc, acc, a, b, c) = best_linear_combo_3(&xs, &ys, &zs, &truths);
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.3} | {:.3} | {:.3} |\n",
            na, nb, nc, auc, acc, a, b, c
        ));
    }

    print!("{}", summary);
    fs::write(&summary_path, &summary).expect("write summary.md");
    eprintln!("\n[wrote {}]", tsv_path.display());
    eprintln!("[wrote {}]", summary_path.display());
}
