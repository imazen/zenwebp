//! Validate zenwebp's content classifier against 219 GPT-labelled
//! images from the coefficient classifier-eval corpus.
//!
//! Compares three classifiers on each image:
//!  - **old**: alpha-histogram + Y-plane uniformity (original
//!    `classify_image_type`)
//!  - **stable**: zenanalyze stable signals only (no experimental)
//!  - **experimental**: stable + `PaletteFitsIn256` / `LineArtScore`
//!    (current `classify_image_type_rgb8`)
//!
//! GPT 8-way categories collapse to zenwebp's 3-way bucket:
//!  - `photo_*`            -> Photo
//!  - `screen_*`           -> Drawing
//!  - `illustration`       -> Drawing
//!
//! Outputs:
//!  - per-image TSV to /mnt/v/output/zenwebp/zenanalyze-validate/run_<ts>.tsv
//!  - confusion matrices + accuracy summary to stdout
//!
//! Usage (from repo root):
//!   cargo run --release --features analyzer \
//!     --example zenanalyze_validate_vs_gpt -- \
//!     /home/lilith/work/coefficient/benchmarks/classifier-eval/labels.tsv

#![cfg(feature = "analyzer")]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use zenwebp::encoder::analysis::{
    ImageContentType, ZenanalyzeDiag, classify_image_type, classify_image_type_rgb8_diag,
    decide_bucket_stable,
};

/// Tuned experimental threshold rule (matches `decide_bucket_from_diag`
/// after Task-2 calibration but kept in the harness so we can iterate
/// on it without rebuilding the lib). When this rule converges, copy
/// it back into `classifier.rs`.
fn decide_tuned(diag: &ZenanalyzeDiag, w: u32, h: u32) -> ImageContentType {
    if w <= 128 && h <= 128 {
        return ImageContentType::Icon;
    }
    // Strong drawing signals first.
    if diag.line_art_score > 0.5 {
        return ImageContentType::Drawing;
    }
    // Soft screen-content / text — use >= so the corpus's 0.6000
    // qoi-benchmark websites count.
    if diag.screen_content >= 0.60 || diag.text_likelihood >= 0.55 {
        return ImageContentType::Drawing;
    }
    // Combined screen-content + flat/uniform: catches anti-aliased
    // mid-gradient screen content where screen sits at 0.4-0.6 and
    // the page is dominated by flat blocks.
    if diag.screen_content >= 0.40
        && diag.flat_color_block_ratio >= 0.40
        && diag.uniformity >= 0.85
        && diag.distinct_color_bins < 4096
    {
        return ImageContentType::Drawing;
    }
    // Palette-friendly fallback: tightened `flat` threshold so smooth
    // portraits and uniform photos no longer trip it. Real UI / chart
    // content sits at flat>=0.50; photos cap at ~0.45.
    if diag.flat_color_block_ratio >= 0.50 && diag.distinct_color_bins < 4096 {
        return ImageContentType::Drawing;
    }
    // Fits-in-256-colours is a strong indicator only when paired with
    // low natural likelihood (rules out flat photos / night scenes
    // with tiny palettes that GPT still labels "photo").
    if diag.palette_fits_in_256 && diag.natural_likelihood < 0.10 && diag.screen_content >= 0.50 {
        return ImageContentType::Drawing;
    }
    ImageContentType::Photo
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Truth {
    Photo,
    Drawing,
    Icon,
}

fn map_gpt_category(s: &str) -> Option<Truth> {
    match s {
        "photo_natural" | "photo_detailed" | "photo_portrait" | "photo_uniform" => {
            Some(Truth::Photo)
        }
        "screen_ui" | "screen_document" | "screen_chart" | "illustration" => Some(Truth::Drawing),
        _ => None,
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

fn load_image(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let ext = path.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png") => load_png(path),
        Some("jpg") | Some("jpeg") => load_jpeg(path),
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
        png::ColorType::Grayscale => {
            buf[..info.buffer_size()].iter().flat_map(|&g| [g, g, g]).collect()
        }
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        _ => return None,
    };
    Some((rgb, w, h))
}

fn load_jpeg(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    // Use the `image` crate via the dev-dep tree if available; else fall
    // back to skipping. zenwebp dev-deps don't include a JPEG decoder
    // by default, so we inline a zune-jpeg dep. Actually, we have
    // mozjpeg available as a sibling crate. The simplest portable path:
    // try `image` from the workspace.
    //
    // We don't want to add a JPEG dep to the harness — only one corpus
    // (imageflow) has JPGs, and they're not labeled in counts that
    // matter. Skip and report.
    let _ = path;
    None
}

fn rgb_to_y(rgb: &[u8]) -> Vec<u8> {
    let mut y = Vec::with_capacity(rgb.len() / 3);
    for px in rgb.chunks_exact(3) {
        let yv =
            ((u32::from(px[0]) * 76 + u32::from(px[1]) * 150 + u32::from(px[2]) * 30) >> 8) as u8;
        y.push(yv);
    }
    y
}

fn y_histogram(y: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &v in y {
        h[v as usize] += 1;
    }
    h
}

fn _bucket_to_str(b: ImageContentType) -> &'static str {
    match b {
        ImageContentType::Photo => "Photo",
        ImageContentType::Drawing => "Drawing",
        ImageContentType::Text => "Text",
        ImageContentType::Icon => "Icon",
        _ => "?",
    }
}

fn truth_to_str(t: Truth) -> &'static str {
    match t {
        Truth::Photo => "Photo",
        Truth::Drawing => "Drawing",
        Truth::Icon => "Icon",
    }
}

fn bucket_to_truth(b: ImageContentType) -> Truth {
    match b {
        ImageContentType::Photo => Truth::Photo,
        ImageContentType::Drawing | ImageContentType::Text => Truth::Drawing,
        ImageContentType::Icon => Truth::Icon,
        _ => Truth::Photo, // future-proof for #[non_exhaustive]
    }
}

#[derive(Default)]
struct Confusion {
    // [predicted][truth] for the three buckets we care about
    counts: BTreeMap<(Truth, Truth), u32>,
    total: u32,
    correct: u32,
}

impl Confusion {
    fn record(&mut self, pred: Truth, truth: Truth) {
        *self.counts.entry((pred, truth)).or_insert(0) += 1;
        self.total += 1;
        if pred == truth {
            self.correct += 1;
        }
    }
    fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }
    fn print(&self, name: &str) {
        println!("\n=== {name} (n={}, accuracy={:.3}) ===", self.total, self.accuracy());
        let truths = [Truth::Photo, Truth::Drawing, Truth::Icon];
        print!("            ");
        for t in &truths {
            print!("{:>10}", truth_to_str(*t));
        }
        println!("   <- truth");
        for p in &truths {
            print!("pred {:>7}", truth_to_str(*p));
            for t in &truths {
                let n = self.counts.get(&(*p, *t)).copied().unwrap_or(0);
                print!("{:>10}", n);
            }
            println!();
        }
        // Recall per truth
        for t in &truths {
            let total_t: u32 =
                truths.iter().map(|p| self.counts.get(&(*p, *t)).copied().unwrap_or(0)).sum();
            let correct_t = self.counts.get(&(*t, *t)).copied().unwrap_or(0);
            if total_t > 0 {
                println!(
                    "  recall {:>8}: {} / {} = {:.3}",
                    truth_to_str(*t),
                    correct_t,
                    total_t,
                    correct_t as f32 / total_t as f32
                );
            }
        }
    }
}

#[derive(Default)]
struct PerCorpus {
    by_corpus: BTreeMap<String, Confusion>,
}
impl PerCorpus {
    fn record(&mut self, corpus: &str, pred: Truth, truth: Truth) {
        self.by_corpus.entry(corpus.to_string()).or_default().record(pred, truth);
    }
    fn print(&self, name: &str) {
        println!("\n--- {name}: per-corpus accuracy ---");
        for (k, v) in &self.by_corpus {
            println!("  {:<18} n={:>3}  acc={:.3}", k, v.total, v.accuracy());
        }
    }
}

#[derive(Debug, Clone)]
struct Row {
    corpus: String,
    image: String,
    primary_category: String,
}

fn parse_labels(path: &Path) -> std::io::Result<Vec<Row>> {
    let f = fs::File::open(path)?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    let mut header_seen = false;
    for line in r.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        out.push(Row {
            corpus: cols[0].to_string(),
            image: cols[1].to_string(),
            primary_category: cols[4].to_string(),
        });
    }
    Ok(out)
}

fn main() {
    let argv: Vec<String> = env::args().skip(1).collect();
    let labels_path = argv.first().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/lilith/work/coefficient/benchmarks/classifier-eval/labels.tsv")
    });
    let rows = match parse_labels(&labels_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error reading labels {}: {e}", labels_path.display());
            std::process::exit(2);
        }
    };
    eprintln!("loaded {} rows from {}", rows.len(), labels_path.display());

    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let out_dir = PathBuf::from("/mnt/v/output/zenwebp/zenanalyze-validate");
    let _ = fs::create_dir_all(&out_dir);
    let out_tsv = out_dir.join(format!("run_{ts}.tsv"));
    let mut tsv = BufWriter::new(fs::File::create(&out_tsv).expect("create tsv"));
    writeln!(
        tsv,
        "corpus\timage\tw\th\ttruth\told\tstable\texperimental\ttuned\t\
         screen\ttext\tnatural\tflat\tdistinct\tvariance\tedge\tunif\thfreq\t\
         palette_fits_256\tpalette_width\tline_art\tskin_tone\tedge_slope_stdev\tprimary_category"
    )
    .unwrap();

    let mut conf_old = Confusion::default();
    let mut conf_stable = Confusion::default();
    let mut conf_exp = Confusion::default();
    let mut conf_tuned = Confusion::default();
    let mut per_corpus_exp = PerCorpus::default();
    let mut per_corpus_tuned = PerCorpus::default();

    let mut skipped_missing = 0u32;
    let mut skipped_decode = 0u32;
    let mut skipped_unmapped = 0u32;
    let mut total = 0u32;

    // Collect misclassifications for tuning analysis
    let mut misclass: Vec<(Row, Truth, Truth, ZenanalyzeDiag)> = Vec::new();

    for row in &rows {
        let Some(truth) = map_gpt_category(&row.primary_category) else {
            skipped_unmapped += 1;
            continue;
        };
        let Some(root) = corpus_root(&row.corpus) else {
            eprintln!("warn: unknown corpus {}", row.corpus);
            skipped_missing += 1;
            continue;
        };
        let path = Path::new(root).join(&row.image);
        if !path.exists() {
            skipped_missing += 1;
            continue;
        }
        let Some((rgb, w, h)) = load_image(&path) else {
            skipped_decode += 1;
            continue;
        };
        if rgb.len() != (w as usize) * (h as usize) * 3 {
            skipped_decode += 1;
            continue;
        }

        // Old classifier
        let y = rgb_to_y(&rgb);
        let yh = y_histogram(&y);
        let old = bucket_to_truth(classify_image_type(
            &y,
            w as usize,
            h as usize,
            w as usize,
            &yh,
        ));

        // New (with experimental + stable diag)
        let (exp_bucket, diag) = classify_image_type_rgb8_diag(&rgb, w, h);
        let exp = bucket_to_truth(exp_bucket);
        let stable = bucket_to_truth(if w <= 128 && h <= 128 {
            ImageContentType::Icon
        } else {
            decide_bucket_stable(&diag)
        });

        let tuned = bucket_to_truth(decide_tuned(&diag, w, h));

        conf_old.record(old, truth);
        conf_stable.record(stable, truth);
        conf_exp.record(exp, truth);
        conf_tuned.record(tuned, truth);
        per_corpus_exp.record(&row.corpus, exp, truth);
        per_corpus_tuned.record(&row.corpus, tuned, truth);
        total += 1;

        if tuned != truth {
            misclass.push((row.clone(), tuned, truth, diag));
        }

        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
             {:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t\
             {}\t{}\t{:.4}\t{:.4}\t{:.2}\t{}",
            row.corpus,
            row.image,
            w,
            h,
            truth_to_str(truth),
            truth_to_str(old),
            truth_to_str(stable),
            truth_to_str(exp),
            truth_to_str(tuned),
            diag.screen_content,
            diag.text_likelihood,
            diag.natural_likelihood,
            diag.flat_color_block_ratio,
            diag.distinct_color_bins,
            diag.variance,
            diag.edge_density,
            diag.uniformity,
            diag.high_freq_energy_ratio,
            diag.palette_fits_in_256,
            diag.indexed_palette_width,
            diag.line_art_score,
            diag.skin_tone_fraction,
            diag.edge_slope_stdev,
            row.primary_category,
        )
        .unwrap();
    }
    drop(tsv);

    println!(
        "\n=== summary ===\nrows in labels:     {}\nclassified:         {}\nskipped (missing):  {}\nskipped (decode):   {}\nskipped (unmapped): {}\nper-image TSV:      {}",
        rows.len(),
        total,
        skipped_missing,
        skipped_decode,
        skipped_unmapped,
        out_tsv.display(),
    );

    conf_old.print("OLD (alpha-hist + Y-plane)");
    conf_stable.print("STABLE (zenanalyze, no experimental)");
    conf_exp.print("EXPERIMENTAL (zenanalyze + palette_fits / line_art)");
    conf_tuned.print("TUNED (experimental + harness-tuned thresholds)");
    per_corpus_exp.print("EXPERIMENTAL");
    per_corpus_tuned.print("TUNED");

    // Misclassification breakdown — focus on photo->drawing (which
    // is the cost-regression direction for users).
    let mut p_to_d: Vec<&(Row, Truth, Truth, ZenanalyzeDiag)> = misclass
        .iter()
        .filter(|(_, p, t, _)| *t == Truth::Photo && *p == Truth::Drawing)
        .collect();
    let mut d_to_p: Vec<&(Row, Truth, Truth, ZenanalyzeDiag)> = misclass
        .iter()
        .filter(|(_, p, t, _)| *t == Truth::Drawing && *p == Truth::Photo)
        .collect();
    p_to_d.sort_by(|a, b| {
        b.3.line_art_score
            .partial_cmp(&a.3.line_art_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    d_to_p.sort_by(|a, b| {
        b.3.distinct_color_bins.cmp(&a.3.distinct_color_bins)
    });

    println!("\n--- TUNED: photo->drawing misclassifications ({}) ---", p_to_d.len());
    println!("       (sorted by line_art_score desc; truth=Photo, predicted Drawing)");
    println!(
        "{:<14} {:<40} {:<18} screen text natural flat dist line_art skin slopeSD pal256",
        "corpus", "image", "category"
    );
    for (r, _p, _t, d) in p_to_d.iter().take(40) {
        println!(
            "{:<14} {:<40} {:<18} {:.2} {:.2} {:.2}    {:.2} {:>5} {:.2}     {:.3} {:>6.2}  {}",
            r.corpus,
            r.image,
            r.primary_category,
            d.screen_content,
            d.text_likelihood,
            d.natural_likelihood,
            d.flat_color_block_ratio,
            d.distinct_color_bins,
            d.line_art_score,
            d.skin_tone_fraction,
            d.edge_slope_stdev,
            d.palette_fits_in_256 as u8,
        );
    }

    println!("\n--- TUNED: drawing->photo misclassifications ({}) ---", d_to_p.len());
    println!("       (sorted by distinct_color_bins desc; truth=Drawing, predicted Photo)");
    println!(
        "{:<14} {:<40} {:<18} screen text natural flat dist line_art skin slopeSD pal256",
        "corpus", "image", "category"
    );
    for (r, _p, _t, d) in d_to_p.iter().take(40) {
        println!(
            "{:<14} {:<40} {:<18} {:.2} {:.2} {:.2}    {:.2} {:>5} {:.2}     {:.3} {:>6.2}  {}",
            r.corpus,
            r.image,
            r.primary_category,
            d.screen_content,
            d.text_likelihood,
            d.natural_likelihood,
            d.flat_color_block_ratio,
            d.distinct_color_bins,
            d.line_art_score,
            d.skin_tone_fraction,
            d.edge_slope_stdev,
            d.palette_fits_in_256 as u8,
        );
    }
}
