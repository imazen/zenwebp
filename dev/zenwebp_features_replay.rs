//! Feature-only replay against an existing pareto features TSV or
//! against a corpus manifest TSV.
//!
//! Two input modes:
//!
//! 1. `--input <features.tsv>` — reads (image_path, size_class, width,
//!    height) tuples from the first 4 columns of an existing features
//!    TSV (header row required). Loads each source file (PNG only),
//!    resizes to recorded (width, height) using zenresize Mitchell,
//!    writes new features TSV with current zenanalyze schema.
//!
//! 2. `--manifest <manifest.tsv> --root <DIR>` — reads
//!    (relative_path, bytes, width, height, axis_class, ...) rows
//!    from a curated corpus manifest. Loads each file at
//!    `<root>/<relative_path>` at its NATIVE resolution (no resize)
//!    and writes the features TSV with `axis_class` as the
//!    `size_class` column.
//!
//! Decoders: PNG via zenpng. JPEG via jpeg-decoder. AVIF unsupported
//! (skipped with warning) — only used by HDR class which already
//! cannot surface HDR signal through rgb8.
//!
//! Used to regenerate features for a frozen image set when the
//! analyzer's feature columns have changed but the encode TSV is
//! reused as-is (the cull only moved feature column ids, not encode
//! results). This is cheaper and more deterministic than rerunning
//! the full corpus walk in `zenwebp_pareto --features-only`.
//!
//! Usage:
//!   # Replay against existing features TSV:
//!   cargo run --release --features analyzer --example zenwebp_features_replay -- \
//!     --input  benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv \
//!     --output benchmarks/zenwebp_pareto_features_2026-05-01_post_cull.tsv
//!
//!   # Replay from a manifest TSV:
//!   cargo run --release --features analyzer --example zenwebp_features_replay -- \
//!     --manifest /mnt/v/output/codec-corpus-2026-05-01-multiaxis/manifest.tsv \
//!     --root     /mnt/v/output/codec-corpus-2026-05-01-multiaxis \
//!     --output   benchmarks/zenwebp_features_expanded_2026-05-02.tsv

#![forbid(unsafe_code)]

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use rayon::prelude::*;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

fn full_feature_set() -> FeatureSet {
    FeatureSet::SUPPORTED
}

fn feature_columns() -> Vec<AnalysisFeature> {
    full_feature_set().iter().collect()
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

fn load_png_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    use enough::Unstoppable;
    use zenpixels_convert::PixelBufferConvertTypedExt;
    use zenpng::PngDecodeConfig;

    let bytes = std::fs::read(path).ok()?;
    let output = zenpng::decode(&bytes, &PngDecodeConfig::default(), &Unstoppable).ok()?;
    let w = output.info.width;
    let h = output.info.height;

    let rgb_buf = output.pixels.to_rgb8();
    let slice = rgb_buf.as_slice();
    if let Some(contiguous) = slice.as_contiguous_bytes() {
        return Some((contiguous.to_vec(), w, h));
    }
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        let row = slice.row(y);
        out.extend_from_slice(&row[..(w as usize) * 3]);
    }
    Some((out, w, h))
}

fn load_jpeg_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(path).ok()?;
    let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let pixels = dec.decode().ok()?;
    let info = dec.info()?;
    let w = info.width as u32;
    let h = info.height as u32;
    use jpeg_decoder::PixelFormat;
    match info.pixel_format {
        PixelFormat::RGB24 => Some((pixels, w, h)),
        PixelFormat::L8 => {
            // Promote grayscale to RGB by triplicating.
            let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
            for &v in &pixels {
                out.push(v);
                out.push(v);
                out.push(v);
            }
            Some((out, w, h))
        }
        PixelFormat::CMYK32 => {
            // Naive CMYK→RGB; HDR-class JPEGs in the multiaxis corpus
            // are not CMYK in practice, so this branch is best-effort.
            let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
            for px in pixels.chunks_exact(4) {
                let (c, m, y, k) = (px[0], px[1], px[2], px[3]);
                let r = ((255 - c as i32) * (255 - k as i32) / 255).max(0).min(255) as u8;
                let g = ((255 - m as i32) * (255 - k as i32) / 255).max(0).min(255) as u8;
                let b = ((255 - y as i32) * (255 - k as i32) / 255).max(0).min(255) as u8;
                out.push(r);
                out.push(g);
                out.push(b);
            }
            Some((out, w, h))
        }
        _ => None,
    }
}

fn load_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" => load_png_rgb8(path),
        "jpg" | "jpeg" => load_jpeg_rgb8(path),
        "avif" => None, // unsupported in this binary; skipped with warning
        _ => None,
    }
}

/// Resize to *exact* (new_w, new_h) using Mitchell. If already matching,
/// returns a clone of the input.
fn resize_exact(rgb: &[u8], w: u32, h: u32, new_w: u32, new_h: u32) -> Vec<u8> {
    if w == new_w && h == new_h {
        return rgb.to_vec();
    }
    let cfg = zenresize::ResizeConfig::builder(w, h, new_w, new_h)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .filter(zenresize::Filter::Mitchell)
        .srgb()
        .build();
    zenresize::Resizer::new(&cfg).resize(rgb)
}

#[derive(Default)]
struct Args {
    input: Option<PathBuf>,
    manifest: Option<PathBuf>,
    root: Option<PathBuf>,
    output: Option<PathBuf>,
    /// If true, do NOT resize; use native (width,height) from the manifest.
    /// Default true for --manifest mode, false for --input mode.
    native: bool,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" => a.input = Some(PathBuf::from(it.next().expect("--input PATH"))),
            "--manifest" => a.manifest = Some(PathBuf::from(it.next().expect("--manifest PATH"))),
            "--root" => a.root = Some(PathBuf::from(it.next().expect("--root PATH"))),
            "--output" => a.output = Some(PathBuf::from(it.next().expect("--output PATH"))),
            "--native" => a.native = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if a.manifest.is_some() {
        a.native = true;
    }
    if a.input.is_none() && a.manifest.is_none() {
        panic!("must pass --input <features.tsv> or --manifest <manifest.tsv>");
    }
    if a.manifest.is_some() && a.root.is_none() {
        panic!("--manifest requires --root <DIR>");
    }
    if a.output.is_none() {
        panic!("missing --output");
    }
    a
}

#[derive(Clone, Debug)]
struct Cell {
    image_path: PathBuf, // resolved on disk
    label_path: String,  // logical label written to TSV
    size_class: String,
    width: u32,
    height: u32,
}

fn read_cells_from_features_tsv(input: &PathBuf) -> Vec<Cell> {
    let f = std::fs::File::open(input).expect("open input TSV");
    let reader = BufReader::new(f);
    let mut cells = Vec::new();
    let mut header = true;
    for line in reader.lines() {
        let line = line.expect("read line");
        if header {
            let cols: Vec<&str> = line.split('\t').collect();
            assert!(
                cols.len() >= 4
                    && cols[0] == "image_path"
                    && cols[1] == "size_class"
                    && cols[2] == "width"
                    && cols[3] == "height",
                "input TSV header must start with image_path\\tsize_class\\twidth\\theight"
            );
            header = false;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            continue;
        }
        cells.push(Cell {
            image_path: PathBuf::from(cols[0]),
            label_path: cols[0].to_string(),
            size_class: cols[1].to_string(),
            width: cols[2].parse().expect("width"),
            height: cols[3].parse().expect("height"),
        });
    }
    cells
}

fn read_cells_from_manifest(manifest: &PathBuf, root: &PathBuf) -> Vec<Cell> {
    let f = std::fs::File::open(manifest).expect("open manifest TSV");
    let reader = BufReader::new(f);
    let mut cells = Vec::new();
    let mut col_relpath = 0usize;
    let mut col_w = 2usize;
    let mut col_h = 3usize;
    let mut col_axis = 4usize;
    let mut header = true;
    for line in reader.lines() {
        let line = line.expect("read line");
        if header {
            let cols: Vec<&str> = line.split('\t').collect();
            for (i, c) in cols.iter().enumerate() {
                match *c {
                    "relative_path" => col_relpath = i,
                    "width" => col_w = i,
                    "height" => col_h = i,
                    "axis_class" => col_axis = i,
                    _ => {}
                }
            }
            header = false;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() <= col_axis {
            continue;
        }
        let relpath = cols[col_relpath];
        let w: u32 = match cols[col_w].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let h: u32 = match cols[col_h].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        cells.push(Cell {
            image_path: root.join(relpath),
            label_path: relpath.to_string(),
            size_class: cols[col_axis].to_string(),
            width: w,
            height: h,
        });
    }
    cells
}

fn main() {
    let args = parse_args();
    let output = args.output.clone().unwrap();
    eprintln!("[features_replay] output: {}", output.display());

    let cells = if let Some(input) = &args.input {
        eprintln!(
            "[features_replay] mode: features-tsv input={}",
            input.display()
        );
        read_cells_from_features_tsv(input)
    } else {
        let manifest = args.manifest.as_ref().unwrap();
        let root = args.root.as_ref().unwrap();
        eprintln!(
            "[features_replay] mode: manifest input={} root={} (native={})",
            manifest.display(),
            root.display(),
            args.native,
        );
        read_cells_from_manifest(manifest, root)
    };
    eprintln!("[features_replay] {} cells to process", cells.len());

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if output.exists() {
        eprintln!(
            "[features_replay] WARNING: output exists, will be appended to: {}",
            output.display()
        );
    }
    let feat_is_new = !output.exists();
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output)
        .expect("open output");
    let feat_file = Mutex::new(f);

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
    let total = cells.len();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let native_mode = args.native;

    // Group cells by on-disk image_path so we only decode each file once.
    use std::collections::BTreeMap;
    let mut by_img: BTreeMap<PathBuf, Vec<Cell>> = BTreeMap::new();
    for c in cells {
        by_img.entry(c.image_path.clone()).or_default().push(c);
    }
    let groups: Vec<(PathBuf, Vec<Cell>)> = by_img.into_iter().collect();

    groups.par_iter().for_each(|(path, cells_for_img)| {
        let (rgb_native, w_native, h_native) = match load_rgb8(path) {
            Some(t) => t,
            None => {
                eprintln!("  load failed: {}", path.display());
                failed.fetch_add(cells_for_img.len(), std::sync::atomic::Ordering::Relaxed);
                done.fetch_add(cells_for_img.len(), std::sync::atomic::Ordering::Relaxed);
                return;
            }
        };
        for cell in cells_for_img {
            let (rgb, w_use, h_use) = if native_mode {
                (rgb_native.clone(), w_native, h_native)
            } else {
                (
                    resize_exact(&rgb_native, w_native, h_native, cell.width, cell.height),
                    cell.width,
                    cell.height,
                )
            };
            let analysis = analyze_features_rgb8(&rgb, w_use, h_use, &query);
            {
                let mut f = feat_file.lock().unwrap();
                write!(
                    f,
                    "{}\t{}\t{}\t{}",
                    cell.label_path, cell.size_class, w_use, h_use
                )
                .ok();
                for c in &cols {
                    write!(f, "\t{}", feature_value_str(&analysis, *c)).ok();
                }
                writeln!(f).ok();
                f.flush().ok();
            }
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 32 == 0 || n == total {
                let dt = started.elapsed().as_secs_f64();
                let rate = n as f64 / dt;
                let eta = (total - n) as f64 / rate.max(1e-6);
                eprintln!(
                    "  progress: {}/{}  ({:.1}/sec, ETA {:.0}s)",
                    n, total, rate, eta
                );
            }
        }
    });

    eprintln!(
        "[features_replay] done in {:.1}s. wrote {} (failed: {})",
        started.elapsed().as_secs_f64(),
        output.display(),
        failed.load(std::sync::atomic::Ordering::Relaxed),
    );
}
