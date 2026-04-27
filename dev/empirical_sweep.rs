//! Empirical comparison of zenwebp vs libwebp encoder file sizes.
//!
//! Sweeps configs (preset, quality, method) across a corpus and writes
//! per-file TSV results plus a summary report.
//!
//! Usage: cargo run --release --example empirical_sweep -- <out_tsv> <Q_LIST> <M_LIST> <P_LIST> <SAMPLE> <corpus_label:dir>...
//! Q_LIST: comma quality list e.g. "5,25,50,75,90,95"
//! M_LIST: comma method list e.g. "0,4,6"
//! P_LIST: comma preset list (subset of: Default,Photo,Drawing,Auto)
//! SAMPLE: max images per corpus; 0 = all
//!
//! Decodes PNGs once per file, then runs every (preset, quality, method) cell.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

#[derive(Clone, Copy)]
struct PresetPair {
    name: &'static str,
    zen: Preset,
    lib: Option<webpx::Preset>, // None = "Auto" (zen-only, libwebp uses Default)
}

const ALL_PRESETS: &[PresetPair] = &[
    PresetPair {
        name: "Default",
        zen: Preset::Default,
        lib: Some(webpx::Preset::Default),
    },
    PresetPair {
        name: "Photo",
        zen: Preset::Photo,
        lib: Some(webpx::Preset::Photo),
    },
    PresetPair {
        name: "Drawing",
        zen: Preset::Drawing,
        lib: Some(webpx::Preset::Drawing),
    },
    PresetPair {
        name: "Auto",
        zen: Preset::Auto,
        lib: None,
    },
];

fn pick_presets(names: &str) -> Vec<PresetPair> {
    let want: Vec<&str> = names.split(',').collect();
    want.iter()
        .filter_map(|n| {
            ALL_PRESETS
                .iter()
                .find(|p| p.name.eq_ignore_ascii_case(n.trim()))
                .copied()
        })
        .collect()
}

struct Image {
    corpus: String,
    name: String,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() * 3 / 4);
            for chunk in rgba.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(buf.len() * 3);
            for &g in &buf[..info.buffer_size()] {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity(buf.len() * 3 / 2);
            for chunk in buf[..info.buffer_size()].chunks(2) {
                rgb.extend_from_slice(&[chunk[0], chunk[0], chunk[0]]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: empirical_sweep <out_tsv> <Q_LIST> <M_LIST> <P_LIST> <SAMPLE> <label:dir>..."
        );
        std::process::exit(2);
    }
    let out_path = PathBuf::from(&args[1]);
    let qualities: Vec<f32> = args[2]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let methods: Vec<u8> = args[3]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let presets = pick_presets(&args[4]);
    let sample: usize = args[5].trim().parse().unwrap();
    let mut corpora: Vec<(String, PathBuf)> = Vec::new();
    for spec in &args[6..] {
        let (label, dir) = spec.split_once(':').expect("label:dir");
        corpora.push((label.to_string(), PathBuf::from(dir)));
    }

    eprintln!("Loading corpora...");
    let mut images: Vec<Image> = Vec::new();
    for (label, dir) in &corpora {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap_or_else(|_| panic!("read_dir {}", dir.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        entries.sort();
        // Deterministic sample: stride through to pick `sample` evenly
        if sample > 0 && entries.len() > sample {
            let stride = entries.len() as f64 / sample as f64;
            let mut picked = Vec::with_capacity(sample);
            for i in 0..sample {
                let idx = (i as f64 * stride) as usize;
                picked.push(entries[idx.min(entries.len() - 1)].clone());
            }
            entries = picked;
        }
        for path in entries {
            if let Some((rgb, w, h)) = load_png(&path) {
                images.push(Image {
                    corpus: label.clone(),
                    name: path.file_name().unwrap().to_string_lossy().into_owned(),
                    rgb,
                    w,
                    h,
                });
            }
        }
        eprintln!(
            "  {} -> {} images loaded so far",
            label,
            images.iter().filter(|i| i.corpus == *label).count()
        );
    }
    eprintln!("Total images loaded: {}", images.len());

    let total_cells = presets.len() * qualities.len() * methods.len();
    let total_jobs = total_cells * images.len();
    eprintln!(
        "Sweep: {} presets x {} qualities x {} methods x {} images = {} encode pairs",
        presets.len(),
        qualities.len(),
        methods.len(),
        images.len(),
        total_jobs
    );

    let out = Mutex::new(fs::File::create(&out_path).expect("create tsv"));
    {
        let mut g = out.lock().unwrap();
        writeln!(
            g,
            "corpus\tfile\twidth\theight\tpreset\tquality\tmethod\tzen_bytes\tlib_bytes\tratio"
        )
        .unwrap();
    }

    let start = Instant::now();
    let mut done_cells = 0usize;

    for pp in &presets {
        for &q in &qualities {
            for &m in &methods {
                let cell_start = Instant::now();
                let mut zen_total = 0u64;
                let mut lib_total = 0u64;
                let mut zen_failures = 0usize;
                let mut lib_failures = 0usize;
                let mut rows: Vec<String> = Vec::with_capacity(images.len());

                for img in &images {
                    let cfg = EncoderConfig::with_preset(pp.zen, q).with_method(m);
                    let zen =
                        match EncodeRequest::new(&cfg, &img.rgb, PixelLayout::Rgb8, img.w, img.h)
                            .encode()
                        {
                            Ok(v) => v,
                            Err(_) => {
                                zen_failures += 1;
                                continue;
                            }
                        };
                    let lib_preset = pp.lib.unwrap_or(webpx::Preset::Default);
                    let lib = match webpx::EncoderConfig::with_preset(lib_preset, q)
                        .method(m)
                        .encode_rgb(&img.rgb, img.w, img.h, webpx::Unstoppable)
                    {
                        Ok(v) => v,
                        Err(_) => {
                            lib_failures += 1;
                            continue;
                        }
                    };
                    let zb = zen.len() as u64;
                    let lb = lib.len() as u64;
                    zen_total += zb;
                    lib_total += lb;
                    let ratio = zb as f64 / lb.max(1) as f64;
                    rows.push(format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
                        img.corpus, img.name, img.w, img.h, pp.name, q, m, zb, lb, ratio
                    ));
                }

                {
                    let mut g = out.lock().unwrap();
                    for r in &rows {
                        writeln!(g, "{}", r).unwrap();
                    }
                }
                done_cells += 1;
                let agg = if lib_total > 0 {
                    zen_total as f64 / lib_total as f64
                } else {
                    f64::NAN
                };
                let elapsed = cell_start.elapsed().as_secs_f64();
                let total_elapsed = start.elapsed().as_secs_f64();
                let est_total = total_elapsed * (total_cells as f64 / done_cells as f64);
                eprintln!(
                    "[{:>3}/{:>3}] {:>7} q={:>4.0} m={} -> agg ratio={:.4}x  ({} ok, zen_fail={}, lib_fail={})  cell={:.1}s  total={:.1}s/{:.1}s",
                    done_cells,
                    total_cells,
                    pp.name,
                    q,
                    m,
                    agg,
                    rows.len(),
                    zen_failures,
                    lib_failures,
                    elapsed,
                    total_elapsed,
                    est_total
                );
            }
        }
    }

    eprintln!(
        "Done in {:.1}s. Wrote {}",
        start.elapsed().as_secs_f64(),
        out_path.display()
    );
}
