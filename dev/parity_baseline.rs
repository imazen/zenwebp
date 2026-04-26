//! Combined size + quality + speed parity measurement vs libwebp.
//!
//! Measures, per (file, preset, quality, method) cell:
//!   - encoded size (zenwebp vs libwebp)
//!   - encode wall-clock (median of 3 runs)
//!   - butteraugli (lower = better)
//!   - SSIMULACRA2 (higher = better)
//!
//! Both encoded outputs are decoded with zenwebp's decoder so the metrics
//! reflect encoder quality only, not decoder differences.
//!
//! Usage:
//!   cargo run --release --example parity_baseline -- \
//!       <out_tsv> <Q_LIST> <M_LIST> <P_LIST> <SAMPLE> <label:dir>...
//!
//! Q_LIST / M_LIST / P_LIST: comma lists (e.g. "25,50,75,90", "0,4,6", "Default,Photo")
//! SAMPLE: max images per corpus (deterministic stride sample); 0 = all
//!
//! Output TSV columns:
//!   corpus, file, width, height, preset, quality, method,
//!   zen_bytes, lib_bytes, size_ratio,
//!   zen_encode_ms, lib_encode_ms, speed_ratio,
//!   zen_butter, lib_butter, butter_delta,
//!   zen_ssim2, lib_ssim2, ssim2_delta

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli};
use fast_ssim2::{ColorPrimaries, Rgb as SsimRgb, TransferCharacteristic, compute_frame_ssimulacra2};
use imgref::Img;
use rgb::RGB8;

use zenwebp::decoder::decode_rgb;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

#[derive(Clone, Copy)]
struct PresetPair {
    name: &'static str,
    zen: Preset,
    lib: Option<webpx::Preset>, // None = "Auto" (zen-only); libwebp uses Default
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
    names
        .split(',')
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

/// Median of three encode wall-clocks (ms). Returns the encoded bytes from the
/// last run plus the median time. Falls back to a single run if any encode
/// fails.
fn time_encode_zen(
    cfg: &EncoderConfig,
    rgb: &[u8],
    w: u32,
    h: u32,
    runs: usize,
) -> Option<(Vec<u8>, f64)> {
    let mut times = Vec::with_capacity(runs);
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let out = EncodeRequest::new(cfg, rgb, PixelLayout::Rgb8, w, h)
            .encode()
            .ok()?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(dt);
        last = Some(out);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    Some((last?, median))
}

fn time_encode_lib(
    preset: webpx::Preset,
    quality: f32,
    method: u8,
    rgb: &[u8],
    w: u32,
    h: u32,
    runs: usize,
) -> Option<(Vec<u8>, f64)> {
    let mut times = Vec::with_capacity(runs);
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let out = webpx::EncoderConfig::with_preset(preset, quality)
            .method(method)
            .encode_rgb(rgb, w, h, webpx::Unstoppable)
            .ok()?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(dt);
        last = Some(out);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    Some((last?, median))
}

fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// SSIMULACRA2 score (higher = better). Returns NaN on failure.
fn compute_ssim2(orig: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;
    let orig_rgb: Vec<[f32; 3]> = orig
        .chunks_exact(3)
        .map(|p| [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])])
        .collect();
    let dec_rgb: Vec<[f32; 3]> = decoded
        .chunks_exact(3)
        .map(|p| [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])])
        .collect();
    let orig_img = match SsimRgb::new(
        orig_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    ) {
        Ok(i) => i,
        Err(_) => return f64::NAN,
    };
    let dec_img = match SsimRgb::new(
        dec_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    ) {
        Ok(i) => i,
        Err(_) => return f64::NAN,
    };
    compute_frame_ssimulacra2(orig_img, dec_img).unwrap_or(f64::NAN)
}

/// Butteraugli score (lower = better). Returns NaN on failure.
fn compute_butter(orig: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;
    let src: Vec<RGB8> = orig.chunks_exact(3).map(|c| RGB8::new(c[0], c[1], c[2])).collect();
    let dst: Vec<RGB8> = decoded
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();
    if src.len() != w * h || dst.len() != w * h {
        return f64::NAN;
    }
    let params = ButteraugliParams::default();
    match butteraugli(Img::new(src, w, h).as_ref(), Img::new(dst, w, h).as_ref(), &params) {
        Ok(r) => r.score as f64,
        Err(_) => f64::NAN,
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: parity_baseline <out_tsv> <Q_LIST> <M_LIST> <P_LIST> <SAMPLE> <label:dir>..."
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
        if sample > 0 && entries.len() > sample {
            let stride = entries.len() as f64 / sample as f64;
            let mut picked = Vec::with_capacity(sample);
            for i in 0..sample {
                let idx = (i as f64 * stride) as usize;
                picked.push(entries[idx.min(entries.len() - 1)].clone());
            }
            entries = picked;
        }
        let before = images.len();
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
            "  {} -> {} images loaded",
            label,
            images.len() - before
        );
    }
    eprintln!("Total images loaded: {}", images.len());

    // Speed-tier discipline: median-of-3 for small images, single run for
    // larger ones to keep total wall-clock reasonable. Threshold 1MP.
    fn runs_for(w: u32, h: u32) -> usize {
        if (w as u64) * (h as u64) <= 1_000_000 { 3 } else { 1 }
    }

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
            "corpus\tfile\twidth\theight\tpreset\tquality\tmethod\t\
             zen_bytes\tlib_bytes\tsize_ratio\t\
             zen_encode_ms\tlib_encode_ms\tspeed_ratio\t\
             zen_butter\tlib_butter\tbutter_delta\t\
             zen_ssim2\tlib_ssim2\tssim2_delta"
        )
        .unwrap();
    }

    // Per-corpus aggregates.
    use std::collections::BTreeMap;
    #[derive(Default)]
    struct Agg {
        n: usize,
        zen_bytes: u64,
        lib_bytes: u64,
        zen_ms: f64,
        lib_ms: f64,
        butter_delta: f64,
        ssim2_delta: f64,
    }
    let mut agg: BTreeMap<String, Agg> = BTreeMap::new();
    let mut skipped = 0usize;

    let start = Instant::now();
    let mut done_cells = 0usize;

    for pp in &presets {
        for &q in &qualities {
            for &m in &methods {
                let cell_start = Instant::now();
                let mut rows: Vec<String> = Vec::with_capacity(images.len());
                let mut zen_total = 0u64;
                let mut lib_total = 0u64;
                let mut cell_skipped = 0usize;

                for img in &images {
                    let cfg = EncoderConfig::with_preset(pp.zen, q).with_method(m);
                    let runs = runs_for(img.w, img.h);

                    let (zen_bytes, zen_ms) =
                        match time_encode_zen(&cfg, &img.rgb, img.w, img.h, runs) {
                            Some(v) => v,
                            None => {
                                eprintln!(
                                    "skip zen-encode {} {} q{} m{} preset={}",
                                    img.corpus, img.name, q, m, pp.name
                                );
                                cell_skipped += 1;
                                skipped += 1;
                                continue;
                            }
                        };
                    let lib_preset = pp.lib.unwrap_or(webpx::Preset::Default);
                    let (lib_bytes, lib_ms) =
                        match time_encode_lib(lib_preset, q, m, &img.rgb, img.w, img.h, runs) {
                            Some(v) => v,
                            None => {
                                eprintln!(
                                    "skip lib-encode {} {} q{} m{}",
                                    img.corpus, img.name, q, m
                                );
                                cell_skipped += 1;
                                skipped += 1;
                                continue;
                            }
                        };

                    let zen_dec = match decode_rgb(&zen_bytes) {
                        Ok((d, _, _)) => d,
                        Err(_) => {
                            eprintln!(
                                "skip zen-decode {} {} q{} m{}",
                                img.corpus, img.name, q, m
                            );
                            cell_skipped += 1;
                            skipped += 1;
                            continue;
                        }
                    };
                    let lib_dec = match decode_rgb(&lib_bytes) {
                        Ok((d, _, _)) => d,
                        Err(_) => {
                            eprintln!(
                                "skip lib-decode {} {} q{} m{}",
                                img.corpus, img.name, q, m
                            );
                            cell_skipped += 1;
                            skipped += 1;
                            continue;
                        }
                    };

                    let zen_butter = compute_butter(&img.rgb, &zen_dec, img.w, img.h);
                    let lib_butter = compute_butter(&img.rgb, &lib_dec, img.w, img.h);
                    let zen_ssim2 = compute_ssim2(&img.rgb, &zen_dec, img.w, img.h);
                    let lib_ssim2 = compute_ssim2(&img.rgb, &lib_dec, img.w, img.h);

                    let zb = zen_bytes.len() as u64;
                    let lb = lib_bytes.len() as u64;
                    let size_ratio = zb as f64 / lb.max(1) as f64;
                    let speed_ratio = zen_ms / lib_ms.max(1e-6);
                    let butter_delta = zen_butter - lib_butter;
                    let ssim2_delta = zen_ssim2 - lib_ssim2;

                    zen_total += zb;
                    lib_total += lb;

                    let a = agg.entry(img.corpus.clone()).or_default();
                    a.n += 1;
                    a.zen_bytes += zb;
                    a.lib_bytes += lb;
                    a.zen_ms += zen_ms;
                    a.lib_ms += lib_ms;
                    if butter_delta.is_finite() {
                        a.butter_delta += butter_delta;
                    }
                    if ssim2_delta.is_finite() {
                        a.ssim2_delta += ssim2_delta;
                    }

                    rows.push(format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                        img.corpus,
                        img.name,
                        img.w,
                        img.h,
                        pp.name,
                        q,
                        m,
                        zb,
                        lb,
                        size_ratio,
                        zen_ms,
                        lib_ms,
                        speed_ratio,
                        zen_butter,
                        lib_butter,
                        butter_delta,
                        zen_ssim2,
                        lib_ssim2,
                        ssim2_delta
                    ));
                }

                {
                    let mut g = out.lock().unwrap();
                    for r in &rows {
                        writeln!(g, "{}", r).unwrap();
                    }
                    g.flush().ok();
                }
                done_cells += 1;
                let cell_ratio = if lib_total > 0 {
                    zen_total as f64 / lib_total as f64
                } else {
                    f64::NAN
                };
                let elapsed = cell_start.elapsed().as_secs_f64();
                let total_elapsed = start.elapsed().as_secs_f64();
                let est_total = total_elapsed * (total_cells as f64 / done_cells as f64);
                eprintln!(
                    "[{:>3}/{:>3}] {:>7} q={:>4.0} m={} -> size={:.4}x  ({} ok, {} skip)  cell={:.1}s  total={:.1}s/{:.1}s",
                    done_cells,
                    total_cells,
                    pp.name,
                    q,
                    m,
                    cell_ratio,
                    rows.len(),
                    cell_skipped,
                    elapsed,
                    total_elapsed,
                    est_total
                );
            }
        }
    }

    eprintln!("Done in {:.1}s. Wrote {}", start.elapsed().as_secs_f64(), out_path.display());
    eprintln!("Skipped cells: {}", skipped);
    for (corpus, a) in &agg {
        if a.n == 0 {
            continue;
        }
        let agg_size = a.zen_bytes as f64 / a.lib_bytes.max(1) as f64;
        let agg_speed = a.zen_ms / a.lib_ms.max(1e-6);
        let agg_butter = a.butter_delta / a.n as f64;
        let agg_ssim2 = a.ssim2_delta / a.n as f64;
        println!(
            "corpus={} n={} agg_size={:.4} agg_speed={:.3} agg_butter_delta={:+.3} agg_ssim2_delta={:+.2}",
            corpus, a.n, agg_size, agg_speed, agg_butter, agg_ssim2
        );
    }
}
