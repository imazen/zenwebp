//! Run target_zensim against a corpus and report convergence stats.
//! Used to populate the PR description with real numbers.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use zenwebp::{LossyConfig, ZensimTarget};

fn main() {
    let corpus_dir = env::args().nth(1).unwrap_or_else(|| {
        let home = env::var("HOME").expect("HOME not set");
        format!("{home}/work/codec-corpus/CID22/CID22-512/validation")
    });
    let target_str = env::args().nth(2).unwrap_or_else(|| "80".to_string());
    let target_v: f32 = target_str.parse().unwrap_or(80.0);
    let max_passes: u8 = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut paths: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read corpus dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().is_some_and(|e| e == "png" || e == "PNG")).then_some(p)
        })
        .collect();
    paths.sort();
    if paths.len() > 20 {
        let stride = paths.len() / 20;
        paths = paths
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .take(20)
            .map(|(_, p)| p.clone())
            .collect();
    }

    let mut passes_used: Vec<u8> = Vec::new();
    let mut achieved: Vec<f32> = Vec::new();
    let mut bytes: Vec<usize> = Vec::new();
    let mut targets_met = 0;
    let mut undershoot_count = 0;
    let mut overshoot_count = 0;

    eprintln!(
        "target={} max_passes={} corpus={}",
        target_v, max_passes, corpus_dir
    );
    eprintln!("file\twidth\theight\tachieved\tpasses\tbytes");

    for path in &paths {
        let (rgb, w, h) = match decode_png_rgb(path) {
            Some(t) => t,
            None => continue,
        };
        if w < 32 || h < 32 {
            continue;
        }
        let cfg = LossyConfig::new().with_method(4).with_target_zensim_target(
            ZensimTarget::new(target_v)
                .with_max_overshoot(Some(1.5))
                .with_max_passes(max_passes),
        );
        let result = cfg.encode_rgb_with_metrics(&rgb, w, h);
        match result {
            Ok((b, m)) => {
                eprintln!(
                    "{}\t{}\t{}\t{:.2}\t{}\t{}",
                    path.file_name().unwrap().to_string_lossy(),
                    w,
                    h,
                    m.achieved_score,
                    m.passes_used,
                    b.len(),
                );
                passes_used.push(m.passes_used);
                achieved.push(m.achieved_score);
                bytes.push(b.len());
                if m.targets_met {
                    targets_met += 1;
                }
                if m.achieved_score < target_v {
                    undershoot_count += 1;
                }
                if m.achieved_score > target_v + 1.5 {
                    overshoot_count += 1;
                }
            }
            Err(e) => {
                eprintln!(
                    "{}\tERROR\t{:?}",
                    path.file_name().unwrap().to_string_lossy(),
                    e
                );
            }
        }
    }

    if passes_used.is_empty() {
        eprintln!("no successful encodes");
        return;
    }
    passes_used.sort();
    achieved.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = passes_used.len();
    let p25 = |v: &[u8]| v[(n / 4).min(n - 1)];
    let p50 = |v: &[u8]| v[n / 2];
    let p75 = |v: &[u8]| v[((n * 3) / 4).min(n - 1)];
    let avg_passes: f32 = passes_used.iter().map(|&p| p as f32).sum::<f32>() / n as f32;
    let avg_score: f32 = achieved.iter().sum::<f32>() / n as f32;
    eprintln!();
    eprintln!("=== summary (n={n}, target={target_v}) ===");
    eprintln!(
        "passes_used: avg={:.2} p25={} p50={} p75={}",
        avg_passes,
        p25(&passes_used),
        p50(&passes_used),
        p75(&passes_used),
    );
    eprintln!(
        "achieved: avg={:.2} p25={:.2} p50={:.2} p75={:.2}",
        avg_score,
        achieved[n / 4],
        achieved[n / 2],
        achieved[(n * 3) / 4],
    );
    eprintln!("targets_met: {}/{n}", targets_met);
    eprintln!("undershoot (<target): {}/{n}", undershoot_count);
    eprintln!("overshoot (>target+1.5): {}/{n}", overshoot_count);
    eprintln!(
        "in-band ([target, target+1.5]): {}/{n}",
        n - undershoot_count - overshoot_count
    );
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
