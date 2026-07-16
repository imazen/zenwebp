//! Output-invariance gate for speed work (#38 speed parity).
//!
//! Encodes a deterministic grid across BOTH cost models — lossy (RGB +
//! RGBA-with-alpha, sharp_yuv on/off), lossless — and prints one FNV-1a
//! hash per section plus a combined hash. Speed optimizations must leave
//! every byte unchanged: run before and after, diff the output.
//!
//! Usage:
//!   cargo run --release --features __expert --example output_hash [corpus_dir]
//!
//! The corpus dir (default ~/tmp/abcorpus) adds real images on top of the
//! synthetic grid; absent files are skipped LOUDLY (listed in the output)
//! so a hash produced with a missing corpus is visibly different.

use zenwebp::{CostModel, EncodeRequest, LosslessConfig, LossyConfig, PixelLayout};

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn synth(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 24) as u8 / 8;
            px.extend_from_slice(&[
                ((x * 255 / w.max(1)) as u8).wrapping_add(n),
                ((y * 255 / h.max(1)) as u8).wrapping_add(n),
                (((x + y) * 255 / (w + h).max(1)) as u8).wrapping_add(n),
            ]);
        }
    }
    px
}

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let mut d = png::Decoder::new(std::io::BufReader::new(file));
    d.set_transformations(png::Transformations::normalize_to_color8());
    let mut r = d.read_info().ok()?;
    let mut buf = vec![0u8; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/tmp/abcorpus", std::env::var("HOME").unwrap()));

    // Sources: synthetics (incl. tiny/odd) + corpus PNGs.
    let mut images: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();
    for (w, h, s) in [
        (3u32, 3u32, 13u32),
        (17, 17, 19),
        (33, 17, 23),
        (64, 64, 31),
        (129, 127, 37),
        (256, 255, 41),
    ] {
        images.push((format!("synth_{w}x{h}"), synth(w, h, s), w, h));
    }
    let mut corpus_files: Vec<std::path::PathBuf> = std::fs::read_dir(&corpus)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "png"))
                .collect()
        })
        .unwrap_or_default();
    corpus_files.sort();
    if corpus_files.is_empty() {
        println!("WARNING: no corpus PNGs at {corpus} — synthetic-only hash");
    }
    for p in corpus_files {
        if let Some((rgb, w, h)) = load_png(&p) {
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            images.push((name, rgb, w, h));
        }
    }
    println!("{} images", images.len());

    let mut combined: u64 = 0xcbf29ce484222325;
    let mut section = |label: &str, h: u64| {
        println!("{label}: {h:016x}");
        fnv(&mut combined, &h.to_le_bytes());
    };

    // Lossy RGB, both cost models, m0-m6 × q {5,50,75,90}.
    for &model in &[CostModel::ZenwebpDefault, CostModel::StrictLibwebpParity] {
        let mut h: u64 = 0xcbf29ce484222325;
        for (name, rgb, w, h_px) in &images {
            for m in 0u8..=6 {
                for &q in &[5u8, 50, 75, 90] {
                    let cfg = LossyConfig::new()
                        .with_quality(f32::from(q))
                        .with_method(m)
                        .with_cost_model(model);
                    let out = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, *w, *h_px)
                        .encode()
                        .unwrap_or_else(|e| panic!("{name} m{m} q{q}: {e:?}"));
                    fnv(&mut h, &out);
                }
            }
        }
        section(&format!("lossy_rgb_{model:?}"), h);
    }

    // Lossy RGBA (gradient + checker alpha), both models, m {0,2,4,6} × q {50,90},
    // sns50/flt60/segs4, aq 100/90; plus sharp_yuv on the RGB grid m{0,4} q75.
    for &model in &[CostModel::ZenwebpDefault, CostModel::StrictLibwebpParity] {
        let mut h: u64 = 0xcbf29ce484222325;
        for (name, rgb, w, h_px) in &images {
            let rgba: Vec<u8> = rgb
                .chunks_exact(3)
                .enumerate()
                .flat_map(|(i, p)| {
                    let (x, y) = ((i as u32) % *w, (i as u32) / *w);
                    let a = if (x + y) % 3 == 0 {
                        (x * 255 / (*w).max(1)) as u8
                    } else if ((x / 8) + (y / 8)) % 2 == 0 {
                        0
                    } else {
                        255
                    };
                    [p[0], p[1], p[2], a]
                })
                .collect();
            for &m in &[0u8, 2, 4, 6] {
                for &q in &[50u8, 90] {
                    for &aq in &[100u8, 90] {
                        let mut cfg = LossyConfig::new()
                            .with_quality(f32::from(q))
                            .with_method(m)
                            .with_segments(4)
                            .with_sns_strength(50)
                            .with_filter_strength(60)
                            .with_cost_model(model);
                        cfg.alpha_quality = aq;
                        let out = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, *w, *h_px)
                            .encode()
                            .unwrap_or_else(|e| panic!("{name} alpha m{m} q{q}: {e:?}"));
                        fnv(&mut h, &out);
                    }
                }
            }
            for &m in &[0u8, 4] {
                let cfg = LossyConfig::new()
                    .with_quality(75.0)
                    .with_method(m)
                    .with_sharp_yuv(true)
                    .with_cost_model(model);
                let out = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, *w, *h_px)
                    .encode()
                    .unwrap_or_else(|e| panic!("{name} sharp m{m}: {e:?}"));
                fnv(&mut h, &out);
            }
        }
        section(&format!("lossy_alpha_sharp_{model:?}"), h);
    }

    // Lossless RGB, m0-m6.
    {
        let mut h: u64 = 0xcbf29ce484222325;
        for (name, rgb, w, h_px) in &images {
            for m in 0u8..=6 {
                let cfg = LosslessConfig::new().with_method(m);
                let out = EncodeRequest::lossless(&cfg, rgb, PixelLayout::Rgb8, *w, *h_px)
                    .encode()
                    .unwrap_or_else(|e| panic!("{name} lossless m{m}: {e:?}"));
                fnv(&mut h, &out);
            }
        }
        section("lossless_rgb", h);
    }

    println!("COMBINED: {combined:016x}");
}
