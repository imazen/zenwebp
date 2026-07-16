//! Tuned-default A/B harness for measured adoption of parity-derived
//! candidates (see `benchmarks/tuned_candidates_2026-07-14.md` for the
//! discipline this feeds).
//!
//! Encodes every PNG in a corpus directory with `CostModel::ZenwebpDefault`
//! at the default preset knobs across a (method × quality) grid and emits
//! one TSV row per cell: `variant image m q size zsim ms` — the same shape
//! as `benchmarks/tuned_candidates_2026-07-14.tsv`. `ms` is the min of 3
//! encode repetitions; `zsim` scores the decoded output against the source
//! with the default zensim profile. Rows APPEND to `--out`, so the baseline
//! build and the candidate build write into one file distinguished by
//! `--variant`.
//!
//! Deliberately single-threaded: the `ms` column stays comparable and the
//! whole 15-image × 3m × 4q run is well under a minute.
//!
//! Usage (run once per build variant):
//!   cargo run --release --example tuned_ab_sweep -- \
//!     --corpus ~/tmp/abcorpus --variant baseline \
//!     --methods 4,5,6 --qs 25,50,75,90 --out ~/tmp/ab_trellis_skip.tsv
//!
//! Optional knob overrides (default preset values when absent):
//!   --segments N --sns N — for candidates that only fire off-default
//!   (e.g. the segs1 uv_alpha dq_uv adoption A/B).

#![forbid(unsafe_code)]

use std::io::Write as _;
use std::time::Instant;

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

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

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_u8_list(s: &str) -> Vec<u8> {
    s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_value(&args, "--corpus").expect("--corpus <dir> required");
    let variant = arg_value(&args, "--variant").expect("--variant <label> required");
    let out = arg_value(&args, "--out").expect("--out <tsv> required");
    let methods = parse_u8_list(&arg_value(&args, "--methods").unwrap_or_else(|| "4,5,6".into()));
    let qs = parse_u8_list(&arg_value(&args, "--qs").unwrap_or_else(|| "25,50,75,90".into()));
    let segments: Option<u8> = arg_value(&args, "--segments").and_then(|s| s.parse().ok());
    let sns: Option<u8> = arg_value(&args, "--sns").and_then(|s| s.parse().ok());

    let mut images: Vec<std::path::PathBuf> = std::fs::read_dir(&corpus)
        .expect("corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    images.sort();
    assert!(!images.is_empty(), "no PNGs in {corpus}");

    let need_header = !std::path::Path::new(&out).exists();
    let mut tsv = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .expect("open --out");
    if need_header {
        writeln!(tsv, "variant\timage\tm\tq\tsize\tzsim\tms").unwrap();
    }

    let z = Zensim::new(ZensimProfile::latest());
    for path in &images {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some((rgb, w, h)) = load_png(path) else {
            eprintln!("skip (load failed): {}", path.display());
            continue;
        };
        let src_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb);
        let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
        let pre = z
            .precompute_reference(&src_slice)
            .expect("zensim precompute");

        for &m in &methods {
            for &q in &qs {
                let mut cfg = LossyConfig::new().with_quality(f32::from(q)).with_method(m);
                if let Some(segs) = segments {
                    cfg = cfg.with_segments(segs);
                }
                if let Some(sns) = sns {
                    cfg = cfg.with_sns_strength(sns);
                }
                let mut best_ms = f64::MAX;
                let mut webp: Vec<u8> = Vec::new();
                for _ in 0..3 {
                    let t = Instant::now();
                    webp = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
                        .encode()
                        .expect("encode");
                    best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                let (dec, w2, h2) = zenwebp::oneshot::decode_rgb(&webp).expect("decode");
                assert_eq!((w2, h2), (w, h));
                let n = (w as usize) * (h as usize) * 3;
                let dec_chunks: &[[u8; 3]] = bytemuck::cast_slice(&dec[..n]);
                let dec_slice = RgbSlice::new(dec_chunks, w as usize, h as usize);
                let zsim = z
                    .compute_with_ref(&pre, &dec_slice)
                    .expect("zensim")
                    .score();
                writeln!(
                    tsv,
                    "{variant}\t{name}\t{m}\t{q}\t{}\t{zsim:.3}\t{best_ms:.2}",
                    webp.len()
                )
                .unwrap();
            }
        }
        eprintln!("done: {name}");
    }
    eprintln!("wrote {out} (variant={variant})");
}
