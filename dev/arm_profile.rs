//! Standalone encode + decode hot-path profiling driver for zenwebp.
//!
//! Loads a PNG (path from argv[1] or ZENWEBP_PROF_IMG env), encodes it to WebP
//! once, then runs tight encode/decode loops suitable for `perf record`
//! sampling. Unlike the criterion benches this has no warmup/statistics
//! machinery in the way, so the perf samples land squarely on the codec.
//!
//! Build: cargo build --release --features _profiling --example arm_profile
//! Run:   perf record -F 1999 -g -- \
//!          target/release/examples/arm_profile <png> [enc_iters] [dec_iters]
//!
//! Env:   ZENWEBP_PROF_MODE = both | enc | dec  (default both)
//!        ZENWEBP_PROF_IMG  = path to PNG if argv[1] absent
//!
//! Used for the 2026-05-30 ARM (aarch64) hot-path profile in
//! benchmarks/zenwebp_arm_profile_2026-05-30.tsv.

#[cfg(feature = "_profiling")]
use std::hint::black_box;
#[cfg(feature = "_profiling")]
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout};

#[cfg(feature = "_profiling")]
fn load_png(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| &c[..3])
            .copied()
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => buf[..info.buffer_size()].to_vec(),
    };
    (rgb, info.width, info.height)
}

#[cfg(feature = "_profiling")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("ZENWEBP_PROF_IMG").ok())
        .expect("usage: arm_profile <png> [enc_iters] [dec_iters]");
    let enc_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let dec_iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
    let mode = std::env::var("ZENWEBP_PROF_MODE").unwrap_or_else(|_| "both".into());

    let (rgb, w, h) = load_png(&path);
    eprintln!("loaded {path} {w}x{h} ({} bytes rgb)", rgb.len());

    let config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    let webp = EncodeRequest::new(&config, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();
    eprintln!("encoded {} bytes webp", webp.len());

    if mode == "both" || mode == "enc" {
        for _ in 0..enc_iters {
            let out = EncodeRequest::new(&config, black_box(&rgb), PixelLayout::Rgb8, w, h)
                .encode()
                .unwrap();
            black_box(out);
        }
        eprintln!("encode loop done ({enc_iters} iters)");
    }

    if mode == "both" || mode == "dec" {
        let dcfg = DecodeConfig::default();
        for _ in 0..dec_iters {
            let frame = DecodeRequest::new(&dcfg, black_box(&webp))
                .decode_rgba()
                .unwrap();
            black_box(frame);
        }
        eprintln!("decode loop done ({dec_iters} iters)");
    }
}

#[cfg(not(feature = "_profiling"))]
fn main() {
    eprintln!("build with --features _profiling");
}
