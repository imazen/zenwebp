//! Minimal encoder for callgrind profiling.
//! Usage: cargo run --release --example callgrind_encode_m4 -- <raw_rgb_file> <width> <height>

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let raw_path = args.get(1).expect("raw RGB file path");
    let width: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);

    let rgb_data = fs::read(raw_path).expect("Failed to read raw file");
    assert_eq!(
        rgb_data.len(),
        (width * height * 3) as usize,
        "File size mismatch"
    );

    let cfg = zenwebp::LossyConfig::new()
        .with_quality(75.0)
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_segments(1);
    let output =
        zenwebp::EncodeRequest::lossy(&cfg, &rgb_data, zenwebp::PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap();
    eprintln!("Output: {} bytes", output.len());
}
