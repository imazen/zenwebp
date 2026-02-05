//! Minimal encoder for callgrind profiling (no PNG dependency).
//! Usage: cargo run --release --features simd --example callgrind_encode -- <raw_rgb_file> <width> <height> <quality> <method>
//! Pre-convert: convert image.png -depth 8 RGB:image_WxH.rgb

use std::env;
use std::fs;
use zenwebp::EncoderConfig;

fn main() {
    let args: Vec<String> = env::args().collect();
    let raw_path = args.get(1).expect("raw RGB file path");
    let width: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let height: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let quality: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(75.0);
    let method: u8 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);

    let rgb_data = fs::read(raw_path).expect("Failed to read raw file");
    assert_eq!(
        rgb_data.len(),
        (width * height * 3) as usize,
        "File size mismatch"
    );

    let output = EncoderConfig::new()
        .quality(quality)
        .method(method)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb_data, width, height)
        .unwrap();

    eprintln!("Output: {} bytes", output.len());
}
