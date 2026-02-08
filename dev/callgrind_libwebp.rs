//! Minimal libwebp encoder for callgrind profiling (via webpx, no PNG dependency).
//! Usage: cargo run --release --example callgrind_libwebp -- <raw_rgb_file> <width> <height> <quality> <method>
//! Pre-convert: convert image.png -depth 8 RGB:image_WxH.rgb
//!
//! Matches callgrind_encode.rs interface for fair comparison:
//!   valgrind --tool=callgrind target/release/examples/callgrind_encode  file.rgb W H 75 4
//!   valgrind --tool=callgrind target/release/examples/callgrind_libwebp file.rgb W H 75 4

use std::env;
use std::fs;

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

    let diagnostic = args.get(6).is_none_or(|s| s != "default");

    let output = if diagnostic {
        webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1)
            .encode_rgb(&rgb_data, width, height, webpx::Unstoppable)
            .unwrap()
    } else {
        webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
            .method(method)
            .encode_rgb(&rgb_data, width, height, webpx::Unstoppable)
            .unwrap()
    };

    eprintln!("Output: {} bytes", output.len());
}
