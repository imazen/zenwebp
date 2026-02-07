//! Run a single encode/decode operation for heaptrack profiling.
//!
//! Usage:
//!   cargo build --release --example heaptrack_test
//!   heaptrack target/release/examples/heaptrack_test encode 1920 1080 4 75
//!   heaptrack target/release/examples/heaptrack_test decode 1920 1080

use std::env;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} encode <width> <height> <method> <quality>", args[0]);
        eprintln!("  {} decode <width> <height>", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "encode" => {
            let width: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1920);
            let height: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1080);
            let method: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
            let quality: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(75.0);

            eprintln!(
                "Encoding {}x{} (method={}, quality={})",
                width, height, method, quality
            );

            let img = generate_test_image(width, height, 4);
            let config = EncoderConfig::new().quality(quality).method(method);

            match EncodeRequest::new(&config, &img, PixelLayout::Rgba8, width, height).encode() {
                Ok(output) => {
                    eprintln!("Encoded to {} bytes", output.len());
                }
                Err(e) => {
                    eprintln!("Encode error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "decode" => {
            let width: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1920);
            let height: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1080);

            eprintln!("Decoding {}x{}", width, height);

            // First encode
            let img = generate_test_image(width, height, 4);
            let config = EncoderConfig::new().quality(75.0).method(4);
            let webp = EncodeRequest::new(&config, &img, PixelLayout::Rgba8, width, height)
                .encode()
                .expect("Failed to encode test image");

            // Now decode (this is what we're measuring)
            match zenwebp::decode_rgba(&webp) {
                Ok((pixels, w, h)) => {
                    eprintln!("Decoded to {}x{}, {} bytes", w, h, pixels.len());
                }
                Err(e) => {
                    eprintln!("Decode error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown mode: {}", args[1]);
            std::process::exit(1);
        }
    }
}

fn generate_test_image(width: u32, height: u32, bpp: usize) -> Vec<u8> {
    let mut img = vec![0u8; (width * height) as usize * bpp];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) as usize) * bpp;

            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = ((x + y) * 127 / (width + height)) as u8;

            img[idx] = r;
            if bpp > 1 {
                img[idx + 1] = g;
            }
            if bpp > 2 {
                img[idx + 2] = b;
            }
            if bpp > 3 {
                img[idx + 3] = 255;
            }
        }
    }

    img
}
