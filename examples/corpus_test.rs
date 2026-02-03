//! Corpus-level file size comparison between zenwebp and libwebp.
//!
//! Usage: cargo run --release --example corpus_test [directory]
//! Default directory: /tmp/CID22/original

use std::env;
use std::fs;
use zenwebp::{EncoderConfig, Preset};

fn main() {
    let args: Vec<_> = env::args().collect();
    let dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/tmp/CID22/original");

    let mut zen_total = 0u64;
    let mut lib_total = 0u64;
    let mut count = 0;

    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .collect();

    for entry in entries {
        let path = entry.path();
        if path.extension().map(|e| e != "png").unwrap_or(true) {
            continue;
        }

        let file = fs::File::open(&path).unwrap();
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = match decoder.read_info() {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) => continue,
        };

        let (rgb, w, h) = match info.color_type {
            png::ColorType::Rgb => (buf[..info.buffer_size()].to_vec(), info.width, info.height),
            png::ColorType::Rgba => {
                let rgba = &buf[..info.buffer_size()];
                let mut rgb = Vec::with_capacity(rgba.len() * 3 / 4);
                for chunk in rgba.chunks(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                }
                (rgb, info.width, info.height)
            }
            _ => continue,
        };

        let zen = match EncoderConfig::with_preset(Preset::Default, 75.0)
            .method(4)
            .sns_strength(0)
            .filter_strength(0)
            .segments(1)
            .encode_rgb(&rgb, w, h)
        {
            Ok(v) => v,
            Err(_) => continue,
        };

        let lib = match webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(4)
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1)
            .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        {
            Ok(v) => v,
            Err(_) => continue,
        };

        zen_total += zen.len() as u64;
        lib_total += lib.len() as u64;
        count += 1;

        let ratio = zen.len() as f64 / lib.len() as f64;
        let flag = if ratio > 1.05 {
            " <<<"
        } else if ratio < 0.95 {
            " >>>"
        } else {
            ""
        };
        eprintln!(
            "{}: zen={} lib={} ({:.3}x){}",
            path.file_name().unwrap().to_str().unwrap(),
            zen.len(),
            lib.len(),
            ratio,
            flag
        );
    }

    println!("\n=== Corpus Results (m4, Q75, SNS=0) ===");
    println!("Directory: {}", dir);
    println!("Images: {}", count);
    println!("zenwebp total: {} bytes", zen_total);
    println!("libwebp total: {} bytes", lib_total);
    if lib_total > 0 {
        println!("Ratio: {:.4}x", zen_total as f64 / lib_total as f64);
    }
}
