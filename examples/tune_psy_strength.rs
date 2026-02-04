//! Compare encoding methods with butteraugli quality metric.
//!
//! Tests different method levels (2-6) to compare size and perceptual quality.
//!
//! Usage: cargo run --release --example tune_psy_strength [directory]

use butteraugli::{butteraugli, ButteraugliParams};
use imgref::Img;
use rgb::RGB8;
use std::env;
use std::fs;
use zenwebp::{decode_rgb, EncoderConfig, Preset};

fn main() {
    let args: Vec<_> = env::args().collect();
    let dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/lilith/work/codec-corpus/CID22/CID22-512/training");

    // Find PNG files
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .take(20)
        .collect();

    println!("Testing {} images", entries.len());
    println!();
    println!("Method | Avg Size | Avg Butteraugli");
    println!("-------|----------|----------------");

    for method in 2..=6 {
        let mut total_size = 0usize;
        let mut total_score = 0.0f64;
        let mut count = 0usize;

        for entry in &entries {
            let path = entry.path();

            // Load PNG
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
                png::ColorType::Rgb => (
                    buf[..info.buffer_size()].to_vec(),
                    info.width as usize,
                    info.height as usize,
                ),
                png::ColorType::Rgba => {
                    let rgba = &buf[..info.buffer_size()];
                    let mut rgb = Vec::with_capacity(rgba.len() * 3 / 4);
                    for chunk in rgba.chunks(4) {
                        rgb.extend_from_slice(&chunk[..3]);
                    }
                    (rgb, info.width as usize, info.height as usize)
                }
                _ => continue,
            };

            // Encode
            let webp = match EncoderConfig::with_preset(Preset::Default, 75.0)
                .method(method)
                .encode_rgb(&rgb, w as u32, h as u32)
            {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Decode
            let (decoded_rgb, _, _) = match decode_rgb(&webp) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Compute butteraugli
            let src_rgb8: Vec<RGB8> = rgb.chunks(3).map(|c| RGB8::new(c[0], c[1], c[2])).collect();
            let dst_rgb8: Vec<RGB8> = decoded_rgb
                .chunks(3)
                .map(|c| RGB8::new(c[0], c[1], c[2]))
                .collect();
            let params = ButteraugliParams::default();
            let result = match butteraugli(
                Img::new(src_rgb8, w, h).as_ref(),
                Img::new(dst_rgb8, w, h).as_ref(),
                &params,
            ) {
                Ok(r) => r,
                Err(_) => continue,
            };

            total_size += webp.len();
            total_score += result.score;
            count += 1;
        }

        if count > 0 {
            println!(
                "m{} | {:>6.1} KB | {:.3}",
                method,
                total_size as f64 / count as f64 / 1024.0,
                total_score / count as f64
            );
        }
    }

    println!();
    println!("Lower butteraugli = better quality");
}
