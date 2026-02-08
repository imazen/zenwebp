//! Tune psy-rd parameters using butteraugli feedback.
//!
//! This example encodes test images with different psy-rd configurations
//! and measures butteraugli scores to find optimal parameters.
//!
//! Usage: cargo run --release --example tune_psy_rd

use butteraugli::{butteraugli, ButteraugliParams};
use imgref::Img;
use rgb::RGB8;
use std::env;
use std::fs;
use zenwebp::{decode_rgb, EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let args: Vec<_> = env::args().collect();
    let dir = if let Some(d) = args.get(1) {
        std::path::PathBuf::from(d)
    } else {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        corpus.get("CID22/CID22-512/training").expect("corpus path unavailable")
    };

    // Find PNG files
    let entries: Vec<_> = fs::read_dir(&dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .take(15) // Test on 15 images for reasonable speed
        .collect();

    println!("Testing {} images from {}", entries.len(), dir.display());
    println!();

    // Test configurations: (method, description)
    let configs = [
        (3, "m3 (CSF only)"),
        (4, "m4 (CSF + psy-rd)"),
        (5, "m5 (CSF + psy-rd + psy-trellis + JND)"),
    ];

    for (method, desc) in configs {
        let mut sizes = Vec::new();
        let mut scores = Vec::new();

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
                png::PixelLayout::Rgb => (
                    buf[..info.buffer_size()].to_vec(),
                    info.width as usize,
                    info.height as usize,
                ),
                png::PixelLayout::Rgba => {
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
            let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0).with_method(method);
            let webp = match EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w as u32, h as u32)
                .encode()
            {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Decode
            let (decoded_rgb, _, _) = match decode_rgb(&webp) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Convert to RGB8 for butteraugli
            let src_rgb8: Vec<RGB8> = rgb.chunks(3).map(|c| RGB8::new(c[0], c[1], c[2])).collect();
            let dst_rgb8: Vec<RGB8> = decoded_rgb
                .chunks(3)
                .map(|c| RGB8::new(c[0], c[1], c[2]))
                .collect();

            let src_img = Img::new(src_rgb8, w, h);
            let dst_img = Img::new(dst_rgb8, w, h);

            // Compute butteraugli
            let params = ButteraugliParams::default();
            let result = match butteraugli(src_img.as_ref(), dst_img.as_ref(), &params) {
                Ok(r) => r,
                Err(_) => continue,
            };

            sizes.push(webp.len());
            scores.push(result.score);
        }

        if !sizes.is_empty() {
            let avg_size: f64 = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
            let avg_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

            println!(
                "{}: avg_size={:.0}KB, avg_butteraugli={:.3}",
                desc,
                avg_size / 1024.0,
                avg_score
            );
        }
    }

    println!();
    println!("Lower butteraugli score = better perceptual quality");
    println!("Target: smaller files with same or better butteraugli");
}
