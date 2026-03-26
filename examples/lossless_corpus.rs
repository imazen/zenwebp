#![cfg(not(target_arch = "wasm32"))]
//! Lossless corpus benchmark: zenwebp vs libwebp (C) vs image-webp across methods and images.
//! Reports both file size and encode time.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench lossless_corpus

use std::path::PathBuf;

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgba(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32, String)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        _ => return None,
    };
    let name = path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    Some((rgba, info.width, info.height, name))
}

struct ImageData {
    rgba: Vec<u8>,
    w: u32,
    h: u32,
    name: String,
}

fn load_corpus(subdir: &str, files: &[&str]) -> Vec<ImageData> {
    files
        .iter()
        .filter_map(|f| {
            let path = corpus_path(subdir, f)?;
            let (rgba, w, h, name) = load_png_rgba(&path)?;
            Some(ImageData { rgba, w, h, name })
        })
        .collect()
}

fn encode_zenwebp(rgba: &[u8], w: u32, h: u32, method: u8) -> Vec<u8> {
    let config = zenwebp::EncoderConfig::new_lossless().with_method(method);
    zenwebp::EncodeRequest::new(&config, rgba, zenwebp::PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

fn encode_libwebp(rgba: &[u8], w: u32, h: u32, method: u8) -> Vec<u8> {
    webpx::EncoderConfig::new_lossless()
        .method(method)
        .encode_rgba(rgba, w, h, webpx::Unstoppable)
        .unwrap()
}

fn encode_image_webp(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let encoder = image_webp::WebPEncoder::new(&mut out);
    encoder
        .encode(rgba, w, h, image_webp::ColorType::Rgba8)
        .unwrap();
    out
}

fn main() {
    let photos: Vec<&str> = vec![
        "792079.png",
        "750463.png",
        "725551.png",
        "890595.png",
        "816411.png",
    ];
    let screenshots: Vec<&str> = vec![
        "codec_wiki.png",
        "terminal.png",
        "windows.png",
        "imac_dark.png",
        "imac_g3.png",
    ];

    let photo_images = load_corpus("CID22/CID22-512/validation", &photos);
    let screenshot_images = load_corpus("gb82-sc", &screenshots);

    if photo_images.is_empty() && screenshot_images.is_empty() {
        eprintln!("No corpus images found. Set CODEC_CORPUS_DIR or install codec-corpus.");
        return;
    }

    println!("# Lossless WebP Corpus Benchmark");
    println!("# zenwebp vs libwebp (C) vs image-webp");
    println!("# All encoders: lossless mode, same method, RGBA input");
    println!();

    for (corpus_name, images) in [("CID22 photos", &photo_images), ("Screenshots", &screenshot_images)] {
        if images.is_empty() {
            continue;
        }
        println!("## {corpus_name} ({} images)", images.len());
        println!();
        println!(
            "{:<20} {:>6} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
            "image", "method", "zenwebp", "ms", "libwebp", "ms", "image-webp", "ms"
        );
        println!("{:-<96}", "");

        for method in [0u8, 2, 4, 6] {
            let mut zen_total = 0u64;
            let mut lib_total = 0u64;
            let mut iwp_total = 0u64;
            let mut zen_ms_total = 0.0f64;
            let mut lib_ms_total = 0.0f64;
            let mut iwp_ms_total = 0.0f64;

            for img in images.iter() {
                // Warm up
                let _ = encode_zenwebp(&img.rgba, img.w, img.h, method);
                let _ = encode_libwebp(&img.rgba, img.w, img.h, method);

                // zenwebp: best of 3
                let mut zen_best = std::time::Duration::MAX;
                let mut zen_out = Vec::new();
                for _ in 0..3 {
                    let start = std::time::Instant::now();
                    zen_out = encode_zenwebp(&img.rgba, img.w, img.h, method);
                    let elapsed = start.elapsed();
                    if elapsed < zen_best {
                        zen_best = elapsed;
                    }
                }

                // libwebp: best of 3
                let mut lib_best = std::time::Duration::MAX;
                let mut lib_out = Vec::new();
                for _ in 0..3 {
                    let start = std::time::Instant::now();
                    lib_out = encode_libwebp(&img.rgba, img.w, img.h, method);
                    let elapsed = start.elapsed();
                    if elapsed < lib_best {
                        lib_best = elapsed;
                    }
                }

                // image-webp: best of 3 (no method control, always its default)
                let mut iwp_best = std::time::Duration::MAX;
                let mut iwp_out = Vec::new();
                if method <= 4 {
                    // only bench once per image since it has no method param
                    for _ in 0..3 {
                        let start = std::time::Instant::now();
                        iwp_out = encode_image_webp(&img.rgba, img.w, img.h);
                        let elapsed = start.elapsed();
                        if elapsed < iwp_best {
                            iwp_best = elapsed;
                        }
                    }
                }

                let zen_size = zen_out.len() as u64;
                let lib_size = lib_out.len() as u64;
                let iwp_size = if iwp_out.is_empty() { 0 } else { iwp_out.len() as u64 };

                zen_total += zen_size;
                lib_total += lib_size;
                iwp_total += iwp_size;
                zen_ms_total += zen_best.as_secs_f64() * 1000.0;
                lib_ms_total += lib_best.as_secs_f64() * 1000.0;
                if method <= 4 {
                    iwp_ms_total += iwp_best.as_secs_f64() * 1000.0;
                }

                let iwp_str = if iwp_size > 0 {
                    format!("{:>10}", iwp_size)
                } else {
                    format!("{:>10}", "—")
                };
                let iwp_ms_str = if method <= 4 {
                    format!("{:>7.1}", iwp_best.as_secs_f64() * 1000.0)
                } else {
                    format!("{:>7}", "—")
                };

                println!(
                    "{:<20} {:>6} {:>10} {:>7.1} {:>10} {:>7.1} {} {}",
                    img.name,
                    format!("m{method}"),
                    zen_size,
                    zen_best.as_secs_f64() * 1000.0,
                    lib_size,
                    lib_best.as_secs_f64() * 1000.0,
                    iwp_str,
                    iwp_ms_str,
                );
            }

            // Summary for this method
            let ratio = if lib_total > 0 {
                zen_total as f64 / lib_total as f64
            } else {
                0.0
            };
            let speed_ratio = if zen_ms_total > 0.0 {
                lib_ms_total / zen_ms_total
            } else {
                0.0
            };
            let iwp_ratio = if iwp_total > 0 {
                format!("{:.3}x", zen_total as f64 / iwp_total as f64)
            } else {
                "—".to_string()
            };

            println!(
                "{:<20} {:>6} {:>10} {:>7.1} {:>10} {:>7.1} {:>10} {:>7}",
                format!("TOTAL m{method}"),
                "",
                zen_total,
                zen_ms_total,
                lib_total,
                lib_ms_total,
                if iwp_total > 0 { iwp_total.to_string() } else { "—".into() },
                if method <= 4 { format!("{:.1}", iwp_ms_total) } else { "—".into() },
            );
            println!(
                "  zen/lib size: {:.3}x | lib/zen time: {:.1}x faster | zen/iwp size: {}",
                ratio, speed_ratio, iwp_ratio,
            );
            println!();
        }
    }
}
