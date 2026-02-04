//! Comprehensive decoder/encoder benchmark across multiple corpora

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn benchmark_image(path: &Path) -> Option<(String, u32, u32, f64, f64, f64, f64, f64)> {
    // Load PNG
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() * 3 / 4);
            for chunk in rgba.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    // Encode with libwebp for decode testing
    let webp_data = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(5)
        .encode_rgb(&rgb, info.width, info.height, webpx::Unstoppable)
        .ok()?;

    let iterations = 10;
    let pixels = info.width as f64 * info.height as f64;

    // Decode benchmarks
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = zenwebp::decode_rgb(&webp_data);
    }
    let zen_decode = start.elapsed().as_secs_f64() / iterations as f64;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut d = image_webp::WebPDecoder::new(std::io::Cursor::new(&webp_data)).unwrap();
        let mut buf = vec![0u8; d.output_buffer_size().unwrap()];
        let _ = d.read_image(&mut buf);
    }
    let upstream_decode = start.elapsed().as_secs_f64() / iterations as f64;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = webpx::decode_rgb(&webp_data);
    }
    let lib_decode = start.elapsed().as_secs_f64() / iterations as f64;

    Some((
        path.file_name()?.to_string_lossy().to_string(),
        info.width,
        info.height,
        pixels / zen_decode / 1e6,       // zenwebp MPix/s
        pixels / upstream_decode / 1e6,  // image-webp MPix/s
        pixels / lib_decode / 1e6,       // libwebp MPix/s
        zen_decode / lib_decode,         // zenwebp ratio
        upstream_decode / lib_decode,    // upstream ratio
    ))
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let dir = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/clic2025");
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("Benchmarking {} images from {}", limit, dir);
    println!();

    let mut results = Vec::new();
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
        .take(limit)
        .collect();

    for entry in &entries {
        if let Some(r) = benchmark_image(&entry.path()) {
            println!("{}: {}x{}", r.0, r.1, r.2);
            println!("  zenwebp:   {:6.1} MPix/s ({:.2}x vs libwebp)", r.3, r.6);
            println!("  image-webp:{:6.1} MPix/s ({:.2}x vs libwebp)", r.4, r.7);
            println!("  libwebp:   {:6.1} MPix/s", r.5);
            results.push(r);
        }
    }

    if !results.is_empty() {
        let avg_zen: f64 = results.iter().map(|r| r.6).sum::<f64>() / results.len() as f64;
        let avg_upstream: f64 = results.iter().map(|r| r.7).sum::<f64>() / results.len() as f64;
        let total_pixels: f64 = results.iter().map(|r| r.1 as f64 * r.2 as f64).sum();

        println!();
        println!("=== Summary ({} images, {:.1} megapixels total) ===", results.len(), total_pixels / 1e6);
        println!("zenwebp avg:    {:.2}x vs libwebp", avg_zen);
        println!("image-webp avg: {:.2}x vs libwebp", avg_upstream);
        println!("zenwebp is {:.1}x faster than image-webp", avg_upstream / avg_zen);
    }
}
