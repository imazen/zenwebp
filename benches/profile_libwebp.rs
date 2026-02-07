use std::path::Path;
use std::time::Instant;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb_data = match info.color_type {
        png::PixelLayout::Rgb => buf[..info.buffer_size()].to_vec(),
        png::PixelLayout::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => panic!("Unsupported color type"),
    };

    (rgb_data, info.width, info.height)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let image_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or(concat!(env!("HOME"), "/work/codec-corpus/kodak/1.png"));
    let quality: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(75.0);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let method: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    let path = Path::new(image_path);
    println!("Loading: {}", path.display());
    let (rgb_data, width, height) = load_png(path);
    println!("Image: {}x{} ({} pixels)", width, height, width * height);
    println!(
        "Quality: {}, Method: {}, Iterations: {}",
        quality, method, iterations
    );

    // Create config with method setting
    let mut config = webp::WebPConfig::new().unwrap();
    config.quality = quality;
    config.method = method;

    // Warmup
    let encoder = webp::Encoder::from_rgb(&rgb_data, width, height);
    let output = encoder.encode_advanced(&config).unwrap();
    println!("Output size: {} bytes", output.len());

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iterations {
        let encoder = webp::Encoder::from_rgb(&rgb_data, width, height);
        let _output = encoder.encode_advanced(&config).unwrap();
    }
    let elapsed = start.elapsed();

    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let mpix_per_sec =
        (width as f64 * height as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;

    println!("\n=== Results (libwebp, method {}) ===", method);
    println!(
        "Total time: {:.2}ms for {} iterations",
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );
    println!("Per iteration: {:.2}ms", ms_per_iter);
    println!("Throughput: {:.2} MPix/s", mpix_per_sec);
}
