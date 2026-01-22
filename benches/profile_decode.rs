use image_webp::{ColorType, EncoderParams, WebPEncoder};
use std::path::Path;
use std::time::Instant;

fn load_webp(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("Failed to read WebP file")
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb_data: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
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

fn benchmark_decode(webp_data: &[u8], iterations: usize) {
    // Decode once to get dimensions and buffer size
    let cursor = std::io::Cursor::new(webp_data);
    let decoder = image_webp::WebPDecoder::new(cursor).unwrap();
    let (width, height) = decoder.dimensions();
    let output_size = decoder.output_buffer_size().unwrap();
    println!("Image: {}x{} ({} pixels)", width, height, width * height);
    println!("WebP size: {} bytes", webp_data.len());
    println!("Output buffer size: {} bytes", output_size);
    println!("Iterations: {}", iterations);

    // Benchmark our decoder
    println!("\n=== Our Decoder ===");
    let start = Instant::now();
    for _ in 0..iterations {
        let cursor = std::io::Cursor::new(webp_data);
        let mut decoder = image_webp::WebPDecoder::new(cursor).unwrap();
        let mut output = vec![0u8; output_size];
        decoder.read_image(&mut output).unwrap();
    }
    let elapsed = start.elapsed();
    let our_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let our_mpix =
        (width as f64 * height as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;
    println!("Per iteration: {:.2}ms", our_ms);
    println!("Throughput: {:.2} MPix/s", our_mpix);

    // Benchmark libwebp decoder
    println!("\n=== libwebp Decoder ===");
    let start = Instant::now();
    for _ in 0..iterations {
        let _decoded = webp::Decoder::new(webp_data).decode().unwrap();
    }
    let elapsed = start.elapsed();
    let libwebp_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let libwebp_mpix =
        (width as f64 * height as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;
    println!("Per iteration: {:.2}ms", libwebp_ms);
    println!("Throughput: {:.2} MPix/s", libwebp_mpix);

    // Summary
    println!("\n=== Summary ===");
    println!("Speed ratio: {:.2}x (ours / libwebp)", our_ms / libwebp_ms);
    if our_ms > libwebp_ms {
        println!("We are {:.1}% slower", (our_ms / libwebp_ms - 1.0) * 100.0);
    } else {
        println!("We are {:.1}% faster", (1.0 - our_ms / libwebp_ms) * 100.0);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let webp_path = args.get(1).map(|s| s.as_str());
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    if let Some(path) = webp_path {
        println!("=== Testing with provided WebP: {} ===\n", path);
        let webp_data = load_webp(Path::new(path));
        benchmark_decode(&webp_data, iterations);
    } else {
        // Test with both libwebp-encoded and our-encoded WebP
        let png_path = concat!(env!("HOME"), "/work/codec-corpus/kodak/1.png");
        println!("Loading PNG: {}\n", png_path);
        let (rgb_data, width, height) = load_png(Path::new(png_path));

        // Test 1: libwebp-encoded WebP
        println!("========================================");
        println!("Test 1: Decoding libwebp-encoded WebP");
        println!("========================================");
        let libwebp_encoded = {
            let encoder = webp::Encoder::from_rgb(&rgb_data, width, height);
            encoder.encode(75.0).to_vec()
        };
        benchmark_decode(&libwebp_encoded, iterations);

        // Test 2: Our encoder's WebP
        println!("\n========================================");
        println!("Test 2: Decoding our-encoded WebP");
        println!("========================================");
        let our_encoded = {
            let mut output = Vec::new();
            let mut encoder = WebPEncoder::new(&mut output);
            encoder.set_params(EncoderParams::lossy(75));
            encoder
                .encode(&rgb_data, width, height, ColorType::Rgb8)
                .unwrap();
            output
        };
        benchmark_decode(&our_encoded, iterations);
    }
}
