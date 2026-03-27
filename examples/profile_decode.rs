// Decode-only profiling binary for callgrind.
// Saves the encoded WebP to /tmp/profile_test.webp so the C profiler can use the same file.
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let webp = if args.len() > 1 {
        // If a WebP file path is given, use it directly
        std::fs::read(&args[1]).expect("Failed to read WebP file")
    } else {
        // Generate a test image (gradient + noise for realistic compression)
        let width = 1024u32;
        let height = 1024u32;
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        let mut seed: u32 = 12345;
        for y in 0..height {
            for x in 0..width {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let noise = ((seed >> 16) & 0x1f) as u8;
                rgb.push(((x * 255) / width) as u8 ^ noise);
                rgb.push(((y * 255) / height) as u8 ^ noise);
                rgb.push((((x + y) * 127) / (width + height)) as u8 ^ noise);
            }
        }

        // Encode to WebP once
        let cfg = EncoderConfig::new_lossy().with_quality(75.0);
        let webp = EncodeRequest::new(&cfg, &rgb, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap();
        eprintln!("Encoded {}x{} to {} bytes", width, height, webp.len());

        // Save to /tmp for C profiler to use
        std::fs::write("/tmp/profile_test.webp", &webp).expect("Failed to write WebP");
        eprintln!("Saved to /tmp/profile_test.webp");
        webp
    };

    eprintln!("WebP size: {} bytes", webp.len());

    let iterations = if args.len() > 2 {
        args[2].parse::<u32>().unwrap_or(10)
    } else {
        10
    };

    // Warm up
    std::hint::black_box(zenwebp::decode_rgb(&webp).unwrap());

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(zenwebp::decode_rgb(&webp).unwrap());
    }
    let elapsed = start.elapsed();
    let per_decode = elapsed / iterations;
    eprintln!(
        "Decoded {} times in {:?} ({:?}/decode)",
        iterations, elapsed, per_decode
    );
}
