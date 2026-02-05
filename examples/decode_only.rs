// Decode-only benchmark for profiling
use zenwebp::EncoderConfig;

fn main() {
    // Generate a test image (gradient)
    let width = 1024u32;
    let height = 1024u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 255) / width) as u8);
            rgb.push(((y * 255) / height) as u8);
            rgb.push((((x + y) * 127) / (width + height)) as u8);
        }
    }
    
    // Encode to WebP once
    let webp = EncoderConfig::new()
        .quality(75.0)
        .encode_rgb(&rgb, width, height)
        .unwrap();
    eprintln!("Encoded 1024x1024 to {} bytes", webp.len());
    
    // Start profiling from here
    eprintln!("Starting 100 decodes...");
    
    // Decode 100 times
    for _ in 0..100 {
        std::hint::black_box(zenwebp::decode_rgb(&webp).unwrap());
    }
    eprintln!("Done");
}
