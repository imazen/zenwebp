// Decode benchmark with noisy image (more coefficients)
use zenwebp::{PixelLayout, EncodeRequest, EncoderConfig};

fn main() {
    // Generate a noisy test image (random noise)
    let width = 1024u32;
    let height = 1024u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    let mut seed = 12345u64;
    for _ in 0..(width * height * 3) {
        // Simple PRNG
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        rgb.push((seed >> 56) as u8);
    }

    // Encode to WebP once
    let _cfg = EncoderConfig::new().quality(75.0);
    let webp = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap();
    eprintln!("Encoded noisy 1024x1024 to {} bytes", webp.len());

    // Decode 100 times
    for _ in 0..100 {
        std::hint::black_box(zenwebp::decode_rgb(&webp).unwrap());
    }
    eprintln!("Done");
}
