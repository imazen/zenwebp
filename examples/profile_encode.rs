fn main() {
    let w = 512u32;
    let h = 512u32;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed = 12345u32;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            rgb[i] = ((x + y + (seed >> 16)) & 0xff) as u8;
            rgb[i + 1] = ((x * 3 + y * 5 + (seed >> 20)) & 0xff) as u8;
            rgb[i + 2] = ((x * 7 + y * 2 + (seed >> 24)) & 0xff) as u8;
        }
    }
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(75.0)
        .with_method(4);
    for _ in 0..5 {
        let out = zenwebp::EncodeRequest::new(&config, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
            .encode()
            .unwrap();
        std::hint::black_box(&out);
    }
    eprintln!("encoded 5x");
}
