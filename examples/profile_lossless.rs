fn main() {
    let w = 512usize;
    let h = 512usize;

    // Same gradient+noise pattern as profile_libwebp2.c
    let mut rgba = vec![0u8; w * h * 4];
    let mut seed: u32 = 12345;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            rgba[i] = ((x + y).wrapping_add((seed >> 16) as usize)) as u8;
            rgba[i + 1] = ((x * 3 + y * 5).wrapping_add((seed >> 20) as usize)) as u8;
            rgba[i + 2] = ((x * 7 + y * 2).wrapping_add((seed >> 24) as usize)) as u8;
            rgba[i + 3] = 255;
        }
    }

    eprintln!("{}x{} RGBA, {} pixels", w, h, w * h);
    let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
    let out = zenwebp::EncodeRequest::new(
        &config,
        &rgba,
        zenwebp::PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap();
    eprintln!("encoded {} bytes (method 4 lossless)", out.len());
}
