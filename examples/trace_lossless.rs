/// Trace lossless encoding call counts.
/// Uses a synthetic 512x512 image for reproducible comparison with libwebp.
fn main() {
    // SAFETY: single-threaded at program start, no other threads reading env
    unsafe { std::env::set_var("ZENWEBP_TRACE", "1") };

    let w = 512u32;
    let h = 512u32;
    // Same synthetic image as libwebp C profiler: (i * 7 + 13) & 0xff
    let rgba: Vec<u8> = (0..(w * h * 4) as usize)
        .map(|i| ((i * 7 + 13) & 0xff) as u8)
        .collect();

    eprintln!("{}x{} RGBA, {} pixels (synthetic)", w, h, w * h);

    let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
    let out = zenwebp::EncodeRequest::new(&config, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    eprintln!("encoded {} bytes (method 4 lossless)", out.len());
}
