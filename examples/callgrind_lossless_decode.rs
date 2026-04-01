//! Decode-only lossless profiling. Takes a PNG, encodes to WebP, saves to /tmp,
//! then decodes the WebP N times (the part we're profiling).
//! Usage: target/release/examples/callgrind_lossless_decode <png> [iterations]

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let png_path = args
        .get(1)
        .expect("usage: callgrind_lossless_decode <png> [iterations]");
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    // Load PNG
    let file = std::fs::File::open(png_path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        _ => panic!("unsupported color type"),
    };

    // Encode
    let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
    let webp = zenwebp::EncodeRequest::new(
        &config,
        &rgba,
        zenwebp::PixelLayout::Rgba8,
        info.width,
        info.height,
    )
    .encode()
    .unwrap();
    eprintln!("{}x{} → {} bytes", info.width, info.height, webp.len());

    // Decode N times
    let dc = zenwebp::DecodeConfig::default();
    for _ in 0..iterations {
        let r = zenwebp::DecodeRequest::new(&dc, &webp)
            .decode_rgba()
            .unwrap();
        std::hint::black_box(&r);
    }
    eprintln!("decoded {iterations} times");
}
