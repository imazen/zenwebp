//! Minimal lossless decode for callgrind profiling.
//! Usage: valgrind --tool=callgrind target/release/examples/callgrind_lossless <png> [iterations]
//!
//! Pre-encodes to a temp WebP, then profiles only the decode loop.

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let png_path = args.get(1).expect("usage: callgrind_lossless <png> [iterations]");
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

    // Encode lossless (outside profiling hot path — callgrind_toggle_collect)
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
    eprintln!(
        "encoded {}x{} → {} bytes lossless",
        info.width,
        info.height,
        webp.len()
    );

    // Warmup decode (outside measurement)
    let decode_config = zenwebp::DecodeConfig::default();
    let _ = zenwebp::DecodeRequest::new(&decode_config, &webp)
        .decode_rgba()
        .unwrap();

    // Decode N times — this is what we're profiling
    for _ in 0..iterations {
        let result = zenwebp::DecodeRequest::new(&decode_config, &webp)
            .decode_rgba()
            .unwrap();
        std::hint::black_box(&result);
    }
    eprintln!("decoded {iterations} times");
}
