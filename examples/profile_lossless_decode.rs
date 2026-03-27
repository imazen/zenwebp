fn main() {
    let corpus = codec_corpus::Corpus::new().unwrap();
    let dir = corpus.get("gb82-sc").unwrap();
    let png_path = dir.join("codec_wiki.png");
    let data = std::fs::read(&png_path).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(&data));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        _ => panic!("unsupported"),
    };
    // Encode lossless
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
    // Save for C decoder
    std::fs::write("/tmp/codec_wiki_lossless.webp", &webp).unwrap();
    eprintln!(
        "encoded {}x{} lossless: {} bytes",
        info.width,
        info.height,
        webp.len()
    );
    // Decode 5 times
    for _ in 0..5 {
        let config = zenwebp::DecodeConfig::default();
        let (pixels, _w, _h) = zenwebp::DecodeRequest::new(&config, &webp)
            .decode_rgba()
            .unwrap();
        std::hint::black_box(&pixels);
    }
    eprintln!("decoded 5x");
}
