fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let corpus = codec_corpus::Corpus::new().unwrap();
        let dir = corpus.get("CID22/CID22-512/validation").unwrap();
        dir.join("792079.png").to_string_lossy().into_owned()
    });
    let data = std::fs::read(&path).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(&data));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].chunks(3).flat_map(|c| [c[0],c[1],c[2],255]).collect(),
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        _ => panic!("unsupported color type"),
    };
    eprintln!("{}x{} RGBA, {} pixels", info.width, info.height, info.width * info.height);
    let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
    let out = zenwebp::EncodeRequest::new(&config, &rgba, zenwebp::PixelLayout::Rgba8, info.width, info.height)
        .encode()
        .unwrap();
    eprintln!("encoded {} bytes (method 4 lossless)", out.len());
}
