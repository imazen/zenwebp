fn main() {
    let webp = std::fs::read("/tmp/codec_wiki_lossless.webp").expect("run profile_lossless_decode first to create the file");
    eprintln!("loaded {} bytes", webp.len());
    for _ in 0..5 {
        let config = zenwebp::DecodeConfig::default();
        let (pixels, _w, _h) = zenwebp::DecodeRequest::new(&config, &webp).decode_rgba().unwrap();
        std::hint::black_box(&pixels);
    }
    eprintln!("decoded 5x");
}
