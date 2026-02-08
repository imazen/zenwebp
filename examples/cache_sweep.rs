fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if let Some(p) = args.get(1) {
        p.to_string()
    } else {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        corpus
            .get("CID22/CID22-512/training")
            .expect("corpus path unavailable")
            .join("1183021.png")
            .to_string_lossy()
            .to_string()
    };

    let file = std::fs::File::open(&path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let has_alpha = info.color_type == png::ColorType::Rgba;
    let (w, h) = (info.width, info.height);

    println!("Image: {} ({}x{}, alpha={})", path, w, h, has_alpha);
    println!("{:-<60}", "");

    for cache_bits in 0..=10u8 {
        let config = zenwebp::encoder::vp8l::Vp8lConfig {
            quality: zenwebp::encoder::vp8l::Vp8lQuality {
                quality: 75,
                method: 4,
            },
            use_predictor: true,
            use_cross_color: true,
            use_subtract_green: true,
            use_palette: true,
            cache_bits: Some(cache_bits),
            ..Default::default()
        };
        match zenwebp::encoder::vp8l::encode_vp8l(
            &rgb,
            w,
            h,
            has_alpha,
            &config,
            &enough::Unstoppable,
        ) {
            Ok(vp8l) => {
                let total = 12 + 8 + vp8l.len() + (vp8l.len() % 2);
                println!("cache_bits={:>2}: {:>8} bytes", cache_bits, total);
            }
            Err(e) => println!("cache_bits={:>2}: ERROR {:?}", cache_bits, e),
        }
    }
}
