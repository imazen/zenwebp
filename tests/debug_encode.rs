#![cfg(not(target_arch = "wasm32"))]
//! Debug encoder output
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

#[test]
#[ignore]
fn debug_encode_cid22_first() {
    // First CID22-512 validation photo (Kodak is no longer in codec-corpus).
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let dir = corpus
        .get("CID22/CID22-512/validation")
        .expect("CID22-512 validation corpus unavailable");
    let mut images: Vec<_> = std::fs::read_dir(&dir)
        .expect("read CID22 dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    images.sort();
    let path = images.swap_remove(0);
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => panic!("Unsupported color type"),
    };

    println!("\n=== Comparing encode sizes ===");
    println!(
        "Image: {}x{} = {} pixels",
        info.width,
        info.height,
        info.width * info.height
    );

    for q in [50u8, 75, 90, 95] {
        // Ours
        let config = EncoderConfig::new_lossy().with_quality(q as f32);
        let our_output = EncodeRequest::new(
            &config,
            &rgb_data,
            PixelLayout::Rgb8,
            info.width,
            info.height,
        )
        .encode()
        .unwrap();

        // libwebp
        let lib_encoder = webp::Encoder::from_rgb(&rgb_data, info.width, info.height);
        let lib_output = lib_encoder.encode(q as f32).to_vec();

        let ratio = 100.0 * our_output.len() as f64 / lib_output.len() as f64;
        println!(
            "Q{}: ours={} bytes, libwebp={} bytes, ratio={:.1}%",
            q,
            our_output.len(),
            lib_output.len(),
            ratio
        );

        // Save both for inspection
        std::fs::write(format!("/tmp/ours_q{}.webp", q), &our_output).unwrap();
        std::fs::write(format!("/tmp/libwebp_q{}.webp", q), &lib_output).unwrap();
    }
}
