//! Compare all methods (0-6) between zenwebp and libwebp apples-to-apples

use std::time::Instant;
use zenwebp::{EncodeRequest, PixelLayout};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        corpus
            .get("CID22/CID22-512/validation")
            .expect("corpus path unavailable")
            .join("792079.png")
            .to_string_lossy()
            .to_string()
    });

    let file = std::fs::File::open(&path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| &c[..3])
            .copied()
            .collect(),
        _ => panic!("Unsupported color type"),
    };
    let (w, h) = (info.width, info.height);
    let iterations = 20;

    println!("Image: {}x{} ({})", w, h, path);
    println!(
        "Settings: Q75, SNS=0, filter=0, segments=1, {} iterations\n",
        iterations
    );
    println!(
        "{:>6} {:>10} {:>8} {:>10} {:>8} {:>10} {:>10}",
        "Method", "zen_size", "zen_ms", "lib_size", "lib_ms", "size_ratio", "speed_ratio"
    );
    println!("{}", "-".repeat(76));

    for method in 0..=6u8 {
        // zenwebp
        let zen_start = Instant::now();
        let mut zen_size = 0;
        for _ in 0..iterations {
            let _cfg = zenwebp::EncoderConfig::new_lossy()
                .with_quality(75.0)
                .with_method(method)
                .with_sns_strength(0)
                .with_filter_strength(0)
                .with_segments(1);
            let out = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
                .encode()
                .unwrap();
            zen_size = out.len();
        }
        let zen_ms = zen_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // libwebp via webpx
        let lib_start = Instant::now();
        let mut lib_size = 0;
        for _ in 0..iterations {
            let out = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                .method(method)
                .sns_strength(0)
                .filter_strength(0)
                .filter_sharpness(0)
                .segments(1)
                .encode_rgb(&rgb, w, h, webpx::Unstoppable)
                .unwrap();
            lib_size = out.len();
        }
        let lib_ms = lib_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let size_ratio = zen_size as f64 / lib_size as f64;
        let speed_ratio = zen_ms / lib_ms;

        println!(
            "{:>6} {:>10} {:>8.1} {:>10} {:>8.1} {:>10.3}x {:>10.2}x",
            method, zen_size, zen_ms, lib_size, lib_ms, size_ratio, speed_ratio
        );
    }
}
