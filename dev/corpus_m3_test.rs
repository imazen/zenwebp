//! Corpus comparison at method 3
use std::env;
use std::fs;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let args: Vec<_> = env::args().collect();
    let dir = if let Some(d) = args.get(1) {
        std::path::PathBuf::from(d)
    } else {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        corpus
            .get("CID22/CID22-512/validation")
            .expect("corpus path unavailable")
    };

    let mut zen_total = 0u64;
    let mut lib_total = 0u64;
    let mut count = 0;

    let entries: Vec<_> = fs::read_dir(&dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .collect();

    for entry in entries {
        let path = entry.path();
        if path.extension().map(|e| e != "png").unwrap_or(true) {
            continue;
        }

        let file = fs::File::open(&path).unwrap();
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = match decoder.read_info() {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let rgb = buf[..info.buffer_size()].to_vec();
        let (w, h) = (info.width, info.height);

        let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
            .with_method(3) // Method 3
            .with_sns_strength(0)
            .with_filter_strength(0)
            .with_segments(1);
        let zen = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
            .encode()
            .unwrap();

        let lib = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(3) // Method 3
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1)
            .encode_rgb(&rgb, w, h, webpx::Unstoppable)
            .unwrap();

        zen_total += zen.len() as u64;
        lib_total += lib.len() as u64;
        count += 1;

        let ratio = zen.len() as f64 / lib.len() as f64;
        let name = path.file_name().unwrap().to_str().unwrap();
        if !(0.99..=1.01).contains(&ratio) {
            println!(
                "{}: zen={} lib={} ({:.3}x)",
                name,
                zen.len(),
                lib.len(),
                ratio
            );
        }
    }

    println!("\n=== Corpus Results (m3, Q75, SNS=0) ===");
    println!("Directory: {}", dir.display());
    println!("Images: {}", count);
    println!("zenwebp total: {} bytes", zen_total);
    println!("libwebp total: {} bytes", lib_total);
    println!("Ratio: {:.4}x", zen_total as f64 / lib_total as f64);
}
