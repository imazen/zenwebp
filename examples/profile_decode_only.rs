//! Profile decoder only - no encoding overhead

use std::fs;

fn main() {
    // Use pre-encoded webp file or encode once
    let webp_path = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/profile_test.webp".to_string());

    let webp_data = if std::path::Path::new(&webp_path).exists() {
        fs::read(&webp_path).expect("Failed to read webp")
    } else {
        // Encode from PNG
        let png_path = std::env::args()
            .nth(2)
            .unwrap_or("/tmp/clic2025/11f2b039b293758398b1a7a8afa64bb2.png".to_string());
        let file = fs::File::open(&png_path).expect("Failed to open PNG");
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
            _ => panic!("Unsupported"),
        };

        let webp = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(5)
            .encode_rgb(&rgb, info.width, info.height, webpx::Unstoppable)
            .unwrap();
        fs::write("/tmp/profile_test.webp", &webp).ok();
        webp
    };

    eprintln!("WebP size: {} bytes", webp_data.len());

    // Decode many times for profiling
    for _ in 0..50 {
        let _ = zenwebp::decode_rgb(&webp_data).unwrap();
    }
}
