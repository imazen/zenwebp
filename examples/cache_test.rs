//! Test different cache sizes for outlier image
//! Usage: cargo run --release --example cache_test
use std::fs;

fn wrap_vp8l_in_riff(vp8l_data: &[u8]) -> Vec<u8> {
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + vp8l_data.len() + (vp8l_data.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(vp8l_data.len() as u32).to_le_bytes());
    webp.extend_from_slice(vp8l_data);
    if !vp8l_data.len().is_multiple_of(2) {
        webp.push(0);
    }
    webp
}

fn main() {
    let png_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/lilith/work/codec-corpus/CID22/CID22-512/training/3616956.png".to_string()
    });

    // Load PNG
    let file = fs::File::open(&png_path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let width = info.width;
    let height = info.height;
    let has_alpha = info.color_type == png::ColorType::Rgba;

    println!(
        "Image: {} ({}x{}, alpha={})",
        png_path, width, height, has_alpha
    );

    // Try each forced cache size (Some(N) = search 0..=N, picks best)
    for cache_bits in 0..=10u8 {
        let config = zenwebp::encoder::vp8l::Vp8lConfig {
            quality: zenwebp::encoder::vp8l::Vp8lQuality {
                quality: 75,
                method: 4,
            },
            use_predictor: true,
            use_cross_color: true,
            use_subtract_green: true,
            use_palette: false,
            cache_bits: Some(cache_bits),
            ..Default::default()
        };

        let vp8l =
            match zenwebp::encoder::vp8l::encode_vp8l(&rgb, width, height, has_alpha, &config, &enough::Unstoppable) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("cache_bits={}: ERROR {:?}", cache_bits, e);
                    continue;
                }
            };
        let webp = wrap_vp8l_in_riff(&vp8l);
        println!("cache_bits={:2}: {} bytes", cache_bits, webp.len());
    }

    // Also test auto-detect (None)
    let config_auto = zenwebp::encoder::vp8l::Vp8lConfig {
        quality: zenwebp::encoder::vp8l::Vp8lQuality {
            quality: 75,
            method: 4,
        },
        use_predictor: true,
        use_cross_color: true,
        use_subtract_green: true,
        use_palette: false,
        cache_bits: None,
        ..Default::default()
    };
    let vp8l =
        zenwebp::encoder::vp8l::encode_vp8l(&rgb, width, height, has_alpha, &config_auto, &enough::Unstoppable).unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    println!("cache=auto: {} bytes", webp.len());
}
