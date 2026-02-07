//! Debug: test with a pattern that forces multiple predictor modes.
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
    // Create a 32x32 image with 4 different quadrants (should force different modes)
    let width = 32u32;
    let height = 32u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            if x < 16 && y < 16 {
                // Horizontal gradient
                rgb.extend_from_slice(&[(x * 16) as u8, 128, 64]);
            } else if x >= 16 && y < 16 {
                // Vertical gradient
                rgb.extend_from_slice(&[64, (y * 16) as u8, 128]);
            } else if x < 16 && y >= 16 {
                // Flat color
                rgb.extend_from_slice(&[200, 100, 50]);
            } else {
                // Checkerboard
                let c = if (x + y) % 2 == 0 { 255 } else { 0 };
                rgb.extend_from_slice(&[c as u8, c as u8, c as u8]);
            }
        }
    }

    // Test with predictor_bits=3 (8x8 blocks, 4x4=16 blocks for this 32x32 image)
    for bits in 2..=5u8 {
        let config = zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: bits,
            ..Default::default()
        };
        let vp8l =
            zenwebp::encoder::vp8l::encode_vp8l(&rgb, width, height, false, &config, &enough::Unstoppable).unwrap();
        let webp = wrap_vp8l_in_riff(&vp8l);
        let path = format!("/tmp/test_pred_debug_bits{}.webp", bits);
        std::fs::write(&path, &webp).unwrap();

        let dwebp_ok = std::process::Command::new("/home/lilith/work/libwebp/examples/dwebp")
            .args([&path, "-o", &format!("{}.ppm", path)])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        let mismatches = match zenwebp::decode_rgba(&webp) {
            Ok((data, _, _)) => (0..(width * height) as usize)
                .filter(|&i| {
                    data[i * 4] != rgb[i * 3]
                        || data[i * 4 + 1] != rgb[i * 3 + 1]
                        || data[i * 4 + 2] != rgb[i * 3 + 2]
                })
                .count(),
            Err(e) => {
                println!("  bits={} DECODE ERROR: {:?}", bits, e);
                usize::MAX
            }
        };

        let blocks = ((width + (1 << bits) - 1) >> bits) * ((height + (1 << bits) - 1) >> bits);
        println!(
            "  bits={} blocks={} size={} dwebp={} mismatches={}/{}",
            bits,
            blocks,
            webp.len(),
            if dwebp_ok { "OK" } else { "FAIL" },
            mismatches,
            width * height
        );
    }
}
