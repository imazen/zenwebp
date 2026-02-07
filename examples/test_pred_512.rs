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
fn test(name: &str, rgb: &[u8], w: u32, h: u32, bits: u8) {
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: true,
        use_subtract_green: false,
        use_palette: false,
        predictor_bits: bits,
        ..Default::default()
    };
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(rgb, w, h, false, &config, &enough::Unstoppable)
        .unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    let path = format!("/tmp/test_pred_512_{}.webp", name);
    std::fs::write(&path, &webp).unwrap();
    let dwebp_ok = std::process::Command::new("/home/lilith/work/libwebp/examples/dwebp")
        .args([&path, "-o", &format!("{}.ppm", path)])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let mismatches = match zenwebp::decode_rgba(&webp) {
        Ok((data, _, _)) => (0..(w * h) as usize)
            .filter(|&i| {
                data[i * 4] != rgb[i * 3]
                    || data[i * 4 + 1] != rgb[i * 3 + 1]
                    || data[i * 4 + 2] != rgb[i * 3 + 2]
            })
            .count(),
        Err(e) => {
            println!("  {} ERROR: {:?}", name, e);
            usize::MAX
        }
    };
    println!(
        "  {}: size={} dwebp={} mismatches={}",
        name,
        webp.len(),
        if dwebp_ok { "OK" } else { "FAIL" },
        mismatches
    );
}
fn main() {
    let w = 512u32;
    let h = 512u32;

    // Solid gray (1 mode)
    let rgb1: Vec<u8> = [128, 128, 128].repeat((w * h) as usize);
    test("solid", &rgb1, w, h, 4);

    // Horizontal gradient (likely all Left = 1 mode)
    let mut rgb2 = Vec::new();
    for _y in 0..h {
        for x in 0..w {
            rgb2.extend_from_slice(&[(x / 2) as u8, 100, 100]);
        }
    }
    test("hgrad", &rgb2, w, h, 4);

    // Random noise (many modes)
    let mut rgb3 = vec![0u8; (w * h * 3) as usize];
    let mut seed = 42u64;
    for b in rgb3.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (seed >> 33) as u8;
    }
    test("noise", &rgb3, w, h, 4);

    // Two halves (2 modes)
    let mut rgb4 = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if y < h / 2 {
                rgb4.extend_from_slice(&[(x / 2) as u8, 100, 100]);
            } else {
                rgb4.extend_from_slice(&[100, (y / 2) as u8, 100]);
            }
        }
    }
    test("halves", &rgb4, w, h, 4);

    // Gradient both directions (should trigger multiple modes)
    let mut rgb5 = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb5.extend_from_slice(&[(x / 2) as u8, (y / 2) as u8, 100]);
        }
    }
    test("bigrad", &rgb5, w, h, 4);
}
