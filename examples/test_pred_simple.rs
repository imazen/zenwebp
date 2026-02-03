//! Test: create images that force 1, 2, and 3+ unique predictor modes.
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

fn test(name: &str, rgb: &[u8], width: u32, height: u32, bits: u8) {
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: true,
        use_subtract_green: false,
        use_palette: false,
        predictor_bits: bits,
        ..Default::default()
    };
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(rgb, width, height, false, &config).unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    let path = format!("/tmp/test_pred_{}.webp", name);
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
            println!("  {} DECODE ERROR: {:?}", name, e);
            return;
        }
    };
    let blocks = ((width + (1 << bits) - 1) >> bits) * ((height + (1 << bits) - 1) >> bits);
    println!(
        "  {}: blocks={} dwebp={} mismatches={}",
        name,
        blocks,
        if dwebp_ok { "OK" } else { "FAIL" },
        mismatches
    );
}

fn main() {
    // Image 1: Uniform horizontal gradient (should all choose Left = mode 1) → 1 unique mode
    let w = 32u32;
    let h = 32u32;
    let mut rgb1 = Vec::new();
    for _y in 0..h {
        for x in 0..w {
            rgb1.extend_from_slice(&[(x * 8) as u8, 100, 100]);
        }
    }
    test("1_mode_horiz", &rgb1, w, h, 3);

    // Image 2: Uniform vertical gradient (should all choose Top = mode 2) → 1 unique mode
    let mut rgb2 = Vec::new();
    for y in 0..h {
        for _x in 0..w {
            rgb2.extend_from_slice(&[100, (y * 8) as u8, 100]);
        }
    }
    test("1_mode_vert", &rgb2, w, h, 3);

    // Image 3: Half horizontal, half vertical → 2 unique modes
    let mut rgb3 = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if y < h / 2 {
                rgb3.extend_from_slice(&[(x * 8) as u8, 100, 100]);
            } else {
                rgb3.extend_from_slice(&[100, (y * 8) as u8, 100]);
            }
        }
    }
    test("2_modes", &rgb3, w, h, 3);

    // Image 4: Three quadrants with different patterns → 3 unique modes
    let mut rgb4 = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if y < h / 2 && x < w / 2 {
                rgb4.extend_from_slice(&[(x * 8) as u8, 100, 100]);
            } else if y < h / 2 {
                rgb4.extend_from_slice(&[100, (y * 8) as u8, 100]);
            } else {
                rgb4.extend_from_slice(&[200, 200, 200]);
            } // flat
        }
    }
    test("3_modes", &rgb4, w, h, 3);

    // Image 5: Same but with bits=4 (fewer blocks)
    test("3_modes_bits4", &rgb4, w, h, 4);

    // Image 6: Just 2 blocks (bits=4, 16x16 image)
    let w2 = 16u32;
    let h2 = 32u32;
    let mut rgb6 = Vec::new();
    for y in 0..h2 {
        for x in 0..w2 {
            if y < h2 / 2 {
                rgb6.extend_from_slice(&[(x * 16) as u8, 100, 100]);
            } else {
                rgb6.extend_from_slice(&[100, (y * 8) as u8, 100]);
            }
        }
    }
    test("2blocks_16x32", &rgb6, w2, h2, 4);

    // Image 7: 4 blocks, all same content (should pick same mode)
    let mut rgb7 = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb7.extend_from_slice(&[(x * 8) as u8, (y * 8) as u8, 100]);
        }
    }
    test("4blocks_same_bits4", &rgb7, w, h, 4);
}
