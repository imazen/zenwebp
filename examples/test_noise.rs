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
fn test(name: &str, rgb: &[u8], w: u32, h: u32, config: &zenwebp::encoder::vp8l::Vp8lConfig) {
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(rgb, w, h, false, config, &enough::Unstoppable).unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
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
    println!("  {}: size={} mismatches={}", name, webp.len(), mismatches);
}
fn main() {
    let w = 512u32;
    let h = 512u32;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed = 42u64;
    for b in rgb.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (seed >> 33) as u8;
    }

    println!("Random noise 512x512:");

    // No transforms
    test(
        "none",
        &rgb,
        w,
        h,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: false,
            use_subtract_green: false,
            use_palette: false,
            ..Default::default()
        },
    );

    // Subtract green only
    test(
        "sg_only",
        &rgb,
        w,
        h,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: false,
            use_subtract_green: true,
            use_palette: false,
            ..Default::default()
        },
    );

    // Predictor only with bits=9 (1 block, trivial)
    test(
        "pred_bits9",
        &rgb,
        w,
        h,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 9,
            ..Default::default()
        },
    );

    // Predictor only with bits=4 (32x32 blocks)
    test(
        "pred_bits4",
        &rgb,
        w,
        h,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );

    // Try different quality levels for LZ77
    for q in [0, 25, 50, 75, 100] {
        let mut config = zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        };
        config.quality.quality = q;
        test(&format!("pred_bits4_q{}", q), &rgb, w, h, &config);
    }

    // Smaller image sizes
    for sz in [64, 128, 256] {
        let small_rgb: Vec<u8> = rgb[..(sz * sz * 3) as usize].to_vec();
        test(
            &format!("pred_bits4_{}x{}", sz, sz),
            &small_rgb,
            sz,
            sz,
            &zenwebp::encoder::vp8l::Vp8lConfig {
                use_predictor: true,
                use_subtract_green: false,
                use_palette: false,
                predictor_bits: 4,
                ..Default::default()
            },
        );
    }
}
