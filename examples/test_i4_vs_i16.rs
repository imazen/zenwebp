// Analyze where I4 vs I16 decisions differ
use zenwebp::decoder::vp8::Vp8Decoder;
use zenwebp::decoder::LumaMode;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    // Encode with both
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_segments(1);
    let zen_webp = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    let lib_webp = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_filter_sharpness(0)
        .with_segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    // Decode both
    let zen_vp8 = extract_vp8(&zen_webp).unwrap();
    let lib_vp8 = extract_vp8(&lib_webp).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Count where decisions differ
    let mut zen_i4_lib_i16 = 0;
    let mut zen_i16_lib_i4 = 0;
    let mut both_i4 = 0;
    let mut both_i16 = 0;

    let mut first_zen_i4_lib_i16 = Vec::new();
    let mut first_zen_i16_lib_i4 = Vec::new();

    for (i, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        let mbx = i % zen_diag.mb_width as usize;
        let mby = i / zen_diag.mb_width as usize;

        let zen_is_i4 = z.luma_mode == LumaMode::B;
        let lib_is_i4 = l.luma_mode == LumaMode::B;

        match (zen_is_i4, lib_is_i4) {
            (true, true) => both_i4 += 1,
            (false, false) => both_i16 += 1,
            (true, false) => {
                zen_i4_lib_i16 += 1;
                if first_zen_i4_lib_i16.len() < 10 {
                    first_zen_i4_lib_i16.push((mbx, mby));
                }
            }
            (false, true) => {
                zen_i16_lib_i4 += 1;
                if first_zen_i16_lib_i4.len() < 10 {
                    first_zen_i16_lib_i4.push((mbx, mby));
                }
            }
        }
    }

    println!("I4 vs I16 decision comparison:");
    println!("  Both I4:       {}", both_i4);
    println!("  Both I16:      {}", both_i16);
    println!(
        "  Zen I4, Lib I16: {} (zen picks I4, lib picks I16)",
        zen_i4_lib_i16
    );
    println!(
        "  Zen I16, Lib I4: {} (zen picks I16, lib picks I4)",
        zen_i16_lib_i4
    );

    println!("\nFirst cases where zen picks I4 but lib picks I16:");
    for (mbx, mby) in &first_zen_i4_lib_i16 {
        println!("  MB({}, {})", mbx, mby);
    }

    println!("\nFirst cases where zen picks I16 but lib picks I4:");
    for (mbx, mby) in &first_zen_i16_lib_i4 {
        println!("  MB({}, {})", mbx, mby);
    }

    // Count nonzero coefficients in disputed blocks
    let mut zen_nz_where_zen_i4_lib_i16 = 0usize;
    let mut lib_nz_where_zen_i4_lib_i16 = 0usize;

    for (z, l) in zen_diag.macroblocks.iter().zip(lib_diag.macroblocks.iter()) {
        let zen_is_i4 = z.luma_mode == LumaMode::B;
        let lib_is_i4 = l.luma_mode == LumaMode::B;

        if zen_is_i4 && !lib_is_i4 {
            // Count nonzero Y coefficients
            for blk in &z.y_blocks {
                zen_nz_where_zen_i4_lib_i16 += blk.levels.iter().filter(|&&l| l != 0).count();
            }
            for blk in &l.y_blocks {
                lib_nz_where_zen_i4_lib_i16 += blk.levels.iter().filter(|&&l| l != 0).count();
            }
        }
    }

    if zen_i4_lib_i16 > 0 {
        println!("\nIn disputed blocks (zen=I4, lib=I16):");
        println!(
            "  Avg zen Y nonzero per MB: {:.1}",
            zen_nz_where_zen_i4_lib_i16 as f64 / zen_i4_lib_i16 as f64
        );
        println!(
            "  Avg lib Y nonzero per MB: {:.1}",
            lib_nz_where_zen_i4_lib_i16 as f64 / zen_i4_lib_i16 as f64
        );
    }

    // Find disputed blocks with nonzero coefficients
    println!("\nDisputed blocks (zen=I4, lib=I16) with nonzero coefficients:");
    for (i, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        let mbx = i % zen_diag.mb_width as usize;
        let mby = i / zen_diag.mb_width as usize;

        let zen_is_i4 = z.luma_mode == LumaMode::B;
        let lib_is_i4 = l.luma_mode == LumaMode::B;

        if zen_is_i4 && !lib_is_i4 {
            let zen_nz: usize = z
                .y_blocks
                .iter()
                .map(|b| b.levels.iter().filter(|&&lv| lv != 0).count())
                .sum();
            let lib_nz: usize = l
                .y_blocks
                .iter()
                .map(|b| b.levels.iter().filter(|&&lv| lv != 0).count())
                .sum();
            if zen_nz > 0 || lib_nz > 0 {
                println!(
                    "  MB({:2}, {:2}): zen_nz={:2}, lib_nz={:2}, zen_modes={:?}",
                    mbx, mby, zen_nz, lib_nz, z.bpred_modes
                );
            }
        }
    }
}

fn extract_vp8(webp: &[u8]) -> Option<Vec<u8>> {
    if webp.len() < 12 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= webp.len() {
        let fourcc = &webp[pos..pos + 4];
        let size = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().ok()?) as usize;
        if fourcc == b"VP8 " {
            let end = (pos + 8 + size).min(webp.len());
            return Some(webp[pos + 8..end].to_vec());
        }
        pos += 8 + size + (size & 1);
    }
    None
}
