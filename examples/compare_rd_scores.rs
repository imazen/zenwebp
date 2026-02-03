// Compare RD score components for a specific macroblock
//
// This directly computes the I16 and I4 RD scores and their components
// to understand why the mode decisions differ.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (debug_mbx, debug_mby) = if args.len() >= 3 {
        (args[1].parse().unwrap_or(25), args[2].parse().unwrap_or(4))
    } else {
        (25, 4) // A macroblock with significant coefficients
    };

    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    println!(
        "=== RD Score Comparison for MB({}, {}) ===\n",
        debug_mbx, debug_mby
    );

    // Compare file sizes to quantify the impact
    use zenwebp::{EncoderConfig, Preset};

    let zen = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let lib = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    println!("Output sizes:");
    println!("  zenwebp: {} bytes", zen.len());
    println!("  libwebp: {} bytes", lib.len());
    println!("  ratio: {:.3}x", zen.len() as f64 / lib.len() as f64);
    println!();

    // Decode to get actual coefficients
    use zenwebp::decoder::vp8::Vp8Decoder;
    use zenwebp::decoder::LumaMode;

    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    let mb_idx = debug_mby * zen_diag.mb_width as usize + debug_mbx;
    let zen_mb = &zen_diag.macroblocks[mb_idx];
    let lib_mb = &lib_diag.macroblocks[mb_idx];

    println!("Mode decisions:");
    println!("  zenwebp: {:?}", zen_mb.luma_mode);
    println!("  libwebp: {:?}", lib_mb.luma_mode);
    println!();

    // Count coefficients more carefully
    let zen_y_nz: usize = zen_mb
        .y_blocks
        .iter()
        .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
        .sum();
    let zen_y2_nz = zen_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();

    let lib_y_nz: usize = lib_mb
        .y_blocks
        .iter()
        .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
        .sum();
    let lib_y2_nz = lib_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();

    println!("Coefficient counts (nonzero):");
    println!("  zenwebp: Y={}, Y2={}", zen_y_nz, zen_y2_nz);
    println!("  libwebp: Y={}, Y2={}", lib_y_nz, lib_y2_nz);
    println!();

    // For I4 mode, Y blocks contain all 16 coefficients (DC+AC)
    // For I16 mode, Y blocks contain AC only (coeff[0]=0), Y2 contains DC transform
    if zen_mb.luma_mode == LumaMode::B {
        println!("zenwebp (I4) coefficient breakdown:");
        for (i, b) in zen_mb.y_blocks.iter().enumerate() {
            let nz = b.levels.iter().filter(|&&l| l != 0).count();
            let dc = b.levels[0];
            if nz > 0 {
                let ac_sum: i32 = b.levels[1..].iter().map(|l| l.abs()).sum();
                println!("  Block {}: dc={:3}, ac_sum={:3}, nz={}", i, dc, ac_sum, nz);
            }
        }
    }
    if lib_mb.luma_mode != LumaMode::B {
        println!("libwebp (I16) coefficient breakdown:");
        println!("  Y2 (DC transform): {:?}", &lib_mb.y2_block.levels[..4]); // First 4 DC values
        for (i, b) in lib_mb.y_blocks.iter().enumerate() {
            let nz = b.levels.iter().filter(|&&l| l != 0).count();
            if nz > 0 {
                let ac_sum: i32 = b.levels[1..].iter().map(|l| l.abs()).sum();
                println!("  Y1[{}] AC: sum={:3}, nz={}", i, ac_sum, nz);
            }
        }
    }

    println!("\n=== Hypotheses ===");
    println!();

    // Check if I4 is producing more bits
    let zen_total_bits = estimate_bits(&zen_mb.y_blocks, zen_mb.luma_mode == LumaMode::B);
    let lib_total_bits = estimate_bits(&lib_mb.y_blocks, lib_mb.luma_mode == LumaMode::B);

    println!("Estimated bit cost (rough):");
    println!("  zenwebp: {} bits", zen_total_bits);
    println!("  libwebp: {} bits", lib_total_bits);
    println!();

    if zen_mb.luma_mode == LumaMode::B && lib_mb.luma_mode != LumaMode::B {
        println!("CASE: zenwebp chose I4, libwebp chose I16");
        println!();
        println!("This means zenwebp's RD score for I4 was lower than I16.");
        println!("Possible causes:");
        println!("  1. I4 SSE was lower (prediction was better)");
        println!("  2. I16 SSE was higher (prediction was worse)");
        println!("  3. I4 coefficient cost was underestimated");
        println!("  4. I16 coefficient cost was overestimated");
        println!();
        println!(
            "Total bits: zen I4 used {} bits, lib I16 used {} bits",
            zen_total_bits, lib_total_bits
        );
        if zen_total_bits > lib_total_bits {
            println!("  -> I4 used MORE bits, so SSE must have been much lower");
        } else {
            println!("  -> I4 used FEWER bits, rate savings justified the choice");
        }
    }
}

fn estimate_bits(blocks: &[zenwebp::decoder::vp8::BlockDiagnostic], is_i4: bool) -> u32 {
    let mut bits = 0u32;
    for b in blocks {
        for &l in &b.levels {
            if l != 0 {
                // Rough estimate: each nonzero costs ~3-5 bits + level encoding
                bits += 3 + (l.unsigned_abs() as u32).max(1).ilog2();
            }
        }
    }
    // I4 has 16 mode signaling costs (roughly 1-3 bits each)
    if is_i4 {
        bits += 16 * 2;
    }
    // I16 has 1 mode signaling cost (2-3 bits)
    else {
        bits += 3;
    }
    bits
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
