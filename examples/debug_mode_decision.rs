// Debug a specific macroblock's I4 vs I16 decision
//
// This prints detailed RD score components to understand why
// zenwebp picks I4 when libwebp picks I16.

use zenwebp::encoder::cost::{calc_lambda_i16, calc_lambda_i4, calc_lambda_mode, RD_DISTO_MULT};

fn main() {
    // Find a disputed macroblock with nonzero coefficients
    let args: Vec<String> = std::env::args().collect();
    let (debug_mbx, debug_mby) = if args.len() >= 3 {
        (args[1].parse().unwrap_or(14), args[2].parse().unwrap_or(1))
    } else {
        (14, 1) // default
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
        "=== Mode Decision Debug for MB({}, {}) ===\n",
        debug_mbx, debug_mby
    );

    // Get quantizer at Q75
    // Q75 → q_index = (127 - 75) = 52 for luma
    let q = 52u32;
    let lambda_i16 = calc_lambda_i16(q);
    let lambda_i4 = calc_lambda_i4(q);
    let lambda_mode = calc_lambda_mode(q);

    println!("Lambda values at Q75 (q=52):");
    println!("  lambda_i16 = {}", lambda_i16);
    println!("  lambda_i4 = {}", lambda_i4);
    println!("  lambda_mode = {}", lambda_mode);
    println!("  RD_DISTO_MULT = {}", RD_DISTO_MULT);
    println!();

    // The I4 vs I16 comparison uses lambda_mode for both
    // I16 score: (mode_cost + coeff_cost) * lambda_mode + RD_DISTO_MULT * sse
    // I4 score: 211 * lambda_mode + sum_over_blocks[(mode_cost + coeff_cost) * lambda_mode + RD_DISTO_MULT * sse]

    // BMODE_COST penalty for I4 (fixed overhead)
    const BMODE_COST: u64 = 211;
    let i4_initial_penalty = BMODE_COST * u64::from(lambda_mode);
    println!("I4 initial penalty (BMODE_COST * lambda_mode):");
    println!(
        "  {} * {} = {}",
        BMODE_COST, lambda_mode, i4_initial_penalty
    );
    println!();

    // Show what the threshold means
    // I4 beats I16 if: i4_score < i16_score
    // At minimum, I4 starts with i4_initial_penalty, so:
    // For I4 to win with minimal coefficients, I16 must have score > i4_initial_penalty + (I4 block costs)

    // Encode and compare
    use zenwebp::decoder::vp8::Vp8Decoder;
    use zenwebp::decoder::LumaMode;
    use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);
    let zen_webp = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    let lib_webp = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    // Decode both
    let zen_vp8 = extract_vp8(&zen_webp).unwrap();
    let lib_vp8 = extract_vp8(&lib_webp).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    let mb_idx = debug_mby * zen_diag.mb_width as usize + debug_mbx;
    let zen_mb = &zen_diag.macroblocks[mb_idx];
    let lib_mb = &lib_diag.macroblocks[mb_idx];

    println!("Encoded macroblock comparison:");
    println!("  zenwebp: {:?}", zen_mb.luma_mode);
    println!("  libwebp: {:?}", lib_mb.luma_mode);
    println!();

    // Count nonzero coefficients
    let zen_nz: usize = zen_mb
        .y_blocks
        .iter()
        .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
        .sum();
    let lib_nz: usize = lib_mb
        .y_blocks
        .iter()
        .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
        .sum();

    println!("Nonzero Y coefficients:");
    println!("  zenwebp ({:?}): {}", zen_mb.luma_mode, zen_nz);
    println!("  libwebp ({:?}): {}", lib_mb.luma_mode, lib_nz);
    println!();

    // For zenwebp I4, show per-block breakdown
    if zen_mb.luma_mode == LumaMode::B {
        println!("zenwebp I4 block modes: {:?}", zen_mb.bpred_modes);
        println!("Per-block nonzero counts:");
        for (i, b) in zen_mb.y_blocks.iter().enumerate() {
            let nz = b.levels.iter().filter(|&&l| l != 0).count();
            if nz > 0 {
                println!("  Block {}: {} nonzero", i, nz);
            }
        }
    }

    // For libwebp I16, show the DC block
    if lib_mb.luma_mode != LumaMode::B {
        println!("\nlibwebp I16 mode: {:?}", lib_mb.luma_mode);
        let dc_nz = lib_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
        println!("  Y2 (DC) block: {} nonzero", dc_nz);
        println!("Per Y1 block nonzero counts:");
        for (i, b) in lib_mb.y_blocks.iter().enumerate() {
            let nz = b.levels.iter().filter(|&&l| l != 0).count();
            if nz > 0 {
                println!("  Block {}: {} nonzero (AC only)", i, nz);
            }
        }
    }

    // For I16, also count Y2 (DC block) coefficients
    let lib_y2_nz = lib_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
    let lib_total_nz = lib_nz + lib_y2_nz;

    println!("\n=== Coefficient Analysis ===");
    println!("zenwebp I4: {} total nonzero coefficients", zen_nz);
    println!(
        "libwebp I16: {} Y1 AC + {} Y2 DC = {} total nonzero",
        lib_nz, lib_y2_nz, lib_total_nz
    );
    println!();

    // Sum the absolute values to estimate actual bit cost
    let zen_sum: i32 = zen_mb
        .y_blocks
        .iter()
        .flat_map(|b| b.levels.iter())
        .map(|&l| l.abs())
        .sum();
    let lib_y1_sum: i32 = lib_mb
        .y_blocks
        .iter()
        .flat_map(|b| b.levels.iter())
        .map(|&l| l.abs())
        .sum();
    let lib_y2_sum: i32 = lib_mb.y2_block.levels.iter().map(|&l| l.abs()).sum();

    println!("Sum of |coefficients|:");
    println!("  zenwebp I4: {}", zen_sum);
    println!("  libwebp I16 Y1: {}", lib_y1_sum);
    println!("  libwebp I16 Y2: {}", lib_y2_sum);
    println!("  libwebp total: {}", lib_y1_sum + lib_y2_sum);

    println!("\n=== Analysis ===");
    println!();
    println!("The key question: why does zenwebp's I4 RD score beat I16?");
    println!();
    println!("For I4 to win over I16:");
    println!("  I4_penalty + sum(I4 block costs) < I16 score");
    println!("  {} + sum(I4 blocks) < I16 score", i4_initial_penalty);
    println!();

    if zen_nz > lib_total_nz {
        println!(
            "I4 has MORE coefficients ({} vs {}) - should have higher rate!",
            zen_nz, lib_total_nz
        );
        println!("This means either:");
        println!("  1. I4 distortion is much lower (saving on distortion term)");
        println!("  2. Our I16 cost estimation is wrong");
    } else {
        println!(
            "I4 has FEWER coefficients ({} vs {}) - rate may be lower",
            zen_nz, lib_total_nz
        );
        println!(
            "The question is whether the I4 mode signaling overhead (BMODE_COST + per-block modes)"
        );
        println!("plus distortion justifies the choice.");
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
