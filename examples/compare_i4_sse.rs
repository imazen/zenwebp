// Compare prediction SSE for I4 modes
//
// For blocks where zenwebp chose DC but libwebp chose VE,
// compare the prediction SSE to see why DC won.

use zenwebp::decoder::vp8::Vp8Decoder;
use zenwebp::decoder::{IntraMode, LumaMode};
use zenwebp::{PixelLayout, EncodeRequest, EncoderConfig, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    println!("=== I4 Mode SSE Comparison ===\n");

    // Encode with both
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);
    let zen = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    let lib = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Find blocks where zen chose DC but lib chose VE
    let mut dc_vs_ve_count = 0;
    let mut examples = Vec::new();

    for (mb_idx, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        let zen_is_i4 = z.luma_mode == LumaMode::B;
        let lib_is_i4 = l.luma_mode == LumaMode::B;

        if zen_is_i4 && lib_is_i4 {
            for (block_idx, (&zm, &lm)) in
                z.bpred_modes.iter().zip(l.bpred_modes.iter()).enumerate()
            {
                // Look for DC vs VE disagreements
                if zm == IntraMode::DC && lm == IntraMode::VE {
                    dc_vs_ve_count += 1;
                    if examples.len() < 10 {
                        let mbx = mb_idx % zen_diag.mb_width as usize;
                        let mby = mb_idx / zen_diag.mb_width as usize;

                        // Get coefficient counts
                        let zen_nz = z.y_blocks[block_idx]
                            .levels
                            .iter()
                            .filter(|&&l| l != 0)
                            .count();
                        let lib_nz = l.y_blocks[block_idx]
                            .levels
                            .iter()
                            .filter(|&&l| l != 0)
                            .count();

                        // Sum of absolute coefficients
                        let zen_sum: i32 =
                            z.y_blocks[block_idx].levels.iter().map(|&l| l.abs()).sum();
                        let lib_sum: i32 =
                            l.y_blocks[block_idx].levels.iter().map(|&l| l.abs()).sum();

                        examples.push((mbx, mby, block_idx, zen_nz, lib_nz, zen_sum, lib_sum));
                    }
                }
            }
        }
    }

    println!("Blocks where zen=DC but lib=VE: {}\n", dc_vs_ve_count);

    if !examples.is_empty() {
        println!("Examples (MB, block, zen_nz, lib_nz, zen_sum, lib_sum):");
        println!(
            "{:10} {:5} {:6} {:6} {:8} {:8}",
            "Location", "Block", "ZenNZ", "LibNZ", "ZenSum", "LibSum"
        );
        for (mbx, mby, block, znz, lnz, zsum, lsum) in &examples {
            println!(
                "MB({:2},{:2})  {:5} {:6} {:6} {:8} {:8}",
                mbx, mby, block, znz, lnz, zsum, lsum
            );
        }
    }

    // Mode cost calculation
    println!("\n=== Mode Cost Context Analysis ===\n");
    println!("When top=DC, left=DC:");
    println!("  DC mode cost: 40");
    println!("  VE mode cost: 1723");
    println!("  Difference: 1683 (in 1/256 bit units)");
    println!();
    println!("With lambda_i4=21:");
    println!("  DC rate advantage: 1683 * 21 = 35343 score units");
    println!("  VE needs SSE advantage of: 35343 / 256 = ~138 per block");
    println!();
    println!("For a 4x4 block (16 pixels), SSE of 138 means:");
    println!("  Average per-pixel squared error diff of ~8.6");
    println!("  Or ~3 intensity levels difference per pixel on average");
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
