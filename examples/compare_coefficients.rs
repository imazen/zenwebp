//! Compare quantized coefficient levels between zenwebp and libwebp.
//!
//! For blocks where both encoders chose the same mode, compare the actual
//! coefficient values to identify quantization/trellis differences.

use std::io::BufReader;
use zenwebp::decoder::vp8::Vp8Decoder;
use zenwebp::decoder::LumaMode;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    // Encode with both
    let _cfg = LossyConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_segments(1);
    let zen = EncodeRequest::lossy(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
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

    println!("=== Coefficient Level Comparison ===\n");
    println!("File sizes: zenwebp={} libwebp={}", zen.len(), lib.len());
    println!();

    // For same-mode blocks, compare coefficient levels
    let mut same_mode_i4_blocks = 0u32;
    let mut exact_match = 0u32;
    let mut diff_levels = 0u32;
    let mut zen_total_abs_level = 0i64;
    let mut lib_total_abs_level = 0i64;
    let mut zen_nz_count = 0u32;
    let mut lib_nz_count = 0u32;

    for (mb_idx, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        if z.luma_mode != LumaMode::B || l.luma_mode != LumaMode::B {
            continue;
        }

        // Both chose I4 - compare block by block
        for (block_idx, (z_block, l_block)) in z.y_blocks.iter().zip(l.y_blocks.iter()).enumerate()
        {
            // Only compare if same I4 sub-mode
            if z.bpred_modes[block_idx] != l.bpred_modes[block_idx] {
                continue;
            }
            same_mode_i4_blocks += 1;

            let mut block_match = true;
            for (pos, (&zl, &ll)) in z_block.levels.iter().zip(l_block.levels.iter()).enumerate() {
                zen_total_abs_level += zl.abs() as i64;
                lib_total_abs_level += ll.abs() as i64;
                if zl != 0 {
                    zen_nz_count += 1;
                }
                if ll != 0 {
                    lib_nz_count += 1;
                }

                if zl != ll {
                    block_match = false;
                    diff_levels += 1;
                    // Show first few differences
                    if diff_levels <= 10 {
                        let mbx = mb_idx % zen_diag.mb_width as usize;
                        let mby = mb_idx / zen_diag.mb_width as usize;
                        println!(
                            "  MB({:2},{:2}) block {:2} pos {:2}: zen={:3} lib={:3}",
                            mbx, mby, block_idx, pos, zl, ll
                        );
                    }
                }
            }
            if block_match {
                exact_match += 1;
            }
        }
    }

    println!("\nSame-mode I4 blocks analyzed: {}", same_mode_i4_blocks);
    println!(
        "Exact coefficient match: {} ({:.1}%)",
        exact_match,
        100.0 * exact_match as f64 / same_mode_i4_blocks as f64
    );
    println!("Different levels: {} positions", diff_levels);
    println!();
    println!("Total |level| sum:");
    println!("  zenwebp: {}", zen_total_abs_level);
    println!("  libwebp: {}", lib_total_abs_level);
    println!(
        "  ratio:   {:.4}x",
        zen_total_abs_level as f64 / lib_total_abs_level as f64
    );
    println!();
    println!("Non-zero coefficient count:");
    println!("  zenwebp: {}", zen_nz_count);
    println!("  libwebp: {}", lib_nz_count);
    println!(
        "  ratio:   {:.4}x",
        zen_nz_count as f64 / lib_nz_count as f64
    );
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
