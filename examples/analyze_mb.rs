// Analyze coefficient counts for a specific MB
use std::io::BufReader;
use zenwebp::decoder::vp8::Vp8Decoder;
use zenwebp::decoder::LumaMode;
use zenwebp::{EncoderConfig, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    // Encode with both at m4
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

    fn extract_vp8(webp: &[u8]) -> Option<Vec<u8>> {
        let mut pos = 12;
        while pos + 8 <= webp.len() {
            let fourcc = &webp[pos..pos + 4];
            let size = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().ok()?) as usize;
            if fourcc == b"VP8 " {
                return Some(webp[pos + 8..pos + 8 + size].to_vec());
            }
            pos += 8 + size + (size & 1);
        }
        None
    }

    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Analyze disputed MBs
    println!("=== Disputed MB Analysis ===\n");

    for mby in 0..zen_diag.mb_height as usize {
        for mbx in 0..zen_diag.mb_width as usize {
            let mb_idx = mby * zen_diag.mb_width as usize + mbx;
            let z = &zen_diag.macroblocks[mb_idx];
            let l = &lib_diag.macroblocks[mb_idx];

            let zen_is_i4 = z.luma_mode == LumaMode::B;
            let lib_is_i4 = l.luma_mode == LumaMode::B;

            // Only analyze where we disagree
            if zen_is_i4 == lib_is_i4 {
                continue;
            }

            let zen_nz: usize = z
                .y_blocks
                .iter()
                .flat_map(|b| b.levels.iter())
                .filter(|&&lv| lv != 0)
                .count();
            let lib_nz: usize = l
                .y_blocks
                .iter()
                .flat_map(|b| b.levels.iter())
                .filter(|&&lv| lv != 0)
                .count();

            let zen_sum: i32 = z
                .y_blocks
                .iter()
                .flat_map(|b| b.levels.iter())
                .map(|&lv| lv.abs() as i32)
                .sum();
            let lib_sum: i32 = l
                .y_blocks
                .iter()
                .flat_map(|b| b.levels.iter())
                .map(|&lv| lv.abs() as i32)
                .sum();

            println!(
                "MB({:2},{:2}): zen={:?} lib={:?}",
                mbx, mby, z.luma_mode, l.luma_mode
            );
            println!(
                "  Y nonzero: zen={:2} lib={:2} (diff={:+})",
                zen_nz,
                lib_nz,
                zen_nz as i32 - lib_nz as i32
            );
            println!(
                "  |level| sum: zen={:3} lib={:3} (diff={:+})",
                zen_sum,
                lib_sum,
                zen_sum - lib_sum
            );
            if zen_is_i4 && !lib_is_i4 {
                println!("  zen I4 modes: {:?}", z.bpred_modes);
                println!("  lib I16 mode: {:?}", l.luma_mode);
            }
            println!();
        }
    }
}
