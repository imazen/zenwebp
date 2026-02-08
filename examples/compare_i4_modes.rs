// Compare I4 mode choices between zenwebp and libwebp
//
// For macroblocks where both chose I4, compare the per-block mode decisions

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

    println!("=== I4 Mode Comparison ===\n");

    // Encode with both
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_segments(1);
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

    // Count mode distributions for macroblocks where BOTH chose I4
    let mut both_i4_count = 0;
    let mut zen_mode_counts = [0usize; 10];
    let mut lib_mode_counts = [0usize; 10];
    let mut mode_matches = 0usize;
    let mut _mode_mismatches = 0usize;

    let mut mismatch_examples = Vec::new();

    for (mb_idx, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        let zen_is_i4 = z.luma_mode == LumaMode::B;
        let lib_is_i4 = l.luma_mode == LumaMode::B;

        if zen_is_i4 && lib_is_i4 {
            both_i4_count += 1;

            // Compare per-block modes
            for (block_idx, (&zm, &lm)) in
                z.bpred_modes.iter().zip(l.bpred_modes.iter()).enumerate()
            {
                zen_mode_counts[zm as usize] += 1;
                lib_mode_counts[lm as usize] += 1;

                if zm == lm {
                    mode_matches += 1;
                } else {
                    _mode_mismatches += 1;
                    if mismatch_examples.len() < 20 {
                        let mbx = mb_idx % zen_diag.mb_width as usize;
                        let mby = mb_idx / zen_diag.mb_width as usize;
                        mismatch_examples.push((mbx, mby, block_idx, zm, lm));
                    }
                }
            }
        }
    }

    println!(
        "Macroblocks where both chose I4: {} ({:.1}%)",
        both_i4_count,
        100.0 * both_i4_count as f64 / zen_diag.macroblocks.len() as f64
    );
    println!();

    let total_blocks = both_i4_count * 16;
    let match_pct = 100.0 * mode_matches as f64 / total_blocks as f64;
    println!(
        "Per-block mode match rate: {}/{} ({:.1}%)",
        mode_matches, total_blocks, match_pct
    );
    println!();

    println!("Mode distribution (for shared I4 macroblocks):");
    println!("Mode    | zenwebp | libwebp | diff");
    println!("--------|---------|---------|-----");
    let mode_names = ["DC", "TM", "VE", "HE", "LD", "RD", "VR", "VL", "HD", "HU"];
    for (i, name) in mode_names.iter().enumerate() {
        let diff = zen_mode_counts[i] as i32 - lib_mode_counts[i] as i32;
        println!(
            "{:7} | {:7} | {:7} | {:+}",
            name, zen_mode_counts[i], lib_mode_counts[i], diff
        );
    }

    if !mismatch_examples.is_empty() {
        println!("\nFirst mode mismatches:");
        for (mbx, mby, block, zm, lm) in mismatch_examples.iter().take(10) {
            println!(
                "  MB({:2},{:2}) block {:2}: zen={:?}, lib={:?}",
                mbx, mby, block, zm, lm
            );
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
