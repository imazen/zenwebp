// Verify I4 vs I16 cost estimation accuracy
// Compare estimated cost vs actual encoded size for disputed blocks
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

    // Compare I16-only vs I4-enabled
    let zen_i16 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(0) // I16 only
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let zen_m4 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let lib_m4 = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    println!("File sizes:");
    println!("  zenwebp I16-only: {} bytes", zen_i16.len());
    println!("  zenwebp m4:       {} bytes", zen_m4.len());
    println!("  libwebp m4:       {} bytes", lib_m4.len());
    println!();

    // Count how many MBs are I4 vs I16
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

    let zen_i16_vp8 = extract_vp8(&zen_i16).unwrap();
    let zen_m4_vp8 = extract_vp8(&zen_m4).unwrap();
    let lib_m4_vp8 = extract_vp8(&lib_m4).unwrap();

    let (_, zen_i16_diag) = Vp8Decoder::decode_diagnostic(&zen_i16_vp8).unwrap();
    let (_, zen_m4_diag) = Vp8Decoder::decode_diagnostic(&zen_m4_vp8).unwrap();
    let (_, lib_m4_diag) = Vp8Decoder::decode_diagnostic(&lib_m4_vp8).unwrap();

    let zen_m4_i4 = zen_m4_diag
        .macroblocks
        .iter()
        .filter(|mb| mb.luma_mode == LumaMode::B)
        .count();
    let lib_m4_i4 = lib_m4_diag
        .macroblocks
        .iter()
        .filter(|mb| mb.luma_mode == LumaMode::B)
        .count();
    let total_mbs = zen_m4_diag.macroblocks.len();

    println!("I4 mode usage:");
    println!(
        "  zenwebp m4: {} / {} MBs ({:.1}%)",
        zen_m4_i4,
        total_mbs,
        100.0 * zen_m4_i4 as f64 / total_mbs as f64
    );
    println!(
        "  libwebp m4: {} / {} MBs ({:.1}%)",
        lib_m4_i4,
        total_mbs,
        100.0 * lib_m4_i4 as f64 / total_mbs as f64
    );
    println!();

    // If we use I4 more but get larger files, our I4 selection is bad
    // Expected: more I4 → smaller files (I4 should help)
    let zen_improvement = 100.0 * (1.0 - zen_m4.len() as f64 / zen_i16.len() as f64);
    println!("I4 improvement (I16→m4):");
    println!("  zenwebp: {:.1}%", zen_improvement);
    println!();

    // Effective bytes per I4 macroblock
    let zen_bytes_per_i4 = (zen_i16.len() - zen_m4.len()) as f64 / zen_m4_i4 as f64;
    let lib_bytes_per_i4 = (14522 - lib_m4.len()) as f64 / lib_m4_i4 as f64; // Use estimated I16 size
    println!("Effective savings per I4 MB:");
    println!("  zenwebp: {:.1} bytes/MB", zen_bytes_per_i4);
    println!("  libwebp: ~{:.1} bytes/MB (estimated)", lib_bytes_per_i4);
}
