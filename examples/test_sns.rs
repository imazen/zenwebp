// Test with SNS enabled to verify spectral distortion works
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

    // With SNS=50 (default)
    let zen_m4_sns = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(50)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let lib_m4_sns = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(50)
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

    let zen_vp8 = extract_vp8(&zen_m4_sns).unwrap();
    let lib_vp8 = extract_vp8(&lib_m4_sns).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    let zen_i4 = zen_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let lib_i4 = lib_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();

    println!("=== With SNS=50 (tlambda enabled) ===");
    println!(
        "  zenwebp m4: {} bytes, I4: {} MBs ({:.1}%)",
        zen_m4_sns.len(),
        zen_i4,
        100.0 * zen_i4 as f64 / zen_diag.macroblocks.len() as f64
    );
    println!(
        "  libwebp m4: {} bytes, I4: {} MBs ({:.1}%)",
        lib_m4_sns.len(),
        lib_i4,
        100.0 * lib_i4 as f64 / lib_diag.macroblocks.len() as f64
    );

    // Also test without SNS for comparison
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

    let zen_vp8_nosns = extract_vp8(&zen_m4).unwrap();
    let lib_vp8_nosns = extract_vp8(&lib_m4).unwrap();

    let (_, zen_diag_nosns) = Vp8Decoder::decode_diagnostic(&zen_vp8_nosns).unwrap();
    let (_, lib_diag_nosns) = Vp8Decoder::decode_diagnostic(&lib_vp8_nosns).unwrap();

    let zen_i4_nosns = zen_diag_nosns
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let lib_i4_nosns = lib_diag_nosns
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();

    println!("\n=== With SNS=0 (tlambda=0) ===");
    println!(
        "  zenwebp m4: {} bytes, I4: {} MBs ({:.1}%)",
        zen_m4.len(),
        zen_i4_nosns,
        100.0 * zen_i4_nosns as f64 / zen_diag_nosns.macroblocks.len() as f64
    );
    println!(
        "  libwebp m4: {} bytes, I4: {} MBs ({:.1}%)",
        lib_m4.len(),
        lib_i4_nosns,
        100.0 * lib_i4_nosns as f64 / lib_diag_nosns.macroblocks.len() as f64
    );
}
