//! Compare RD cost calculations between zenwebp and libwebp for specific blocks.
//!
//! This uses decoded diagnostic info to compare how both encoders scored modes.

use std::io::BufReader;
use zenwebp::decoder::vp8::Vp8Decoder;
use zenwebp::decoder::LumaMode;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    println!("=== RD Cost Comparison ===\n");

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
        .with_method(4)
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_filter_sharpness(0)
        .with_segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    println!("File sizes:");
    println!("  zenwebp: {} bytes", zen.len());
    println!("  libwebp: {} bytes", lib.len());
    println!();

    // Decode and compare
    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Find cases where mode selection differs
    let mut same_mode = 0;
    let mut different_mode = 0;
    let mut both_i4 = 0;
    let mut both_i16 = 0;

    for (mb_idx, (z, l)) in zen_diag
        .macroblocks
        .iter()
        .zip(lib_diag.macroblocks.iter())
        .enumerate()
    {
        let mbx = mb_idx % zen_diag.mb_width as usize;
        let mby = mb_idx / zen_diag.mb_width as usize;

        let z_is_i4 = z.luma_mode == LumaMode::B;
        let l_is_i4 = l.luma_mode == LumaMode::B;

        if z_is_i4 == l_is_i4 {
            same_mode += 1;
            if z_is_i4 {
                both_i4 += 1;
            } else {
                both_i16 += 1;
            }
        } else {
            different_mode += 1;
            // Show first few differences
            if different_mode <= 5 {
                println!(
                    "MB({:2},{:2}): zen={:?}, lib={:?}",
                    mbx, mby, z.luma_mode, l.luma_mode
                );
            }
        }
    }

    println!("\nMacroblock type agreement:");
    println!(
        "  Same mode: {} ({:.1}%)",
        same_mode,
        100.0 * same_mode as f64 / zen_diag.macroblocks.len() as f64
    );
    println!(
        "  Different: {} ({:.1}%)",
        different_mode,
        100.0 * different_mode as f64 / zen_diag.macroblocks.len() as f64
    );
    println!("  Both I4:   {}", both_i4);
    println!("  Both I16:  {}", both_i16);
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
