// Test to check TDisto and mode selection details
use zenwebp::decoder::vp8::Vp8Decoder;
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

    // Encode with different methods
    for method in [4u8, 6] {
        let config = EncoderConfig::with_preset(Preset::Default, 75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .segments(1);
        let webp = EncodeRequest::new(&config, &rgb, PixelLayout::Rgb8, w, h)
            .encode()
            .unwrap();

        // Decode and count modes
        let vp8_chunk = extract_vp8(&webp).unwrap();
        let (_, diag) = Vp8Decoder::decode_diagnostic(vp8_chunk).unwrap();

        let i16_count = diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode != zenwebp::decoder::LumaMode::B)
            .count();
        let i4_count = diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode == zenwebp::decoder::LumaMode::B)
            .count();

        println!(
            "Method {}: {} bytes, {} I16, {} I4",
            method,
            webp.len(),
            i16_count,
            i4_count
        );
    }

    // Compare with libwebp
    println!("\nlibwebp:");
    for method in [4u8, 6] {
        let config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1);
        let webp = config.encode_rgb(&rgb, w, h, webpx::Unstoppable).unwrap();

        let vp8_chunk = extract_vp8(&webp).unwrap();
        let (_, diag) = Vp8Decoder::decode_diagnostic(vp8_chunk).unwrap();

        let i16_count = diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode != zenwebp::decoder::LumaMode::B)
            .count();
        let i4_count = diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode == zenwebp::decoder::LumaMode::B)
            .count();

        println!(
            "Method {}: {} bytes, {} I16, {} I4",
            method,
            webp.len(),
            i16_count,
            i4_count
        );
    }
}

fn extract_vp8(webp: &[u8]) -> Option<&[u8]> {
    if webp.len() < 12 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= webp.len() {
        let fourcc = &webp[pos..pos + 4];
        let size = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().ok()?) as usize;
        if fourcc == b"VP8 " {
            let end = (pos + 8 + size).min(webp.len());
            return Some(&webp[pos + 8..end]);
        }
        pos += 8 + size + (size & 1);
    }
    None
}
