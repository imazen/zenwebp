// Compare quantizer values between zenwebp and libwebp at Q75

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    use zenwebp::decoder::vp8::Vp8Decoder;
    use zenwebp::{EncoderConfig, Preset};

    println!("=== Quantizer Comparison at Q75 ===\n");

    // Encode with both
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

    println!("File sizes:");
    println!("  zenwebp: {} bytes", zen.len());
    println!("  libwebp: {} bytes", lib.len());
    println!();

    // Decode both to get segment info
    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Segments are tuples: (ydc, yac, y2dc, y2ac, uvdc, uvac)
    println!("zenwebp segment quantizers:");
    for (i, seg) in zen_diag.segments.iter().enumerate() {
        let (ydc, yac, y2dc, y2ac, uvdc, uvac) = *seg;
        println!(
            "  Segment {}: ydc={}, yac={}, y2dc={}, y2ac={}, uvdc={}, uvac={}",
            i, ydc, yac, y2dc, y2ac, uvdc, uvac
        );
        // Compute q_i4 like the encoder does
        let q_i4 = ((ydc as i32).abs() as u32 + 15 * (yac as i32).abs() as u32 + 8) >> 4;
        let lambda_mode = ((q_i4 * q_i4) >> 7).max(1);
        println!("    q_i4={}, lambda_mode={}", q_i4, lambda_mode);
    }

    println!("\nlibwebp segment quantizers:");
    for (i, seg) in lib_diag.segments.iter().enumerate() {
        let (ydc, yac, y2dc, y2ac, uvdc, uvac) = *seg;
        println!(
            "  Segment {}: ydc={}, yac={}, y2dc={}, y2ac={}, uvdc={}, uvac={}",
            i, ydc, yac, y2dc, y2ac, uvdc, uvac
        );
        // Compute q_i4 like the encoder does
        let q_i4 = ((ydc as i32).abs() as u32 + 15 * (yac as i32).abs() as u32 + 8) >> 4;
        let lambda_mode = ((q_i4 * q_i4) >> 7).max(1);
        println!("    q_i4={}, lambda_mode={}", q_i4, lambda_mode);
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
