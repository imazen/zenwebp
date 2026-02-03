// Compare I16 mode distribution between zenwebp and libwebp
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

    // Method 0 (I16 only)
    let zen = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(0)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let lib = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(0)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    // Extract VP8 and decode
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

    let zen_vp8 = extract_vp8(&zen).unwrap();
    let lib_vp8 = extract_vp8(&lib).unwrap();

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(&zen_vp8).unwrap();
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(&lib_vp8).unwrap();

    // Count modes
    let mut zen_counts = [0usize; 4]; // DC, V, H, TM
    let mut lib_counts = [0usize; 4];

    for mb in &zen_diag.macroblocks {
        let idx = match mb.luma_mode {
            LumaMode::DC => 0,
            LumaMode::V => 1,
            LumaMode::H => 2,
            LumaMode::TM => 3,
            LumaMode::B => panic!("I4 in I16-only mode"),
        };
        zen_counts[idx] += 1;
    }

    for mb in &lib_diag.macroblocks {
        let idx = match mb.luma_mode {
            LumaMode::DC => 0,
            LumaMode::V => 1,
            LumaMode::H => 2,
            LumaMode::TM => 3,
            LumaMode::B => panic!("I4 in I16-only mode"),
        };
        lib_counts[idx] += 1;
    }

    println!("I16 mode distribution (method 0):");
    println!("Mode    zenwebp   libwebp");
    println!("DC      {:5}     {:5}", zen_counts[0], lib_counts[0]);
    println!("V       {:5}     {:5}", zen_counts[1], lib_counts[1]);
    println!("H       {:5}     {:5}", zen_counts[2], lib_counts[2]);
    println!("TM      {:5}     {:5}", zen_counts[3], lib_counts[3]);

    // Count mode agreement
    let mut same = 0;
    let mut diff = 0;
    for (z, l) in zen_diag.macroblocks.iter().zip(lib_diag.macroblocks.iter()) {
        if z.luma_mode == l.luma_mode {
            same += 1;
        } else {
            diff += 1;
        }
    }
    println!(
        "\nMode agreement: {} same, {} different ({:.1}% match)",
        same,
        diff,
        100.0 * same as f64 / (same + diff) as f64
    );

    println!("\nFile sizes:");
    println!("  zenwebp: {} bytes", zen.len());
    println!("  libwebp: {} bytes", lib.len());
}
