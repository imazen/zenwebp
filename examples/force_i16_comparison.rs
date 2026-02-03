// Compare file sizes with different I4 thresholds
//
// This helps determine if our I4 mode selections are beneficial or harmful

use zenwebp::{EncoderConfig, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    println!("=== Force I16 Comparison ===\n");

    let settings = [
        ("zenwebp m0 (I16 only)", 0u8),
        ("zenwebp m1 (I16 only)", 1u8),
        ("zenwebp m2 (limited I4)", 2u8),
        ("zenwebp m4 (full I4)", 4u8),
    ];

    for (name, method) in settings {
        let webp = EncoderConfig::with_preset(Preset::Default, 75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .segments(1)
            .encode_rgb(&rgb, w, h)
            .unwrap();
        println!("{}: {} bytes", name, webp.len());
    }

    println!();

    // Now compare with libwebp
    let lib_settings = [
        ("libwebp m0", 0u8),
        ("libwebp m1", 1u8),
        ("libwebp m2", 2u8),
        ("libwebp m4", 4u8),
    ];

    for (name, method) in lib_settings {
        let webp = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1)
            .encode_rgb(&rgb, w, h, webpx::Unstoppable)
            .unwrap();
        println!("{}: {} bytes", name, webp.len());
    }

    println!("\n=== Analysis ===\n");
    println!("If m4 is smaller than m0/m1, I4 modes help compression.");
    println!("If m4 is larger than m0/m1, I4 modes hurt compression.");
    println!();
    println!("Compare zenwebp m0 vs libwebp m0 to see I16-only baseline.");
    println!("The difference at m4 vs m0 shows I4 benefit/cost.");
}
