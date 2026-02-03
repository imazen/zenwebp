// Debug tool to analyze I16 vs I4 coefficient cost components
use std::io::BufReader;
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

    // Force I16 only (method 0)
    let zen_i16 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(0) // I16 only
        .sns_strength(0)
        .filter_strength(0)
        .segments(1)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    let lib_i16 = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(0) // I16 only
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    println!("I16-only comparison:");
    println!("  zenwebp: {} bytes", zen_i16.len());
    println!("  libwebp: {} bytes", lib_i16.len());
    println!(
        "  ratio: {:.3}x",
        zen_i16.len() as f64 / lib_i16.len() as f64
    );

    // Method 4 (with I4)
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

    println!("\nMethod 4 comparison:");
    println!("  zenwebp: {} bytes", zen_m4.len());
    println!("  libwebp: {} bytes", lib_m4.len());
    println!("  ratio: {:.3}x", zen_m4.len() as f64 / lib_m4.len() as f64);

    println!("\nSize savings from I4:");
    println!(
        "  zenwebp: {} -> {} ({:.1}%)",
        zen_i16.len(),
        zen_m4.len(),
        100.0 * (1.0 - zen_m4.len() as f64 / zen_i16.len() as f64)
    );
    println!(
        "  libwebp: {} -> {} ({:.1}%)",
        lib_i16.len(),
        lib_m4.len(),
        100.0 * (1.0 - lib_m4.len() as f64 / lib_i16.len() as f64)
    );
}
