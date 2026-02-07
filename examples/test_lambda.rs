// Test to check lambda values
use webpx::Unstoppable;
use zenwebp::{EncodeRequest, PixelLayout};

fn main() {
    // Create a minimal image to trigger segment initialization
    let rgb = vec![128u8; 16 * 16 * 3];

    // These values are internal, so we can't directly access them
    // But let me print what we can access from the config

    let config = zenwebp::EncoderConfig::new_lossy()
        .quality(75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);

    // Let's print from libwebp's perspective using webpx debug
    let lib_config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1);

    println!("Q75 settings:");
    println!("  zenwebp config created (internal lambdas not exposed)");
    println!("  libwebp config created (internal lambdas not exposed)");

    // The segment quantizers should match - let's verify by comparing encoded output structure
    let zen_out = EncodeRequest::new(&config, &rgb, PixelLayout::Rgb8, 16, 16)
        .encode()
        .unwrap();
    let lib_out = lib_config.encode_rgb(&rgb, 16, 16, Unstoppable).unwrap();

    println!("\n16x16 test image at Q75, m4:");
    println!("  zenwebp: {} bytes", zen_out.len());
    println!("  libwebp: {} bytes", lib_out.len());
}
