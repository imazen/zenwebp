// Compare I16-only encoding between zenwebp and libwebp
use std::io::BufReader;
use zenwebp::{ColorType, EncodeRequest, EncoderConfig, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    // Method 0 forces I16-only
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(0)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);
    let zen_m0 = EncodeRequest::new(&_cfg, &rgb, ColorType::Rgb8, w, h)
        .encode()
        .unwrap();

    let lib_m0 = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(0)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    // Also method 1
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(1)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);
    let zen_m1 = EncodeRequest::new(&_cfg, &rgb, ColorType::Rgb8, w, h)
        .encode()
        .unwrap();

    let lib_m1 = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(1)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();

    println!("I16-only comparison (SNS=0, filter=0, segments=1):");
    println!(
        "  m0: zenwebp={:6} lib={:6} ratio={:.3}x",
        zen_m0.len(),
        lib_m0.len(),
        zen_m0.len() as f64 / lib_m0.len() as f64
    );
    println!(
        "  m1: zenwebp={:6} lib={:6} ratio={:.3}x",
        zen_m1.len(),
        lib_m1.len(),
        zen_m1.len() as f64 / lib_m1.len() as f64
    );
}
