//! Baseline encoder for issue #32: tiny low-color images at m0 produce 4x libwebp size.
//!
//! Usage:
//!   cargo run --release --features _profiling --example issue_32_baseline -- <png_path>

use png::Decoder;
use std::env;
use std::fs::File;
use std::io::BufReader;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, Preset};

fn decode_png_rgb(path: &str) -> (u32, u32, Vec<u8>) {
    let decoder = Decoder::new(BufReader::new(File::open(path).expect("open")));
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("next frame");
    let (w, h) = (info.width, info.height);
    let bytes = &buf[..info.buffer_size()];
    // Convert to RGB8.
    let rgb = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => bytes
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => bytes
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        other => panic!("unsupported color type {:?}", other),
    };
    (w, h, rgb)
}

fn encode_libwebp_photo(rgb: &[u8], w: u32, h: u32, q: f32) -> Vec<u8> {
    use libwebp_sys as ll;
    unsafe {
        let mut config: ll::WebPConfig = core::mem::zeroed();
        // Photo preset, mirroring libwebp cwebp -preset photo
        let ok = ll::WebPConfigInitInternal(
            &mut config,
            ll::WebPPreset::WEBP_PRESET_PHOTO,
            q,
            ll::WEBP_ENCODER_ABI_VERSION as i32,
        );
        assert_eq!(ok, 1, "WebPConfigInitInternal");
        config.method = 0;
        let mut pic: ll::WebPPicture = core::mem::zeroed();
        ll::WebPPictureInitInternal(&mut pic, ll::WEBP_ENCODER_ABI_VERSION as i32);
        pic.width = w as i32;
        pic.height = h as i32;
        pic.use_argb = 0;
        ll::WebPPictureAlloc(&mut pic);
        // Import RGB
        let stride = (w * 3) as i32;
        ll::WebPPictureImportRGB(&mut pic, rgb.as_ptr(), stride);

        let mut writer: ll::WebPMemoryWriter = core::mem::zeroed();
        ll::WebPMemoryWriterInit(&mut writer);
        pic.writer = Some(ll::WebPMemoryWrite);
        pic.custom_ptr = &mut writer as *mut _ as *mut _;
        let success = ll::WebPEncode(&config, &mut pic);
        assert_eq!(success, 1, "WebPEncode failed");
        let out = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
        ll::WebPPictureFree(&mut pic);
        ll::WebPMemoryWriterClear(&mut writer);
        out
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let png_path = args
        .get(1)
        .expect("PNG path required (e.g. d88de4b9e6efe211.png)");

    let (w, h, pixels) = decode_png_rgb(png_path);
    println!("Image: {} ({}x{}, {} pixels)", png_path, w, h, w * h);

    println!();
    println!("{:>8} {:>3} {:>2} | {:>6} | {:>6} | {:>6}",
             "preset", "q", "m", "zen", "lib", "ratio");
    for &(preset, name) in &[
        (Preset::Photo, "Photo"),
        (Preset::Default, "Default"),
        (Preset::Drawing, "Drawing"),
    ] {
        for &q in &[75.0_f32, 90.0, 95.0] {
            for &m in &[0u8, 4] {
                let cfg = LossyConfig::with_preset(preset, q).with_method(m);
                let zen = EncodeRequest::lossy(&cfg, &pixels, PixelLayout::Rgb8, w, h)
                    .encode()
                    .expect("encode");
                // Only compare libwebp at Photo preset (the failing case).
                let lib = if matches!(preset, Preset::Photo) && m == 0 {
                    encode_libwebp_photo(&pixels, w, h, q)
                } else {
                    Vec::new()
                };
                let lib_len = lib.len();
                let ratio = if lib_len > 0 {
                    zen.len() as f64 / lib_len as f64
                } else {
                    0.0
                };
                println!(
                    "{:>8} {:>3} {:>2} | {:>6} | {:>6} | {:>6.3}",
                    name, q as i32, m, zen.len(), lib_len, ratio
                );
                // Dump worst case for analysis.
                if matches!(preset, Preset::Photo) && q == 95.0 && m == 0 {
                    std::fs::write(format!("/tmp/issue32_zen_q{}_m{}.webp", q as i32, m), &zen)
                        .ok();
                    std::fs::write(format!("/tmp/issue32_lib_q{}_m{}.webp", q as i32, m), &lib)
                        .ok();
                }
            }
        }
    }
}
