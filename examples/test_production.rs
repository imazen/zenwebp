//! Compare production (default) settings between zenwebp and libwebp
use zenwebp::{ColorType, EncodeRequest};
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/CID22/original/792079.png".into());
    let output = std::process::Command::new("convert")
        .args([&path, "-depth", "8", "RGB:-"])
        .output()
        .unwrap();
    let pixels = output.stdout;
    let identify = std::process::Command::new("identify")
        .args(["-format", "%w %h", &path])
        .output()
        .unwrap();
    let dims = String::from_utf8_lossy(&identify.stdout);
    let parts: Vec<&str> = dims.trim().split(' ').collect();
    let (w, h): (u32, u32) = (parts[0].parse().unwrap(), parts[1].parse().unwrap());

    println!("Image: {} ({}x{})\n", path, w, h);
    println!("Default presets (SNS=50, filter=60, segments=4):\n");
    println!("  Q    zenwebp   libwebp    ratio");
    println!("----  --------  --------  -------");

    for q in [50, 75, 90] {
        let _cfg = zenwebp::EncoderConfig::new().quality(q as f32).method(4);
        let zen = EncodeRequest::new(&_cfg, &pixels, ColorType::Rgb8, w, h)
            .encode()
            .unwrap();

        let lib = webpx::EncoderConfig::new()
            .quality(q as f32)
            .method(4)
            .encode_rgb(&pixels, w, h, webpx::Unstoppable)
            .unwrap();

        println!(
            " {:3}    {:6}    {:6}   {:.3}x",
            q,
            zen.len(),
            lib.len(),
            zen.len() as f64 / lib.len() as f64
        );
    }
}
