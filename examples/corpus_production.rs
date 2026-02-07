//! Corpus test with production (default) settings
use std::env;
use std::fs;
use std::process::Command;
use zenwebp::{EncodeRequest, PixelLayout};

fn read_png_rgb(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let output = Command::new("convert")
        .args([path, "-depth", "8", "RGB:-"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let identify = Command::new("identify")
        .args(["-format", "%w %h", path])
        .output()
        .ok()?;
    let dims = String::from_utf8_lossy(&identify.stdout);
    let parts: Vec<&str> = dims.trim().split(' ').collect();
    let width: u32 = parts.first()?.parse().ok()?;
    let height: u32 = parts.get(1)?.parse().ok()?;
    Some((output.stdout, width, height))
}

fn main() {
    let dir = env::args().nth(1).unwrap_or(".".into());
    let q = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(75.0f32);

    let mut entries: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut zen_total = 0u64;
    let mut lib_total = 0u64;
    let mut count = 0;

    for entry in &entries {
        let path = entry.path();
        let path_str = path.to_string_lossy();

        if let Some((pixels, w, h)) = read_png_rgb(&path_str) {
            // Production settings (defaults)
            let _cfg = zenwebp::EncoderConfig::new_lossy()
                .with_quality(q)
                .with_method(4);
            let zen = EncodeRequest::new(&_cfg, &pixels, PixelLayout::Rgb8, w, h).encode();

            let lib = webpx::EncoderConfig::new()
                .with_quality(q)
                .with_method(4)
                .encode_rgb(&pixels, w, h, webpx::Unstoppable);

            if let (Ok(z), Ok(l)) = (zen, lib) {
                zen_total += z.len() as u64;
                lib_total += l.len() as u64;
                count += 1;
            }
        }
    }

    println!("=== Corpus Results (m4, Q{}, PRODUCTION) ===", q);
    println!("Directory: {}", dir);
    println!("Images: {}", count);
    println!("zenwebp total: {} bytes", zen_total);
    println!("libwebp total: {} bytes", lib_total);
    println!("Ratio: {:.4}x", zen_total as f64 / lib_total as f64);
}
