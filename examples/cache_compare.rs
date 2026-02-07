//! Compare file sizes with auto vs forced cache=0

use std::fs;
use std::path::Path;

fn wrap_vp8l_in_riff(vp8l_data: &[u8]) -> Vec<u8> {
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + vp8l_data.len() + (vp8l_data.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(vp8l_data.len() as u32).to_le_bytes());
    webp.extend_from_slice(vp8l_data);
    if !vp8l_data.len().is_multiple_of(2) {
        webp.push(0);
    }
    webp
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-corpus/CID22/CID22-512/training".to_string());
    let max = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50usize);

    let mut pngs: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? == "png" {
                Some(p.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    pngs.sort();
    pngs.truncate(max);

    println!(
        "{:<25} {:>8} {:>8} {:>6}",
        "Image", "auto", "cache=0", "delta"
    );
    println!("{:-<55}", "");

    let mut total_auto = 0u64;
    let mut total_no_cache = 0u64;
    let mut auto_wins = 0;
    let mut no_cache_wins = 0;
    let mut ties = 0;

    for png_path in &pngs {
        let name = Path::new(png_path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let file = match fs::File::open(png_path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = match decoder.read_info() {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let (rgb_pixels, has_alpha) = match info.color_type {
            png::PixelLayout::Rgba => (buf[..info.buffer_size()].to_vec(), true),
            png::PixelLayout::Rgb => (buf[..info.buffer_size()].to_vec(), false),
            _ => continue,
        };
        let (w, h) = (info.width, info.height);

        // Auto cache
        let config_auto = zenwebp::encoder::vp8l::Vp8lConfig {
            quality: zenwebp::encoder::vp8l::Vp8lQuality {
                quality: 75,
                method: 4,
            },
            use_predictor: true,
            use_cross_color: true,
            use_subtract_green: true,
            use_palette: true,
            ..Default::default()
        };
        let auto_size =
            match zenwebp::encoder::vp8l::encode_vp8l(&rgb_pixels, w, h, has_alpha, &config_auto, &enough::Unstoppable) {
                Ok(d) => wrap_vp8l_in_riff(&d).len() as u64,
                Err(_) => continue,
            };

        // No cache
        let config_no = zenwebp::encoder::vp8l::Vp8lConfig {
            cache_bits: Some(0),
            ..config_auto.clone()
        };
        let no_cache_size =
            match zenwebp::encoder::vp8l::encode_vp8l(&rgb_pixels, w, h, has_alpha, &config_no, &enough::Unstoppable) {
                Ok(d) => wrap_vp8l_in_riff(&d).len() as u64,
                Err(_) => continue,
            };

        let delta = auto_size as i64 - no_cache_size as i64;
        let delta_pct = delta as f64 / no_cache_size as f64 * 100.0;
        println!(
            "{:<25} {:>8} {:>8} {:>+5.1}%",
            &name[..name.len().min(25)],
            auto_size,
            no_cache_size,
            delta_pct,
        );

        total_auto += auto_size;
        total_no_cache += no_cache_size;
        if auto_size < no_cache_size {
            auto_wins += 1;
        } else if no_cache_size < auto_size {
            no_cache_wins += 1;
        } else {
            ties += 1;
        }
    }

    println!("{:-<55}", "");
    println!(
        "{:<25} {:>8} {:>8} {:>+5.1}%",
        "TOTAL",
        total_auto,
        total_no_cache,
        (total_auto as f64 - total_no_cache as f64) / total_no_cache as f64 * 100.0,
    );
    println!(
        "auto better: {}, cache=0 better: {}, tie: {}",
        auto_wins, no_cache_wins, ties
    );
}
