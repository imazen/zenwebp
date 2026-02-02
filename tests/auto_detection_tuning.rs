//! Auto-detection tuning tests for CID22 and screenshot corpora.
//! Run with: cargo test --release --features _corpus_tests --test auto_detection_tuning -- --nocapture

#![cfg(feature = "_corpus_tests")]

use std::fs;
use std::io::BufReader;
use zenwebp::encoder::analysis::{analyze_image, classify_image_type_diag, ContentType};

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn encode_zen(rgb: &[u8], w: u32, h: u32, preset: zenwebp::Preset) -> usize {
    zenwebp::EncoderConfig::with_preset(preset, 75.0)
        .method(4)
        .encode_rgb(rgb, w, h)
        .unwrap()
        .len()
}

fn encode_wpx(rgb: &[u8], w: u32, h: u32, preset: webpx::Preset) -> usize {
    webpx::EncoderConfig::with_preset(preset, 75.0)
        .method(4)
        .encode_rgb(rgb, w, h, webpx::Unstoppable)
        .unwrap()
        .len()
}

/// Get classifier diagnostics for an RGB image by converting to YUV and running analysis.
fn classify_rgb(rgb: &[u8], w: u32, h: u32) -> zenwebp::ClassifierDiag {
    // Convert to YUV using the same path as the encoder
    let width = w as usize;
    let height = h as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;

    // Simple RGB→Y conversion (matching encoder's path approximately)
    let mut y_buf = vec![0u8; y_stride * mb_h * 16];
    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 3;
            let r = rgb[idx] as i32;
            let g = rgb[idx + 1] as i32;
            let b = rgb[idx + 2] as i32;
            // BT.601 luma
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_buf[row * y_stride + col] = y.clamp(0, 255) as u8;
        }
    }

    // We need U/V for analyze_image but only use Y for the classifier
    let u_buf = vec![128u8; uv_stride * mb_h * 8];
    let v_buf = vec![128u8; uv_stride * mb_h * 8];

    let (_mb_alphas, alpha_histogram) =
        analyze_image(&y_buf, &u_buf, &v_buf, width, height, y_stride, uv_stride);

    classify_image_type_diag(&y_buf, width, height, y_stride, &alpha_histogram)
}

fn ct_label(ct: ContentType) -> &'static str {
    match ct {
        ContentType::Photo => "Photo",
        ContentType::Drawing => "Draw",
        ContentType::Text => "Text",
        ContentType::Icon => "Icon",
    }
}

#[test]
fn auto_detection_cid22() {
    let corpus_dir = "/mnt/v/work/codec-corpus/CID22/CID22-512/validation";

    println!("\n=== CID22 Validation Set (41 diverse images) ===");
    println!(
        "{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>5} {:>5} {:>5} {:>5}",
        "Image",
        "Det",
        "Auto",
        "Deflt",
        "Photo",
        "Draw",
        "wpxD",
        "A/wpx",
        "loF",
        "hiF",
        "edge",
        "unif"
    );
    println!("{}", "-".repeat(120));

    let mut total_auto = 0usize;
    let mut total_default = 0usize;
    let mut total_photo = 0usize;
    let mut total_drawing = 0usize;
    let mut total_wpx_default = 0usize;
    let mut count = 0;
    let mut regressions = 0;

    let mut entries: Vec<_> = fs::read_dir(corpus_dir)
        .expect("CID22 corpus not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if let Some((rgb, w, h)) = load_png(&path.to_string_lossy()) {
            let diag = classify_rgb(&rgb, w, h);
            let auto = encode_zen(&rgb, w, h, zenwebp::Preset::Auto);
            let default = encode_zen(&rgb, w, h, zenwebp::Preset::Default);
            let photo = encode_zen(&rgb, w, h, zenwebp::Preset::Photo);
            let drawing = encode_zen(&rgb, w, h, zenwebp::Preset::Drawing);
            let wpx_default = encode_wpx(&rgb, w, h, webpx::Preset::Default);

            let auto_vs_wpx = auto as f64 / wpx_default as f64;
            let flag = if auto > default { "!!!" } else { "" };
            if auto > default {
                regressions += 1;
            }

            println!(
                "{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5.3}x {:>4.2} {:>4.2} {:>4.2} {:>4.2} {}",
                &name[..name.len().min(15)],
                ct_label(diag.content_type),
                auto, default, photo, drawing, wpx_default, auto_vs_wpx,
                diag.low_frac, diag.high_frac, diag.edge_density, diag.uniformity, flag
            );

            total_auto += auto;
            total_default += default;
            total_photo += photo;
            total_drawing += drawing;
            total_wpx_default += wpx_default;
            count += 1;
        }
    }

    println!("{}", "-".repeat(120));
    println!(
        "TOTAL ({count}): Auto={total_auto} Default={total_default} Photo={total_photo} Draw={total_drawing} wpxDef={total_wpx_default}"
    );
    println!(
        "Auto/wpx: {:.3}x  Default/wpx: {:.3}x  Auto/Default: {:.3}x  Auto/Photo: {:.3}x  Regressions: {regressions}/{count}",
        total_auto as f64 / total_wpx_default as f64,
        total_default as f64 / total_wpx_default as f64,
        total_auto as f64 / total_default as f64,
        total_auto as f64 / total_photo as f64,
    );
}

#[test]
fn auto_detection_screenshots() {
    let corpus_dir = "/mnt/v/work/codec-corpus/gb82-sc";

    println!("\n=== Screenshot Corpus (gb82-sc) ===");
    println!(
        "{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>5} {:>5} {:>5} {:>5}",
        "Image",
        "Det",
        "Auto",
        "Deflt",
        "Photo",
        "Draw",
        "Text",
        "wpxDrw",
        "A/wpx",
        "loF",
        "hiF",
        "edge",
        "unif"
    );
    println!("{}", "-".repeat(130));

    let mut total_auto = 0usize;
    let mut total_default = 0usize;
    let mut total_photo = 0usize;
    let mut total_drawing = 0usize;
    let mut total_text = 0usize;
    let mut total_wpx_drawing = 0usize;
    let mut count = 0;

    let mut entries: Vec<_> = fs::read_dir(corpus_dir)
        .expect("Screenshot corpus not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if let Some((rgb, w, h)) = load_png(&path.to_string_lossy()) {
            let diag = classify_rgb(&rgb, w, h);
            let auto = encode_zen(&rgb, w, h, zenwebp::Preset::Auto);
            let default = encode_zen(&rgb, w, h, zenwebp::Preset::Default);
            let photo = encode_zen(&rgb, w, h, zenwebp::Preset::Photo);
            let drawing = encode_zen(&rgb, w, h, zenwebp::Preset::Drawing);
            let text = encode_zen(&rgb, w, h, zenwebp::Preset::Text);
            let wpx_drawing = encode_wpx(&rgb, w, h, webpx::Preset::Drawing);

            let auto_vs_wpx = auto as f64 / wpx_drawing as f64;

            println!(
                "{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5.3}x {:>4.2} {:>4.2} {:>4.2} {:>4.2}",
                &name[..name.len().min(15)],
                ct_label(diag.content_type),
                auto, default, photo, drawing, text, wpx_drawing, auto_vs_wpx,
                diag.low_frac, diag.high_frac, diag.edge_density, diag.uniformity,
            );

            total_auto += auto;
            total_default += default;
            total_photo += photo;
            total_drawing += drawing;
            total_text += text;
            total_wpx_drawing += wpx_drawing;
            count += 1;
        }
    }

    println!("{}", "-".repeat(130));
    println!(
        "TOTAL ({count}): Auto={total_auto} Default={total_default} Photo={total_photo} Draw={total_drawing} Text={total_text} wpxDrw={total_wpx_drawing}"
    );
    println!(
        "Auto/wpx: {:.3}x  Default/wpx: {:.3}x  Auto/Default: {:.3}x",
        total_auto as f64 / total_wpx_drawing as f64,
        total_default as f64 / total_wpx_drawing as f64,
        total_auto as f64 / total_default as f64,
    );
}
