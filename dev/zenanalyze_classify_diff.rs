//! Compare zenwebp's homegrown alpha-histogram + Y-plane classifier
//! against zenanalyze's likelihood-based classifier on a corpus, and
//! report how many images change buckets between the two.
//!
//! Usage:
//!   zenanalyze_classify_diff <corpus_dir> [<corpus_dir>...]
//!
//! Requires `--features analyzer`.

#![cfg(feature = "analyzer")]
#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;

use zenwebp::encoder::analysis::analyze_image;
use zenwebp::encoder::analysis::{
    ImageContentType, classify_image_type, classify_image_type_rgb8_diag,
};

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32, bool)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;
    let has_alpha = matches!(
        info.color_type,
        png::ColorType::Rgba | png::ColorType::GrayscaleAlpha
    );

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
    Some((rgb, width, height, has_alpha))
}

fn rgb_to_yuv420_y_only(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    // BT.601 Y for the original classifier (matches detect_bucket()).
    let mut y = Vec::with_capacity(w * h);
    for px in rgb.chunks_exact(3) {
        let yv =
            ((u32::from(px[0]) * 76 + u32::from(px[1]) * 150 + u32::from(px[2]) * 30) >> 8) as u8;
        y.push(yv);
    }
    y
}

fn alpha_hist_from_y(y: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &v in y {
        h[v as usize] += 1;
    }
    h
}

fn main() {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("usage: zenanalyze_classify_diff <corpus_dir> [<corpus_dir>...]\n");
        std::process::exit(2);
    }
    let corpora: Vec<PathBuf> = argv.into_iter().map(PathBuf::from).collect();

    let mut total = 0u32;
    let mut same = 0u32;
    let mut changed = 0u32;
    let mut photo_to_drawing = 0u32;
    let mut drawing_to_photo = 0u32;
    let mut other_changes = 0u32;
    let mut zen_photo = 0u32;
    let mut zen_drawing = 0u32;
    let mut zen_icon = 0u32;
    let mut orig_photo = 0u32;
    let mut orig_drawing = 0u32;
    let mut orig_icon = 0u32;

    println!("path\torig_bucket\tzen_bucket\tscreen\ttext\tnatural\tflat\tdistinct\tdims");

    for dir in &corpora {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("warn: skip {}: {e}", dir.display());
                continue;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("png") {
                continue;
            }
            let (rgb, w, h, _has_alpha) = match load_png(&path.to_string_lossy()) {
                Some(t) => t,
                None => continue,
            };
            // Original classifier needs Y plane + alpha histogram.
            // detect_bucket uses Y as both: feed Y as alpha_hist source.
            let y_plane = rgb_to_yuv420_y_only(&rgb, w as usize, h as usize);
            let alpha_hist = alpha_hist_from_y(&y_plane);
            let orig =
                classify_image_type(&y_plane, w as usize, h as usize, w as usize, &alpha_hist);
            let (zen, diag) = classify_image_type_rgb8_diag(&rgb, w, h);
            total += 1;
            if orig == zen {
                same += 1;
            } else {
                changed += 1;
                match (orig, zen) {
                    (ImageContentType::Photo, ImageContentType::Drawing) => photo_to_drawing += 1,
                    (ImageContentType::Drawing, ImageContentType::Photo) => drawing_to_photo += 1,
                    _ => other_changes += 1,
                }
            }
            match orig {
                ImageContentType::Photo => orig_photo += 1,
                ImageContentType::Drawing | ImageContentType::Text => orig_drawing += 1,
                ImageContentType::Icon => orig_icon += 1,
                _ => {}
            }
            match zen {
                ImageContentType::Photo => zen_photo += 1,
                ImageContentType::Drawing | ImageContentType::Text => zen_drawing += 1,
                ImageContentType::Icon => zen_icon += 1,
                _ => {}
            }
            println!(
                "{}\t{:?}\t{:?}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}x{}",
                path.file_name().unwrap().to_string_lossy(),
                orig,
                zen,
                diag.screen_content,
                diag.text_likelihood,
                diag.natural_likelihood,
                diag.flat_color_block_ratio,
                diag.distinct_color_bins,
                w,
                h,
            );
            // `analyze_image` is unused for this comparison but keeps
            // the import warning silent if we extend later.
            let _ = analyze_image;
        }
    }

    eprintln!("=== summary ===");
    eprintln!("total={total} same={same} changed={changed}");
    eprintln!("  P->D={photo_to_drawing}  D->P={drawing_to_photo}  other={other_changes}");
    eprintln!("  orig: photo={orig_photo} drawing={orig_drawing} icon={orig_icon}");
    eprintln!("  zen:  photo={zen_photo}  drawing={zen_drawing}  icon={zen_icon}");
}
