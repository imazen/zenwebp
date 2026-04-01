//! Lossless decode profiling across a corpus of images.
//!
//! Pre-encodes all images, then decodes each N times. When run under callgrind
//! with --toggle-collect='*decode_rgba*', only the decode loop is measured.
//!
//! Usage: target/release/examples/callgrind_lossless [iterations]
//! Default: 3 iterations per image.

use std::path::PathBuf;

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgba(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        _ => return None,
    };
    Some((rgba, info.width, info.height))
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    // --- Screenshots (gb82-sc) ---
    TestImage {
        name: "sc_codec_wiki",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    TestImage {
        name: "sc_gmessages",
        subdir: "gb82-sc",
        filename: "gmessages.png",
    },
    TestImage {
        name: "sc_graph",
        subdir: "gb82-sc",
        filename: "graph.png",
    },
    TestImage {
        name: "sc_gui",
        subdir: "gb82-sc",
        filename: "gui.png",
    },
    TestImage {
        name: "sc_imac_dark",
        subdir: "gb82-sc",
        filename: "imac_dark.png",
    },
    TestImage {
        name: "sc_imac_g3",
        subdir: "gb82-sc",
        filename: "imac_g3.png",
    },
    TestImage {
        name: "sc_imessage",
        subdir: "gb82-sc",
        filename: "imessage.png",
    },
    TestImage {
        name: "sc_terminal",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    TestImage {
        name: "sc_windows95",
        subdir: "gb82-sc",
        filename: "windows95.png",
    },
    TestImage {
        name: "sc_windows",
        subdir: "gb82-sc",
        filename: "windows.png",
    },
    // --- Screenshots (png-conformance) ---
    TestImage {
        name: "sc_4k_wiki",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_a23d1e831e128dff.png",
    },
    TestImage {
        name: "sc_ui_1920",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_c8a458b0cef3d942.png",
    },
    // --- Photos (CLIC2025, first 5) ---
    TestImage {
        name: "ph_clic_01",
        subdir: "clic2025/final-test",
        filename: "ebfd571f1c6824316047a29cb5f376eec15f56dd51821119c1842be068a8b950.png",
    },
    TestImage {
        name: "ph_clic_02",
        subdir: "clic2025/final-test",
        filename: "937476dd72f98638f34299dda5e9c252621b5e809c364692fc2934e000583286.png",
    },
    TestImage {
        name: "ph_clic_03",
        subdir: "clic2025/final-test",
        filename: "ff32adfa29d4b5de26293352f53cc983d12d3ededd71fe65ca0ef0d887be65c4.png",
    },
    TestImage {
        name: "ph_clic_04",
        subdir: "clic2025/final-test",
        filename: "86127fbdb368eb28c3039cf61aff1c4cfdc4ade24070c8f2389968d5ead681e1.png",
    },
    TestImage {
        name: "ph_clic_05",
        subdir: "clic2025/final-test",
        filename: "e0d8e29cadfc99663c7d1a4a5afe20c454ec54d0d873776ec397c59405c74790.png",
    },
];

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);

    // Phase 1: Pre-encode all images
    let mut encoded: Vec<(&str, Vec<u8>)> = Vec::new();
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("skip {}: not found", img.name);
                continue;
            }
        };
        let (rgba, w, h) = match load_png_rgba(&path) {
            Some(d) => d,
            None => {
                eprintln!("skip {}: load failed", img.name);
                continue;
            }
        };
        let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
        let webp = zenwebp::EncodeRequest::new(&config, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
            .encode()
            .unwrap();
        let pixels = w as u64 * h as u64;
        eprintln!(
            "{}: {}x{} ({:.1}MP) → {} bytes",
            img.name,
            w,
            h,
            pixels as f64 / 1e6,
            webp.len()
        );
        encoded.push((img.name, webp));
    }

    eprintln!(
        "\ndecoding {} images × {} iterations",
        encoded.len(),
        iterations
    );

    // Phase 2: Decode loop (this is what callgrind measures)
    let decode_config = zenwebp::DecodeConfig::default();
    for (name, webp) in &encoded {
        for _ in 0..iterations {
            let result = zenwebp::DecodeRequest::new(&decode_config, webp)
                .decode_rgba()
                .unwrap();
            std::hint::black_box(&result);
        }
        eprintln!("  {} done", name);
    }
}
