#![cfg(not(target_arch = "wasm32"))]
//! Lossy decode benchmark: zenwebp vs libwebp (C).
//!
//! Encodes test images as lossy WebP Q75 m4, then benchmarks decoding the
//! same bytes with all three decoders. Reports throughput in Mpixels/s.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench decode_compare

use std::path::PathBuf;
use zenbench::{Throughput, black_box};

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| &c[..3])
            .copied()
            .collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    // --- Screenshots (UI, text, flat areas + sharp edges) ---
    // 3508x2480 Wikipedia article screenshot
    TestImage {
        name: "sc_4k_wiki",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_a23d1e831e128dff.png",
    },
    // 2940x1912 iMac G3 screenshot
    TestImage {
        name: "sc_3k_imac",
        subdir: "gb82-sc",
        filename: "imac_g3.png",
    },
    // 2560x1664 codec wiki screenshot
    TestImage {
        name: "sc_2k_wiki",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    // 1920x1920 large UI screenshot
    TestImage {
        name: "sc_2k_ui",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_c8a458b0cef3d942.png",
    },
    // 1646x1062 terminal screenshot
    TestImage {
        name: "sc_1k_term",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    // --- Photos (natural images, CLIC2025 professional) ---
    // 2048x2048 CLIC2025 square photo
    TestImage {
        name: "ph_2k_sq",
        subdir: "clic2025/final-test",
        filename: "ebfd571f1c6824316047a29cb5f376eec15f56dd51821119c1842be068a8b950.png",
    },
    // 2048x1536 CLIC2025 4:3 photo
    TestImage {
        name: "ph_2k_43",
        subdir: "clic2025/final-test",
        filename: "937476dd72f98638f34299dda5e9c252621b5e809c364692fc2934e000583286.png",
    },
    // 2048x1360 CLIC2025 3:2 photo
    TestImage {
        name: "ph_2k_32",
        subdir: "clic2025/final-test",
        filename: "ff32adfa29d4b5de26293352f53cc983d12d3ededd71fe65ca0ef0d887be65c4.png",
    },
    // 2048x976 CLIC2025 ultrawide photo
    TestImage {
        name: "ph_2k_uw",
        subdir: "clic2025/final-test",
        filename: "86127fbdb368eb28c3039cf61aff1c4cfdc4ade24070c8f2389968d5ead681e1.png",
    },
    // 1360x2048 CLIC2025 portrait photo
    TestImage {
        name: "ph_2k_pt",
        subdir: "clic2025/final-test",
        filename: "e0d8e29cadfc99663c7d1a4a5afe20c454ec54d0d873776ec397c59405c74790.png",
    },
    // --- Small photos (gb82 576x576, CID22 512x512) ---
    TestImage {
        name: "ph_576_baby",
        subdir: "gb82",
        filename: "baby-lossless.png",
    },
    TestImage {
        name: "ph_576_city",
        subdir: "gb82",
        filename: "city-lossless.png",
    },
    TestImage {
        name: "ph_576_flowers",
        subdir: "gb82",
        filename: "flowers-lossless.png",
    },
    TestImage {
        name: "ph_512_cid",
        subdir: "CID22/CID22-512/validation",
        filename: "792079.png",
    },
];

zenbench::main!(decode_compare);

fn decode_compare(suite: &mut zenbench::Suite) {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found", img.name);
                continue;
            }
        };
        let (rgb, w, h) = match load_png_rgb(&path) {
            Some(d) => d,
            None => {
                eprintln!("Skipping {}: failed to load PNG", img.name);
                continue;
            }
        };

        // Encode as lossy WebP Q75 m4 using zenwebp
        let config = zenwebp::EncoderConfig::new_lossy()
            .with_quality(75.0)
            .with_method(4);
        let webp_data =
            zenwebp::EncodeRequest::new(&config, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
                .encode()
                .unwrap();

        let pixels = (w as u64) * (h as u64);
        eprintln!(
            "{}: {}x{} ({:.2} MPix), WebP {} bytes",
            img.name,
            w,
            h,
            pixels as f64 / 1_000_000.0,
            webp_data.len()
        );

        suite.compare(format!("decode_{}", img.name), |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");

            let data = webp_data.clone();
            group.bench("zenwebp", move |b| {
                let d = data.clone();
                let config = zenwebp::DecodeConfig::default();
                b.with_input(move || d.clone()).run(|bytes| {
                    black_box(
                        zenwebp::DecodeRequest::new(&config, black_box(&bytes))
                            .decode_rgba()
                            .unwrap(),
                    )
                })
            });

            let data = webp_data.clone();
            group.bench("libwebp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let decoder = webp::Decoder::new(black_box(&bytes));
                    black_box(decoder.decode().unwrap())
                })
            });
        });
    }
}
