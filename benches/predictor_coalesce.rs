#![cfg(not(target_arch = "wasm32"))]
//! Lossless decode benchmark across screenshot and photo corpus.
//!
//! Run with: cargo bench --bench predictor_coalesce --features _benchmarks

use std::path::PathBuf;
use zenbench::{Throughput, black_box};

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
    // 3508x2480 Wikipedia article screenshot (8.7 MP)
    TestImage {
        name: "sc_4k_wiki",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_a23d1e831e128dff.png",
    },
    // 2940x1912 iMac G3 screenshot (5.6 MP)
    TestImage {
        name: "sc_3k_imac",
        subdir: "gb82-sc",
        filename: "imac_g3.png",
    },
    // 2560x1664 codec wiki screenshot (4.3 MP)
    TestImage {
        name: "sc_2k_wiki",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    // 1920x1920 large UI screenshot (3.7 MP)
    TestImage {
        name: "sc_2k_ui",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_c8a458b0cef3d942.png",
    },
    // 1646x1062 terminal screenshot (1.7 MP)
    TestImage {
        name: "sc_1k_term",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    // 2048x2048 CLIC2025 square photo (4.2 MP)
    TestImage {
        name: "ph_2k_sq",
        subdir: "clic2025/final-test",
        filename: "ebfd571f1c6824316047a29cb5f376eec15f56dd51821119c1842be068a8b950.png",
    },
    // 2048x1536 CLIC2025 4:3 photo (3.1 MP)
    TestImage {
        name: "ph_2k_43",
        subdir: "clic2025/final-test",
        filename: "937476dd72f98638f34299dda5e9c252621b5e809c364692fc2934e000583286.png",
    },
    // 512x512 CID22 photo (0.26 MP)
    TestImage {
        name: "ph_512_cid",
        subdir: "CID22/CID22-512/validation",
        filename: "792079.png",
    },
];

zenbench::main!(lossless_decode);

fn lossless_decode(suite: &mut zenbench::Suite) {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found", img.name);
                continue;
            }
        };
        let (rgba, w, h) = match load_png_rgba(&path) {
            Some(d) => d,
            None => {
                eprintln!("Skipping {}: failed to load PNG", img.name);
                continue;
            }
        };
        let pixels = (w as u64) * (h as u64);

        let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
        let webp_data =
            zenwebp::EncodeRequest::new(&config, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
                .encode()
                .unwrap();

        eprintln!(
            "{}: {}x{} ({:.1} MP), {} bytes",
            img.name,
            w,
            h,
            pixels as f64 / 1e6,
            webp_data.len()
        );

        let label = format!("ll_{}", img.name);

        suite.compare(&label, |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");

            let data = webp_data.clone();
            group.bench("zenwebp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let config = zenwebp::DecodeConfig::default();
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
