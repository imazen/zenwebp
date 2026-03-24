#![cfg(not(target_arch = "wasm32"))]
//! Interleaved decode benchmark using zenbench.
//! Cancels out system load variation by running zenwebp and libwebp in alternating rounds.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench decode_zenbench

use std::path::PathBuf;
use zenbench::black_box;

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn load_png_as_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
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

fn make_webp(path: &std::path::Path, quality: f32) -> Option<Vec<u8>> {
    let (rgb, w, h) = load_png_as_rgb(path)?;
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(4);
    zenwebp::EncodeRequest::new(&config, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .ok()
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    TestImage {
        name: "codec_wiki_2560x1664",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    TestImage {
        name: "terminal_1646x1062",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    TestImage {
        name: "windows_2560x1392",
        subdir: "gb82-sc",
        filename: "windows.png",
    },
];

zenbench::main!(|suite| {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found", img.name);
                continue;
            }
        };

        let webp_data = match make_webp(&path, 75.0) {
            Some(d) => d,
            None => continue,
        };

        let webp = webp_data.clone();
        suite.compare(img.name, |group| {
            let data_zen = webp.clone();
            group.bench("zenwebp", move |b| {
                let config = zenwebp::DecodeConfig::default();
                let webp_bytes = data_zen.clone();
                b.with_input(move || webp_bytes.clone())
                    .run(|data| {
                        black_box(
                            zenwebp::DecodeRequest::new(&config, black_box(&data))
                                .decode_rgba()
                                .unwrap(),
                        )
                    })
            });

            let data_lib = webp.clone();
            group.bench("libwebp", move |b| {
                let webp_bytes = data_lib.clone();
                b.with_input(move || webp_bytes.clone())
                    .run(|data| {
                        let decoder = webp::Decoder::new(black_box(&data));
                        black_box(decoder.decode().unwrap())
                    })
            });
        });
    }
});
