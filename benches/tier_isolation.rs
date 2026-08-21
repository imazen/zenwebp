#![cfg(not(target_arch = "wasm32"))]
//! SIMD-tier isolation benchmark: the native top tier vs forced scalar.
//!
//! zenwebp carries ~125 hand-written NEON functions, but until this bench there
//! was no way to measure what any of them are worth: every existing bench
//! compares zenwebp against libwebp or against another zen crate, never zenwebp
//! against itself with SIMD disabled. That makes a NEON regression invisible —
//! a kernel can be slower than the scalar fallback and nothing reports it.
//!
//! Method: `dangerously_disable_token_process_wide` forces dispatch down to the
//! scalar fallback. The toggle happens in the `with_input` setup closure, which
//! zenbench does not time, so the two arms stay comparable under interleaving.
//!
//! Run with: `cargo bench --bench tier_isolation`
//! Do NOT pass `-C target-cpu=native`: that makes the tier compile-time
//! guaranteed and it can no longer be disabled (the bench says so and skips).

use std::path::PathBuf;
use zenbench::black_box;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

/// Enable or disable the native SIMD tier process-wide.
/// Returns false if the tier cannot be toggled (compile-time guaranteed).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
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

fn make_webp(rgb: &[u8], w: u32, h: u32, quality: f32) -> Option<Vec<u8>> {
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(4);
    zenwebp::EncodeRequest::new(&config, rgb, zenwebp::PixelLayout::Rgb8, w, h)
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
];

zenbench::main!(|suite| {
    if !set_simd(true) {
        eprintln!(
            "[tier_isolation] this target has no toggleable SIMD tier, or the \
             build pinned it at compile time (drop -C target-cpu=native). Skipping."
        );
        return;
    }
    // Confirm the tier is genuinely disableable before reporting any numbers,
    // otherwise the "scalar" arm would silently re-measure the SIMD path.
    if !set_simd(false) {
        eprintln!(
            "[tier_isolation] cannot disable {TIER_NAME} — enable \
             archmage/testable_dispatch. Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    for img in IMAGES {
        let Some(path) = corpus_path(img.subdir, img.filename) else {
            eprintln!("Skipping {}: not found", img.name);
            continue;
        };
        let Some((rgb, w, h)) = load_png_as_rgb(&path) else {
            continue;
        };
        let Some(webp_data) = make_webp(&rgb, w, h, 75.0) else {
            continue;
        };

        // --- decode ---
        let webp = webp_data.clone();
        suite.compare(format!("decode/{}", img.name), |group| {
            for (label, simd) in [(TIER_NAME, true), ("scalar", false)] {
                let data = webp.clone();
                group.bench(label, move |b| {
                    let config = zenwebp::DecodeConfig::default();
                    let bytes = data.clone();
                    b.with_input(move || {
                        // untimed: put the process into the tier under test
                        set_simd(simd);
                        bytes.clone()
                    })
                    .run(move |d| {
                        black_box(
                            zenwebp::DecodeRequest::new(&config, black_box(&d))
                                .decode_rgba()
                                .unwrap(),
                        )
                    })
                });
            }
        });

        // --- encode ---
        let src = rgb.clone();
        suite.compare(format!("encode/{}", img.name), |group| {
            for (label, simd) in [(TIER_NAME, true), ("scalar", false)] {
                let pixels = src.clone();
                group.bench(label, move |b| {
                    let px = pixels.clone();
                    b.with_input(move || {
                        set_simd(simd);
                        px.clone()
                    })
                    .run(move |p| black_box(make_webp(&p, w, h, 75.0)))
                });
            }
        });
    }

    set_simd(true);
});
