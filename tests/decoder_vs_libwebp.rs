//! Decoder-vs-libwebp divergence check for the transparent rose.
//!
//! Imageflow's `test_transparent_webp_to_webp` shows ~47.8% of pixels
//! diverge between the c-codecs build (libwebp decode + encode) and the
//! zen-codecs build (zenwebp decode + encode). The lossless encoder has
//! been verified byte-exact against libwebp's encoder. This test isolates
//! the DECODER by running the same source bytes through both decoders and
//! comparing the RGBA output.
//!
//! Source image: `tests/images/gallery2/1_webp_ll.webp` (400x301 lossless
//! transparent rose on black). The c-codecs pipeline uses libwebp
//! (wrapped by `webpx`) to decode; the zen-codecs pipeline uses zenwebp's
//! own decoder.
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::collections::HashMap;

const ROSE_PATH: &str = "tests/images/gallery2/1_webp_ll.webp";

fn classify(a: u8) -> &'static str {
    match a {
        0 => "a=0",
        255 => "a=255",
        _ => "0<a<255",
    }
}

fn histogram_per_channel(zen: &[u8], lib: &[u8]) {
    assert_eq!(zen.len(), lib.len());
    let mut per_channel = [[0u64; 256]; 4];
    let mut per_class: HashMap<&'static str, (u64, [u32; 4])> = HashMap::new();
    for (z, l) in zen.chunks_exact(4).zip(lib.chunks_exact(4)) {
        let class = classify(z[3]); // use zen alpha for classification
        let entry = per_class.entry(class).or_insert((0, [0u32; 4]));
        entry.0 += 1;
        for c in 0..4 {
            let d = (z[c] as i16 - l[c] as i16).unsigned_abs() as usize;
            per_channel[c][d] += 1;
            entry.1[c] = entry.1[c].max(d as u32);
        }
    }
    eprintln!("\n=== per-channel delta histogram (top 16 bins) ===");
    for (c, name) in ["R", "G", "B", "A"].iter().enumerate() {
        let mut bins: Vec<(usize, u64)> = per_channel[c]
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v > 0)
            .collect();
        bins.sort_by(|a, b| b.0.cmp(&a.0));
        eprintln!(
            "  {}: {:?}",
            name,
            bins.iter().take(16).collect::<Vec<_>>()
        );
    }
    eprintln!("=== per-pixel-class max delta [R, G, B, A] ===");
    for (class, (count, maxd)) in &per_class {
        eprintln!("  {} ({} pixels): maxΔ = {:?}", class, count, maxd);
    }
}

fn dump_first_divergent_pixels(zen: &[u8], lib: &[u8], w: u32, h: u32, n: usize) {
    eprintln!("\n=== first {n} divergent pixels ===");
    let mut found = 0usize;
    for i in 0..(w * h) as usize {
        let z = &zen[i * 4..i * 4 + 4];
        let l = &lib[i * 4..i * 4 + 4];
        if z != l {
            let x = (i as u32) % w;
            let y = (i as u32) / w;
            eprintln!("  px ({x},{y}): zen={:?} lib={:?}", z, l);
            found += 1;
            if found >= n {
                break;
            }
        }
    }
}

#[test]
fn zenwebp_decoder_matches_libwebp_on_rose_rgba() {
    let data = match std::fs::read(ROSE_PATH) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("rose image not found at {ROSE_PATH}: {e} — skipping");
            return;
        }
    };

    // zenwebp's oneshot RGBA decode — this is what `WebpDecoderConfig::new().decode(...)`
    // collapses to for a single-frame RGBA output.
    let (zen_px, zw, zh) = zenwebp::oneshot::decode_rgba(&data).expect("zenwebp decode");
    // webpx wraps libwebp's WebPDecode().
    let (lib_px, lw, lh) = webpx::decode_rgba(&data).expect("libwebp decode");

    assert_eq!((zw, zh), (lw, lh), "dimension mismatch");
    assert_eq!(zen_px.len(), lib_px.len(), "buffer length mismatch");

    if zen_px == lib_px {
        eprintln!(
            "PASS: zenwebp and libwebp produce byte-exact RGBA for {}x{} rose",
            zw, zh
        );
        return;
    }

    // Divergence — report evidence, then fail.
    let total = (zw * zh) as usize;
    let diff_px = zen_px
        .chunks_exact(4)
        .zip(lib_px.chunks_exact(4))
        .filter(|(z, l)| z != l)
        .count();
    let mut max_delta = [0u16; 4];
    for (z, l) in zen_px.chunks_exact(4).zip(lib_px.chunks_exact(4)) {
        for c in 0..4 {
            max_delta[c] = max_delta[c].max((z[c] as i16 - l[c] as i16).unsigned_abs());
        }
    }
    eprintln!(
        "DIVERGE: {diff_px}/{total} ({:.1}%) pixels differ; max Δ = [R={} G={} B={} A={}]",
        100.0 * diff_px as f64 / total as f64,
        max_delta[0],
        max_delta[1],
        max_delta[2],
        max_delta[3]
    );
    histogram_per_channel(&zen_px, &lib_px);
    dump_first_divergent_pixels(&zen_px, &lib_px, zw, zh, 10);

    panic!("zenwebp decoder diverges from libwebp on lossless rose");
}

/// Reproduce imageflow's exact decode path: zencodec dyn-dispatch with
/// `preferred=[BGRA8_SRGB, RGBA8_SRGB, ...]`, then compare the BGRA output
/// (after swizzling to RGBA for comparison) against libwebp's RGBA decode.
#[test]
fn zenwebp_bgra_dyn_dispatch_matches_libwebp_on_rose() {
    use zencodec::decode::DynDecoderConfig;
    use zenpixels::PixelDescriptor;

    let data = match std::fs::read(ROSE_PATH) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("rose image not found at {ROSE_PATH}: {e} — skipping");
            return;
        }
    };

    // Replicate imageflow's preferred-format list exactly.
    let preferred = [
        PixelDescriptor::BGRA8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::GRAY8_SRGB,
    ];

    let config: Box<dyn DynDecoderConfig> =
        Box::new(zenwebp::zencodec::WebpDecoderConfig::new());
    let job = config.dyn_job();
    let decoder = job
        .into_decoder(std::borrow::Cow::Borrowed(&data), &preferred)
        .expect("zenwebp dyn decoder");
    let output = decoder.decode().expect("zenwebp decode");

    // Resolve the decoded frame into a buffer + descriptor.
    let px = output.pixels();
    let descriptor = px.descriptor();
    let (zw, zh) = (px.width(), px.rows());
    let zen_bytes: Vec<u8> = px.contiguous_bytes().into_owned();
    eprintln!(
        "zen dyn dispatch delivered {:?} at {}x{}",
        descriptor, zw, zh
    );

    // libwebp via webpx always delivers RGBA.
    let (lib_rgba, lw, lh) = webpx::decode_rgba(&data).expect("libwebp decode");
    assert_eq!((zw, zh), (lw, lh), "dimension mismatch");

    // Convert zen output to RGBA for comparison.
    let zen_rgba: Vec<u8> = if descriptor == PixelDescriptor::BGRA8_SRGB {
        zen_bytes
            .chunks_exact(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect()
    } else if descriptor == PixelDescriptor::RGBA8_SRGB {
        zen_bytes.clone()
    } else {
        panic!("unexpected descriptor {:?}", descriptor)
    };

    assert_eq!(zen_rgba.len(), lib_rgba.len());
    if zen_rgba == lib_rgba {
        eprintln!("PASS: dyn-dispatch BGRA→RGBA matches libwebp for rose");
        return;
    }

    let total = (zw * zh) as usize;
    let diff_px = zen_rgba
        .chunks_exact(4)
        .zip(lib_rgba.chunks_exact(4))
        .filter(|(z, l)| z != l)
        .count();
    eprintln!(
        "DIVERGE (dyn dispatch): {diff_px}/{total} ({:.1}%) pixels differ",
        100.0 * diff_px as f64 / total as f64
    );
    histogram_per_channel(&zen_rgba, &lib_rgba);
    dump_first_divergent_pixels(&zen_rgba, &lib_rgba, zw, zh, 10);
    panic!("zenwebp dyn-dispatch decode diverges from libwebp on lossless rose");
}
