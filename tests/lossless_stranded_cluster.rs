//! Regression gate for the VP8L stranded-cluster corruption (issue #72).
//!
//! Phase 5 of meta-Huffman clustering used to allocate one final histogram per
//! *active* cluster, including clusters the remap stranded with zero mapped
//! tiles. The encoder then wrote Huffman trees for an unreferenced trailing
//! cluster — but decoders size the group list from the entropy image's max
//! symbol (max+1), so the extra trees get parsed as pixel data and the rest of
//! the bitstream shifts. The result is a **valid-but-wrong** stream: decoders
//! accept it and return garbage pixels instead of erroring, so nothing upstream
//! signals a problem. `build_final_histograms` now compacts to referenced
//! clusters with dense renumbering.
//!
//! Shipped broken in 0.4.4 at DEFAULT settings (84% of pixels wrong on one
//! 512x320 rendition; `BitStreamError` on others). Nothing gated the trigger:
//! `lossless_fast_tier.rs` only covers method(0), `lossless_roundtrip.rs` pins
//! quality 75, and `examples/sweep_validate.rs` runs 3 CID22 images that never
//! saw this content. The trigger is a content x clustering lottery, not a
//! dimension or a dial — so this gate sweeps m4/m5/m6 across the whole quality
//! axis on content built to produce many distinct tile histograms (which is
//! what gives clustering+remap the chance to strand one).
//!
//! Both halves matter:
//!   * roundtrip vs the ORIGINAL pixels catches the silent corruption (zenwebp's
//!     own decoder reproduces the same garbage, so self-consistency proves
//!     nothing — only comparing against the source does);
//!   * the libwebp cross-decode proves we emit a stream the reference decoder
//!     reads identically, not merely one we agree with ourselves about.
//!
//! **Verified to catch the bug, not merely to pass** (2026-07-15, by
//! reintroducing the defect — restoring one final histogram per *active*
//! cluster). Against the broken encoder these fail with `BitStreamError`:
//! `stranded_cluster_pinned_repro_511x320_m4_q25`,
//! `m4_quality_axis_stranding_content` / `m5_…` / `m6_…`, plus the two unit
//! tests in `meta_huffman.rs` (`final_histograms_skip_stranded_cluster`,
//! `final_histograms_preserve_tile_counts_across_compaction`).
//!
//! These, by contrast, still PASS against the broken encoder —
//! `photo_grain_coarse_axis`, `default_quality_m6_multi_region`, and
//! `odd_dimensions_layout_lottery`. They are general nets, not #72 gates: their
//! content does not strand a cluster. That asymmetry is the whole lesson — most
//! content never strands, so a green sweep proves little, which is why the
//! verified trigger is pinned explicitly above. If you widen this file, re-run
//! the revert-and-confirm-failure check rather than trusting a passing sweep.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
}

/// Content with many statistically distinct regions. Distinct regions produce
/// distinct tile histograms, which is what gives entropy-bin + stochastic +
/// greedy clustering (and the remap that follows) the chance to leave a cluster
/// with zero mapped tiles.
fn multi_region(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        for x in 0..w {
            let region = ((x * 5 / w.max(1)) + (y * 5 / h.max(1)) * 5) % 7;
            let (r, g, b, a) = match region {
                0 => (255, 0, 0, 255),
                1 => (rng.byte(), rng.byte(), rng.byte(), 255),
                2 => ((x % 256) as u8, (y % 256) as u8, 128, 255),
                3 => (0, 0, 0, 255),
                4 => (((x ^ y) % 256) as u8, 64, 200, 255),
                5 => (200, 200, 200, rng.byte() | 0x80),
                _ => (
                    (x.wrapping_mul(7) % 256) as u8,
                    (y.wrapping_mul(13) % 256) as u8,
                    rng.byte(),
                    255,
                ),
            };
            px.extend_from_slice(&[r, g, b, a]);
        }
    }
    px
}

/// Smooth-plus-grain content, closest in shape to the photographic renditions
/// that first exposed this (`o_8148.scale512x320` et al).
fn photo_grain(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        for x in 0..w {
            let base = ((x as f32 / w as f32) * 180.0 + (y as f32 / h as f32) * 60.0) as i32;
            let n = i32::from(rng.byte() % 24) - 12;
            let v = (base + n).clamp(0, 255) as u8;
            px.extend_from_slice(&[v, v.wrapping_add(17), v.wrapping_sub(23), 255]);
        }
    }
    px
}

/// Encode lossless and assert the stream decodes back to the EXACT source
/// pixels, through both zenwebp and libwebp.
fn assert_lossless_exact(name: &str, rgba: &[u8], w: u32, h: u32, method: u8, quality: u32) {
    let cfg = EncoderConfig::Lossless(
        zenwebp::LosslessConfig::new()
            .with_quality(quality as f32)
            .with_method(method)
            // `exact` keeps fully-transparent pixels' RGB, so the comparison
            // below is a true bit-for-bit check over all four channels.
            .with_exact(true),
    );
    let webp = EncodeRequest::new(&cfg, rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("{name} m{method} q{quality}: encode failed: {e:?}"));

    let (zen, zw, zh) = zenwebp::oneshot::decode_rgba(&webp)
        .unwrap_or_else(|e| panic!("{name} m{method} q{quality}: zenwebp decode failed: {e:?}"));
    assert_eq!((zw, zh), (w, h), "{name} m{method} q{quality}: dimensions");
    if zen != rgba {
        let bad = zen.iter().zip(rgba).filter(|(a, b)| a != b).count();
        panic!(
            "{name} m{method} q{quality}: lossless roundtrip is NOT lossless — \
             {bad}/{} bytes differ (stranded-cluster corruption, #72)",
            rgba.len()
        );
    }

    // libwebp must read our stream identically. The #72 bug produced a stream
    // every decoder accepted while returning the same garbage, so interop
    // agreement alone is not sufficient — but a divergence here means we emit
    // something the reference decoder reads differently, which is its own bug.
    let (lib, lw, lh) = webpx::decode_rgba(&webp)
        .unwrap_or_else(|e| panic!("{name} m{method} q{quality}: libwebp decode failed: {e:?}"));
    assert_eq!(
        (lw, lh),
        (w, h),
        "{name} m{method} q{quality}: libwebp dimensions"
    );
    assert_eq!(
        lib, rgba,
        "{name} m{method} q{quality}: libwebp decode of our stream diverges from source"
    );
}

/// Sweep the whole quality axis at one method for one image.
fn sweep_quality(name: &str, rgba: &[u8], w: u32, h: u32, method: u8) {
    for quality in (0..=100).step_by(5) {
        assert_lossless_exact(name, rgba, w, h, method, quality);
    }
}

/// The verified-stranding content: a 1-pixel crop of 512x320, seed 42. This is
/// the content the m4/m5/m6 x quality sweeps below run on — sweeping content
/// that never strands would cost CI time while proving nothing about #72
/// (general lossless roundtrip is already covered by `lossless_roundtrip.rs`
/// and `lossless_fast_tier.rs`).
fn stranding_content() -> (Vec<u8>, u32, u32) {
    let (w, h) = (511u32, 320u32);
    (multi_region(w, h, 42), w, h)
}

// The issue's ask: a gate over m4/m5/m6 x the quality axis on content known to
// strand a cluster. One test per method so a failure names the method directly.

#[test]
fn m4_quality_axis_stranding_content() {
    let (px, w, h) = stranding_content();
    sweep_quality("stranding_511x320", &px, w, h, 4);
}

#[test]
fn m5_quality_axis_stranding_content() {
    let (px, w, h) = stranding_content();
    sweep_quality("stranding_511x320", &px, w, h, 5);
}

#[test]
fn m6_quality_axis_stranding_content() {
    let (px, w, h) = stranding_content();
    sweep_quality("stranding_511x320", &px, w, h, 6);
}

/// Photographic content, the shape that first exposed this in the wild
/// (`o_8148.scale512x320` et al). Coarser quality steps — this content is not a
/// verified trigger, so it is a cheap net rather than the gate proper.
#[test]
fn photo_grain_coarse_axis() {
    let (w, h) = (384, 256);
    let px = photo_grain(w, h, 7);
    for method in [4u8, 6] {
        for quality in (0..=100).step_by(25) {
            assert_lossless_exact("photo_grain_384x256", &px, w, h, method, quality);
        }
    }
}

/// **The pinned reproduction.** `multi_region(511x320, seed 42)` at m4/q25 is a
/// verified trigger: with the `build_final_histograms` compaction reverted this
/// case fails with `BitStreamError`, and it passes with the fix. That mirrors
/// the original report exactly — a 1-pixel crop of a 512x320 rendition, failing
/// at m4/m5/m6 @ q=25 and nowhere else on the quality axis.
///
/// If this test ever goes green against a broken encoder, the content lottery
/// has shifted and the gate below is no longer proving anything — re-derive a
/// trigger (revert the compaction, sweep content, confirm a failure) rather
/// than trusting the sweep's silence.
#[test]
fn stranded_cluster_pinned_repro_511x320_m4_q25() {
    let (w, h) = (511, 320);
    assert_lossless_exact("pinned_511x320_s42", &multi_region(w, h, 42), w, h, 4, 25);
}

/// Odd/cropped dimensions: 1-pixel crops flipped the original failure between
/// silent corruption and `BitStreamError`, so tile-edge layout is part of the
/// trigger.
#[test]
fn odd_dimensions_layout_lottery() {
    for (w, h) in [(512u32, 319u32), (192, 256), (129, 97)] {
        let px = multi_region(w, h, 42);
        for method in [4u8, 5, 6] {
            // q25 is the quality the original renditions failed at; the rest is
            // coarse coverage of the axis.
            for quality in [0u32, 25, 50, 75, 100] {
                assert_lossless_exact(&format!("multi_region_{w}x{h}"), &px, w, h, method, quality);
            }
        }
    }
}

/// The default path users actually hit: `o_8384.scale192x256` failed at m6 with
/// default quality 75, so pin the shipped default explicitly.
#[test]
fn default_quality_m6_multi_region() {
    for seed in [1u64, 7, 42, 99] {
        for (w, h) in [(512u32, 320u32), (192, 256)] {
            let px = multi_region(w, h, seed);
            assert_lossless_exact(&format!("multi_region_{w}x{h}_s{seed}"), &px, w, h, 6, 75);
        }
    }
}
