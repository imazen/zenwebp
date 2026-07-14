//! Regression: VP8L pixel-loop bit-window refills on deep-Huffman-tree streams.
//!
//! A literal pixel reads GREEN + RED + BLUE + ALPHA codes, each up to 15 bits
//! (60 bits worst case), but `BitReader::fill()` guarantees only 56. The
//! decoder must therefore refill mid-pixel with enough headroom for BLUE +
//! ALPHA (30 bits) after RED; an undersized guard surfaces as a spurious
//! `BitStreamError` partway through a valid stream. Found 2026-07-13 on a
//! libwebp `-m 0 -lossless` encode of a photographic RGBA image (libwebp's
//! low-effort mode emits literal-heavy streams with near-max-depth trees).
//!
//! The backward-reference path has the analogous budget (DIST symbol 15 bits +
//! up to 18 distance extra bits = 33) and is fixed by the same change; forcing
//! an 18-extra-bit distance requires a >1 MP image so it is covered by budget
//! arithmetic rather than a dedicated stream here.
//!
//! This test synthesizes an image whose channel histograms force 14-15 bit
//! codes on all four trees simultaneously (rare pixels are rare on every
//! channel at once), encodes it with libwebp at several method levels, and
//! requires zenwebp to decode every stream to the same pixels libwebp does.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

/// Deterministic xorshift64* PRNG — no external dependency, stable across runs.
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
        (self.next() >> 32) as u8
    }
}

/// Image whose per-channel value distribution yields maximum-depth Huffman
/// trees: ~99% of pixels draw all channels from a small "common" set (short
/// codes), ~1% draw all channels from a 224-value "tail" (each tail value
/// appears ~20 times in 480k pixels → 14-15 bit codes). Because rarity is
/// correlated across channels, single literals cost up to ~60 bits — the
/// worst case the bit-window refill logic must absorb.
fn deep_tree_image(w: u32, h: u32) -> Vec<u8> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        let rare = rng.next().is_multiple_of(100);
        for _channel in 0..4u8 {
            let v = if rare {
                32 + (rng.byte() % 224) // tail: 32..=255, ~21 occurrences each
            } else {
                rng.byte() % 32 // common: 0..=31, short codes
            };
            rgba.push(v);
        }
    }
    // Avoid alpha=0 so `exact` cleanup semantics never enter the picture.
    for px in rgba.chunks_exact_mut(4) {
        if px[3] == 0 {
            px[3] = 1;
        }
    }
    rgba
}

#[test]
fn decode_libwebp_deep_tree_streams() {
    let (w, h) = (800u32, 600u32);
    let rgba = deep_tree_image(w, h);

    for method in [0u8, 1, 4, 6] {
        let webp = webpx::EncoderConfig::new_lossless()
            .exact(true)
            .method(method)
            .encode_rgba(&rgba, w, h, webpx::Unstoppable)
            .expect("libwebp encode");

        let (lib_pixels, lw, lh) =
            webpx::decode_rgba(&webp).expect("libwebp decodes its own stream");
        assert_eq!((lw, lh), (w, h));
        assert_eq!(
            lib_pixels, rgba,
            "libwebp roundtrip must be lossless (m{method})"
        );

        let (zen_pixels, zw, zh) = zenwebp::oneshot::decode_rgba(&webp).unwrap_or_else(|e| {
            panic!("zenwebp failed to decode valid libwebp m{method} stream: {e:?}")
        });
        assert_eq!((zw, zh), (w, h));
        assert_eq!(
            zen_pixels, lib_pixels,
            "zenwebp pixels diverge from libwebp on m{method} deep-tree stream"
        );
    }
}
