//! Shared by `tests/lossless_yuv.rs` and `dev/lossless_yuv_oracle.rs`: the
//! synthetic RGBA sources and the plane digest. Lossless decoding returns the
//! source pixels exactly, so a digest depends only on these pixels and the
//! RGB->YUV conversion, never on the encoder.

/// Corpus lossless files (codec-corpus dataset, file name).
pub const CORPUS: &[(&str, &str)] = &[
    ("imageflow/test_inputs", "1_webp_ll.webp"),
    ("imageflow/test_inputs", "5_webp_ll.webp"),
    ("webp-conformance/valid", "multi-color.webp"),
    ("webp-conformance/valid", "simple_xmp.webp"),
    ("webp-conformance/valid", "2-color.webp"),
    ("webp-conformance/valid", "simple.webp"),
];

/// (name, width, height). Odd sizes exercise the edge chroma samples.
pub const SYNTHETIC: &[(&str, u32, u32)] = &[
    ("noise_alpha", 1, 1),
    ("noise_alpha", 2, 2),
    ("noise_alpha", 3, 5),
    ("noise_alpha", 17, 9),
    ("noise_alpha", 64, 64),
    ("gradient_opaque", 33, 17),
    ("gradient_opaque", 128, 96),
    ("alpha_ramp", 64, 31),
    ("hidden_rgb", 40, 40),
];

/// Deterministic RGBA pixels for a synthetic case.
pub fn synthetic_rgba(name: &str, w: u32, h: u32) -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32 ^ (w << 16) ^ h;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = next();
            let px = match name {
                // Random colour; alpha 0, 255 or partial in equal thirds.
                "noise_alpha" => {
                    let a = match r % 3 {
                        0 => 0,
                        1 => 255,
                        _ => (r >> 8) as u8,
                    };
                    [(r >> 16) as u8, (r >> 24) as u8, (r >> 4) as u8, a]
                }
                "gradient_opaque" => [
                    (x * 255 / w.max(1)) as u8,
                    (y * 255 / h.max(1)) as u8,
                    ((x + y) * 7) as u8,
                    255,
                ],
                // Smooth colour under an alpha ramp across x.
                "alpha_ramp" => [
                    (y * 8) as u8,
                    200 - (x * 3) as u8,
                    (x * 4) as u8,
                    (x * 255 / (w - 1)) as u8,
                ],
                // Fully transparent pixels keep arbitrary RGB (lossless
                // preserves it), next to opaque ones.
                "hidden_rgb" => {
                    let a = if (x / 3 + y / 2) % 2 == 0 { 0 } else { 255 };
                    [(r >> 16) as u8, (r >> 24) as u8, (r >> 4) as u8, a]
                }
                _ => panic!("unknown case {name}"),
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

/// Lossless encode that keeps RGB under fully transparent pixels, so the
/// decoded pixels are exactly the source.
pub fn encode_exact(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let config = zenwebp::EncoderConfig::Lossless(zenwebp::LosslessConfig::new().with_exact(true));
    zenwebp::EncodeRequest::new(&config, rgba, zenwebp::PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

/// FNV-1a over the plane dimensions and the tight Y, U, V planes.
pub fn yuv_digest(w: u32, h: u32, y: &[u8], u: &[u8], v: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in w
        .to_le_bytes()
        .iter()
        .chain(&h.to_le_bytes())
        .chain(y)
        .chain(u)
        .chain(v)
    {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
