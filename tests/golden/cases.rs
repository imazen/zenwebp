//! Synthetic full-coverage golden cases: encoder + decoder output hashes.
//!
//! Shared by two consumers, so it uses only the public API plus `alloc`:
//! - `src/lib.rs` (`golden_tests`, a `--no-default-features`-compatible lib
//!   unit test): runs on every CI target, wasm32 included, and asserts each
//!   case's encoded-bytes hash and decoded-output hash against
//!   `tests/golden/codec_golden.tsv`.
//! - `tests/golden_codec.rs` (native only): additionally decodes every case
//!   with libwebp in the same modes and requires identical digests, so the
//!   committed decode hashes are libwebp's outputs, not just ours. It is also
//!   the only writer of the golden file (`ZENWEBP_GOLDEN_BLESS=1`).
//!
//! Images are synthetic patterns chosen to reach distinct codec paths: flat
//! areas (skip / DC), gradients (I16 / chroma prediction), 1-px and 8-px
//! checkers (I4, high-frequency coefficients), noise (large coefficients,
//! cat6 tokens, no-LZ77), sharp strokes (loop-filter HEV), saturated bars
//! (chroma), ≤4- and ≤16-colour palettes (VP8L colour indexing at 2 and 4
//! bits per pixel), and smooth sinusoids — at sizes from 1×1 through odd
//! partial-macroblock shapes to multi-row images, with opaque, binary,
//! gradient, noisy and fully-transparent-with-colour alpha.
//!
//! Hashes are FNV-1a 64. A decode hash folds the digests of every decode mode
//! (`modes`) so one line pins all output layouts for that bitstream.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use zenwebp::decoder::UpsamplingMethod;
use zenwebp::mux::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DisposeMethod,
};
use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LosslessConfig, LossyConfig,
    PixelLayout, Preset,
};

/// Bump when the case list, a pattern, a mode, or the hash layout changes.
pub const FORMAT_VERSION: &str = "zenwebp-golden v1";

pub fn fnv(seed: u64, bytes: &[u8]) -> u64 {
    let mut x = seed;
    for &b in bytes {
        x ^= u64::from(b);
        x = x.wrapping_mul(0x100_0000_01b3);
    }
    x
}

pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// Digest of one decoded output: dimensions then bytes.
pub fn digest(px: &[u8], w: u32, h: u32) -> u64 {
    fnv(fnv(FNV0, &[w.to_le_bytes(), h.to_le_bytes()].concat()), px)
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

const PATTERNS: [&str; 11] = [
    "solid", "hgrad", "dgrad", "check1", "check8", "noise", "strokes", "bars", "pal4", "pal16",
    "sines",
];
const ALPHAS: [&str; 5] = ["opaque", "binary", "ramp", "noisy", "clear"];
const SIZES: [(u32, u32); 8] = [
    (1, 1),
    (2, 3),
    (7, 5),
    (16, 16),
    (17, 13),
    (33, 31),
    (64, 48),
    (130, 66),
];

fn hash2(x: u32, y: u32, salt: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ salt.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h
}

/// Deterministic integer-only RGBA for `pattern` with `alpha`.
pub fn image(pattern: usize, alpha: usize, w: u32, h: u32) -> Vec<u8> {
    const BARS: [[u8; 3]; 8] = [
        [255, 255, 255],
        [255, 255, 0],
        [0, 255, 255],
        [0, 255, 0],
        [255, 0, 255],
        [255, 0, 0],
        [0, 0, 255],
        [0, 0, 0],
    ];
    const SINE: [i32; 16] = [
        0, 39, 74, 96, 104, 96, 74, 39, 0, -39, -74, -96, -104, -96, -74, -39,
    ];
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (wd, hd) = (w.max(2) - 1, h.max(2) - 1);
            let rgb: [u8; 3] = match PATTERNS[pattern] {
                "solid" => [93, 140, 201],
                "hgrad" => {
                    let v = (x * 255 / wd) as u8;
                    [v, 255 - v, (v / 2).wrapping_add(64)]
                }
                "dgrad" => {
                    let v = ((x * 255 / wd + y * 255 / hd) / 2) as u8;
                    [v, (y * 255 / hd) as u8, 200u8.wrapping_sub(v / 3)]
                }
                "check1" => {
                    if (x + y).is_multiple_of(2) {
                        [250, 20, 120]
                    } else {
                        [10, 230, 40]
                    }
                }
                "check8" => {
                    if (x / 8 + y / 8).is_multiple_of(2) {
                        [255, 255, 255]
                    } else {
                        [0, 0, 0]
                    }
                }
                "noise" => {
                    let n = hash2(x, y, 1);
                    [n as u8, (n >> 8) as u8, (n >> 16) as u8]
                }
                "strokes" => {
                    let on = (x + 2 * y) % 11 < 2
                        || (3 * x).abs_diff(2 * y).is_multiple_of(13)
                        || x % 9 == 4;
                    if on { [20, 20, 30] } else { [235, 230, 220] }
                }
                "bars" => BARS[((x * 8) / w.max(1)) as usize % 8],
                "pal4" => [[0, 0, 0], [255, 0, 0], [0, 128, 255], [250, 250, 250]]
                    [(hash2(x / 3, y / 2, 4) % 4) as usize],
                "pal16" => {
                    let i = (hash2(x / 2, y / 2, 16) % 16) as u8;
                    // Wrapping, so debug and release builds agree.
                    [i * 16, 255 - i * 13, i.wrapping_mul(37) % 255]
                }
                _ => {
                    // sines
                    let s = |v: u32, k: u32| SINE[((v * k) % 16) as usize];
                    let base = 128 + s(x, 1) + s(y, 2) / 2;
                    let c = |v: i32| v.clamp(0, 255) as u8;
                    [
                        c(base),
                        c(128 + s(x + y, 1)),
                        c(128 - s(x, 3) / 2 + s(y, 1) / 2),
                    ]
                }
            };
            let a: u8 = match ALPHAS[alpha] {
                "opaque" => 255,
                "binary" => {
                    let (cx, cy) = (w as i64 / 2, h as i64 / 2);
                    let (dx, dy) = (x as i64 - cx, y as i64 - cy);
                    if dx * dx + dy * dy <= (cx * cx + cy * cy) / 2 {
                        255
                    } else {
                        0
                    }
                }
                "ramp" => (y * 255 / (h.max(2) - 1)) as u8,
                "noisy" => (hash2(x, y, 99) >> 24) as u8,
                _ => 0, // "clear": invisible colour, kept only by `exact`
            };
            out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Encoder configurations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum Cfg {
    /// quality, method, segments, sns, filter strength, sharpness, alpha quality
    Lossy(u8, u8, u8, u8, u8, u8, u8),
    /// preset, quality
    Preset(Preset, u8),
    /// quality, method, near_lossless, exact
    Lossless(u8, u8, u8, bool),
}

const LOSSY: [Cfg; 14] = [
    Cfg::Lossy(75, 4, 4, 50, 60, 0, 100),
    Cfg::Lossy(0, 0, 1, 0, 0, 0, 100),
    Cfg::Lossy(5, 1, 4, 100, 100, 7, 100),
    Cfg::Lossy(30, 2, 2, 30, 20, 3, 70),
    Cfg::Lossy(50, 3, 3, 80, 40, 5, 50),
    Cfg::Lossy(90, 5, 4, 50, 60, 2, 100),
    Cfg::Lossy(100, 6, 4, 0, 0, 0, 100),
    Cfg::Lossy(60, 6, 1, 0, 0, 0, 0),
    Cfg::Lossy(20, 4, 4, 70, 90, 1, 30),
    Cfg::Lossy(95, 3, 1, 25, 10, 6, 100),
    Cfg::Preset(Preset::Photo, 80),
    Cfg::Preset(Preset::Drawing, 45),
    Cfg::Preset(Preset::Icon, 70),
    Cfg::Preset(Preset::Text, 25),
];

const LOSSLESS: [Cfg; 8] = [
    Cfg::Lossless(75, 4, 100, false),
    Cfg::Lossless(0, 0, 100, false),
    Cfg::Lossless(100, 6, 100, true),
    Cfg::Lossless(50, 2, 60, false),
    Cfg::Lossless(90, 5, 0, false),
    Cfg::Lossless(25, 1, 80, true),
    Cfg::Lossless(100, 3, 40, false),
    Cfg::Lossless(60, 6, 100, true),
];

fn encoder_config(c: Cfg) -> EncoderConfig {
    match c {
        Cfg::Lossy(q, m, s, sns, f, sh, aq) => EncoderConfig::Lossy(
            LossyConfig::new()
                .with_quality(f32::from(q))
                .with_method(m)
                .with_segments(s)
                .with_sns_strength(sns)
                .with_filter_strength(f)
                .with_filter_sharpness(sh)
                .with_alpha_quality(aq),
        ),
        Cfg::Preset(p, q) => EncoderConfig::Lossy(LossyConfig::with_preset(p, f32::from(q))),
        Cfg::Lossless(q, m, nl, ex) => EncoderConfig::Lossless(
            LosslessConfig::new()
                .with_quality(f32::from(q))
                .with_method(m)
                .with_near_lossless(nl)
                .with_exact(ex),
        ),
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum Source {
    Still {
        pattern: usize,
        alpha: usize,
        w: u32,
        h: u32,
        cfg: Cfg,
    },
    /// Animation variant 0..ANIMS.
    Anim(usize),
}

pub struct Case {
    pub name: String,
    pub source: Source,
}

const ANIMS: usize = 6;

/// The fixed case list. Each still image gets two lossy and one lossless
/// configuration, rotating through the lists so every configuration meets
/// many patterns, alpha modes and sizes.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let mut k = 0usize;
    // Names are compact to keep the golden file small:
    // `p<PATTERNS idx>a<ALPHAS idx>-<w>x<h>-<L<LOSSY idx>|N<LOSSLESS idx>>`.
    let mut push = |pattern: usize, alpha: usize, (w, h): (u32, u32), out: &mut Vec<Case>| {
        let (l1, l2, n) = (
            k % LOSSY.len(),
            (k * 5 + 7) % LOSSY.len(),
            k % LOSSLESS.len(),
        );
        for (tag, cfg) in [
            (format!("L{l1:02}"), LOSSY[l1]),
            (format!("L{l2:02}"), LOSSY[l2]),
            (format!("N{n:02}"), LOSSLESS[n]),
        ] {
            out.push(Case {
                name: format!("p{pattern:02}a{alpha}-{w}x{h}-{tag}"),
                source: Source::Still {
                    pattern,
                    alpha,
                    w,
                    h,
                    cfg,
                },
            });
        }
        k += 1;
    };
    // Every pattern at every size, opaque.
    for p in 0..PATTERNS.len() {
        for &s in &SIZES {
            push(p, 0, s, &mut out);
        }
    }
    // Alpha modes on a representative subset.
    for p in [1usize, 5, 6, 8, 10] {
        for a in 1..ALPHAS.len() {
            for &s in &[(1u32, 1u32), (17, 13), (33, 31), (64, 48)] {
                push(p, a, s, &mut out);
            }
        }
    }
    for i in 0..ANIMS {
        out.push(Case {
            name: format!("anim{i}"),
            source: Source::Anim(i),
        });
    }
    out
}

/// Encode a case with zenwebp.
pub fn encode(case: &Case) -> Vec<u8> {
    match case.source {
        Source::Still {
            pattern,
            alpha,
            w,
            h,
            cfg,
        } => {
            let rgba = image(pattern, alpha, w, h);
            let c = encoder_config(cfg);
            let r = match &c {
                EncoderConfig::Lossy(l) => EncodeRequest::lossy(l, &rgba, PixelLayout::Rgba8, w, h),
                EncoderConfig::Lossless(l) => {
                    EncodeRequest::lossless(l, &rgba, PixelLayout::Rgba8, w, h)
                }
            };
            r.encode()
                .unwrap_or_else(|e| panic!("{}: encode failed: {e:?}", case.name))
        }
        Source::Anim(i) => encode_anim(i),
    }
}

/// Animations exercising keyframes, blend vs overwrite, background dispose,
/// sub-frames, mixed lossy/lossless frames, and the delta-frame optimiser.
fn encode_anim(i: usize) -> Vec<u8> {
    let (cw, ch) = (48u32, 40u32);
    let cfg = AnimationConfig {
        minimize_size: i % 3 == 2,
        ..Default::default()
    };
    let mut enc = AnimationEncoder::new(cw, ch, cfg).unwrap();
    let lossy = encoder_config(LOSSY[(i * 3) % LOSSY.len()]);
    let lossless = encoder_config(LOSSLESS[(i * 5) % LOSSLESS.len()]);
    for f in 0..4u32 {
        let full = f == 0 || i.is_multiple_of(2);
        let (fw, fh, x, y) = if full {
            (cw, ch, 0, 0)
        } else {
            (24, 20, 2 * f, 4 * f)
        };
        let alpha = [0usize, 2, 1, 3][((i as u32 + f) % 4) as usize];
        let px = image([10usize, 5, 6, 1][f as usize], alpha, fw, fh);
        let c = if (i + f as usize).is_multiple_of(2) {
            &lossy
        } else {
            &lossless
        };
        let blend = if (i + f as usize).is_multiple_of(3) {
            BlendMethod::Overwrite
        } else {
            BlendMethod::AlphaBlend
        };
        let dispose = if (i * 7 + f as usize) % 4 == 1 {
            DisposeMethod::Background
        } else {
            DisposeMethod::None
        };
        if cfg_is_delta(i) && full {
            enc.add_frame(&px, PixelLayout::Rgba8, f * 100, c).unwrap();
        } else {
            enc.add_frame_advanced(
                &px,
                PixelLayout::Rgba8,
                fw,
                fh,
                x,
                y,
                f * 100,
                c,
                dispose,
                blend,
            )
            .unwrap();
        }
    }
    enc.finalize(100).unwrap()
}

fn cfg_is_delta(i: usize) -> bool {
    i % 3 == 2
}

// ---------------------------------------------------------------------------
// Decode modes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Lossy,
    Lossless,
    Anim,
}

pub fn kind(webp: &[u8]) -> Kind {
    match &webp[12..16] {
        b"VP8 " => Kind::Lossy,
        b"VP8L" => Kind::Lossless,
        _ if webp[20] & 0x02 != 0 => Kind::Anim,
        _ => {
            // VP8X still: walk the chunks for the image chunk.
            let mut off = 12usize;
            while off + 8 <= webp.len() {
                let sz = u32::from_le_bytes(webp[off + 4..off + 8].try_into().unwrap()) as usize;
                match &webp[off..off + 4] {
                    b"VP8 " => return Kind::Lossy,
                    b"VP8L" => return Kind::Lossless,
                    _ => off += 8 + sz + (sz & 1),
                }
            }
            panic!("no image chunk")
        }
    }
}

/// Decode modes per kind, named after the libwebp call they must match.
pub fn modes(k: Kind) -> &'static [&'static str] {
    match k {
        Kind::Anim => &["anim"],
        Kind::Lossless => &[
            "rgba", "rgb", "bgra", "bgr", "argb", "rgbA", "bgrA", "Argb", "rgb565", "rgba4444",
        ],
        Kind::Lossy => &[
            "rgba",
            "rgb",
            "bgra",
            "bgr",
            "argb",
            "rgbA",
            "bgrA",
            "Argb",
            "rgb565",
            "rgba4444",
            "yuv",
            "rgba_nofancy",
            "rgba_dither50",
        ],
    }
}

/// zenwebp's output for one mode, as `(bytes, width, height)`.
pub fn zen_decode(mode: &str, d: &[u8]) -> (Vec<u8>, u32, u32) {
    use zenwebp::oneshot as o;
    let r = match mode {
        "rgba" => o::decode_rgba(d),
        "rgb" => o::decode_rgb(d),
        "bgra" => o::decode_bgra(d),
        "bgr" => o::decode_bgr(d),
        "argb" => o::decode_argb(d),
        "rgbA" => o::decode_rgba_premultiplied(d),
        "bgrA" => o::decode_bgra_premultiplied(d),
        "Argb" => o::decode_argb_premultiplied(d),
        "rgb565" => o::decode_rgb565(d),
        "rgba4444" => o::decode_rgba4444(d),
        "rgba_nofancy" => {
            let c = DecodeConfig::default().upsampling(UpsamplingMethod::Simple);
            DecodeRequest::new(&c, d).decode_rgba()
        }
        "rgba_dither50" => {
            let c = DecodeConfig::default().with_dithering_strength(50);
            DecodeRequest::new(&c, d).decode_rgba()
        }
        "yuv" => {
            let p = o::decode_yuv420(d).unwrap();
            let mut b = p.y.clone();
            b.extend_from_slice(&p.u);
            b.extend_from_slice(&p.v);
            return (b, p.y_width, p.y_height);
        }
        "anim" => {
            let mut dec = AnimationDecoder::new(d).unwrap();
            let frames = dec.decode_all().unwrap();
            let (w, h) = (frames[0].width, frames[0].height);
            let mut b = Vec::new();
            for f in &frames {
                if f.data.len() == (w * h * 3) as usize {
                    for p in f.data.as_chunks::<3>().0 {
                        b.extend_from_slice(&[p[0], p[1], p[2], 255]);
                    }
                } else {
                    b.extend_from_slice(&f.data);
                }
            }
            return (b, w, h * frames.len() as u32);
        }
        other => panic!("unknown mode {other}"),
    };
    r.unwrap_or_else(|e| panic!("zen decode {mode}: {e:?}"))
}

/// Fold per-mode digests into the single decode hash stored per case.
pub fn fold(digests: &[(&str, u64)]) -> u64 {
    let mut x = FNV0;
    for (m, d) in digests {
        x = fnv(x, m.as_bytes());
        x = fnv(x, &d.to_le_bytes());
    }
    x
}

/// `(encoded hash, per-mode digests)` for a case, all through zenwebp.
pub fn zen_hashes(case: &Case) -> (Vec<u8>, u64, Vec<(&'static str, u64)>) {
    let webp = encode(case);
    let enc = fnv(FNV0, &webp);
    let ds = modes(kind(&webp))
        .iter()
        .map(|&m| {
            let (b, w, h) = zen_decode(m, &webp);
            (m, digest(&b, w, h))
        })
        .collect();
    (webp, enc, ds)
}

/// Parse the golden file: `name\tenc\tdec` lines after the version header.
pub fn parse_golden(text: &str) -> Vec<(String, u64, u64)> {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(FORMAT_VERSION),
        "golden file format version"
    );
    lines
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split('\t');
            let name = it.next().unwrap().into();
            let enc = u64::from_str_radix(it.next().unwrap(), 16).unwrap();
            let dec = u64::from_str_radix(it.next().unwrap(), 16).unwrap();
            (name, enc, dec)
        })
        .collect()
}

pub fn format_line(name: &str, enc: u64, dec: u64) -> String {
    format!("{name}\t{enc:016x}\t{dec:016x}")
}
