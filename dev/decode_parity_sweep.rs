//! Wide-corpus decoder parity sweep: zenwebp vs libwebp (webpx, libwebp 1.6.0).
//!
//! Every input file is decoded by both decoders in every output mode that
//! applies, and the outputs are compared byte for byte (tolerance 0):
//!
//! - `rgba`      fancy upsampling, RGBA       (still images)
//! - `rgb`       fancy upsampling, RGB        (still images)
//! - `rgba_nofancy` point-sampled chroma (`-nofancy`) (lossy still images)
//! - `yuv`       raw YUV 4:2:0 planes          (lossy still images)
//! - `anim`      every composited canvas       (animated images)
//!
//! Inputs come from two places:
//! - `--files DIR` (repeatable): every `*.webp` under DIR, as-is. Files that
//!   both decoders reject count as agreement; one-sided rejection is a finding.
//! - `--gen-src DIR` (repeatable): every PNG under DIR is a source. Each source
//!   is cut into several variants (full, odd crops, tiny, synthetic alpha) and
//!   each variant is encoded by BOTH libwebp and zenwebp with a pseudo-random
//!   but seeded draw over the encoder knob space (quality 0-100, method 0-6,
//!   segments, sns, filter strength/sharpness/type, partitions, sharp_yuv,
//!   alpha quality/filter/compression, lossless level, near-lossless, exact).
//!   A few animations per source exercise blend/dispose compositing.
//!
//! Output: a TSV row per (file, mode) to `--out`, a summary on stdout, and every
//! mismatching input copied into `--mismatch-dir` for repro.
//!
//! Usage:
//!   cargo run --release --features __expert --example decode_parity_sweep -- \
//!     --files ~/.cache/codec-corpus/v1/webp-conformance \
//!     --gen-src ~/.cache/codec-corpus/v1/CID22 --encodes-per-variant 4 \
//!     --out benchmarks/decode_parity_2026-10-01.tsv --mismatch-dir ~/tmp/dp-mismatch
#![forbid(unsafe_code)]

use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use zenwebp::decoder::UpsamplingMethod;
use zenwebp::mux::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DisposeMethod,
};
use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LosslessConfig, LossyConfig,
    PixelLayout,
};

// ---------------------------------------------------------------------------
// RNG (splitmix64, seeded per source so runs are reproducible)
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u32, hi_incl: u32) -> u32 {
        lo + (self.next() % u64::from(hi_incl - lo + 1)) as u32
    }
    fn chance(&mut self, pct: u32) -> bool {
        self.range(0, 99) < pct
    }
}

fn hash_str(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Row {
    file: String,
    mode: &'static str,
    status: &'static str, // match | diff | dim | zen_err | lib_err | both_err | zen_panic
    max_diff: u8,
    diff_bytes: u64,
    total_bytes: u64,
    note: String,
}

fn cmp_bytes(a: &[u8], b: &[u8]) -> (u8, u64) {
    let mut max = 0u8;
    let mut n = 0u64;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        if d != 0 {
            n += 1;
            max = max.max(d);
        }
    }
    (max, n)
}

type Out = Result<(Vec<u8>, u32, u32), String>;

fn compare(file: &str, mode: &'static str, zen: std::thread::Result<Out>, lib: Out) -> Row {
    let mut row = Row {
        file: file.to_string(),
        mode,
        status: "match",
        max_diff: 0,
        diff_bytes: 0,
        total_bytes: 0,
        note: String::new(),
    };
    let zen = match zen {
        Ok(z) => z,
        Err(_) => {
            row.status = "zen_panic";
            return row;
        }
    };
    match (zen, lib) {
        (Ok((zp, zw, zh)), Ok((lp, lw, lh))) => {
            row.total_bytes = lp.len() as u64;
            if (zw, zh) != (lw, lh) || zp.len() != lp.len() {
                row.status = "dim";
                row.note = format!("zen {zw}x{zh}/{} lib {lw}x{lh}/{}", zp.len(), lp.len());
            } else if zp != lp {
                let (m, n) = cmp_bytes(&zp, &lp);
                row.status = "diff";
                row.max_diff = m;
                row.diff_bytes = n;
            }
        }
        (Err(ze), Ok(_)) => {
            row.status = "zen_err";
            row.note = ze;
        }
        (Ok(_), Err(le)) => {
            row.status = "lib_err";
            row.note = le;
        }
        (Err(_), Err(_)) => row.status = "both_err",
    }
    row.note = row.note.replace(['\t', '\n'], " ");
    row.note.truncate(160);
    row
}

fn quiet_catch<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> std::thread::Result<T> {
    std::panic::catch_unwind(f)
}

/// Pack YUV planes tightly (y, then u, then v) so both sides compare equal-shape.
#[allow(clippy::too_many_arguments)]
fn pack_planes(
    y: &[u8],
    ys: usize,
    u: &[u8],
    us: usize,
    v: &[u8],
    vs: usize,
    w: usize,
    h: usize,
) -> Option<Vec<u8>> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for r in 0..h {
        out.extend_from_slice(y.get(r * ys..r * ys + w)?);
    }
    for r in 0..ch {
        out.extend_from_slice(u.get(r * us..r * us + cw)?);
    }
    for r in 0..ch {
        out.extend_from_slice(v.get(r * vs..r * vs + cw)?);
    }
    Some(out)
}

enum Kind {
    Lossy,
    Lossless,
    Animated,
    Unknown,
}

fn kind(data: &[u8]) -> Kind {
    if data.len() < 16 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Kind::Unknown;
    }
    match &data[12..16] {
        b"VP8 " => Kind::Lossy,
        b"VP8L" => Kind::Lossless,
        b"VP8X" => {
            if data.len() > 20 && data[20] & 0x02 != 0 {
                return Kind::Animated;
            }
            // Find the image chunk.
            let mut off = 12usize;
            while off + 8 <= data.len() {
                let id = &data[off..off + 4];
                let sz = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
                if id == b"VP8 " {
                    return Kind::Lossy;
                }
                if id == b"VP8L" {
                    return Kind::Lossless;
                }
                off = off.saturating_add(8).saturating_add(sz + (sz & 1));
            }
            Kind::Unknown
        }
        _ => Kind::Unknown,
    }
}

fn check_file(name: &str, data: &[u8]) -> Vec<Row> {
    let mut rows = Vec::new();
    let k = kind(data);
    if matches!(k, Kind::Animated) {
        let zen = quiet_catch(|| -> Out {
            let mut d = AnimationDecoder::new(data).map_err(|e| format!("{e}"))?;
            let frames = d.decode_all().map_err(|e| format!("{e}"))?;
            let (w, h) = frames
                .first()
                .map(|f| (f.width, f.height))
                .unwrap_or((0, 0));
            let mut all = Vec::new();
            for f in &frames {
                // AnimFrame.data is RGB (3 bpp) when the file has no alpha;
                // libwebp's WebPAnimDecoder always emits RGBA.
                if f.data.len() == (f.width * f.height * 3) as usize {
                    for p in f.data.as_chunks::<3>().0.iter() {
                        all.extend_from_slice(&[p[0], p[1], p[2], 255]);
                    }
                } else {
                    all.extend_from_slice(&f.data);
                }
            }
            Ok((all, w, h * frames.len() as u32))
        });
        let lib = (|| -> Out {
            let mut d = webpx::AnimationDecoder::new(data).map_err(|e| format!("{e}"))?;
            let frames = d.decode_all().map_err(|e| format!("{e}"))?;
            let (w, h) = frames
                .first()
                .map(|f| (f.width, f.height))
                .unwrap_or((0, 0));
            let mut all = Vec::new();
            for f in &frames {
                all.extend_from_slice(&f.data);
            }
            Ok((all, w, h * frames.len() as u32))
        })();
        rows.push(compare(name, "anim", zen, lib));
        return rows;
    }

    let zen = quiet_catch(|| zenwebp::oneshot::decode_rgba(data).map_err(|e| format!("{e}")));
    let lib = webpx::decode_rgba(data).map_err(|e| format!("{e}"));
    rows.push(compare(name, "rgba", zen, lib));

    let zen = quiet_catch(|| zenwebp::oneshot::decode_rgb(data).map_err(|e| format!("{e}")));
    let lib = webpx::decode_rgb(data).map_err(|e| format!("{e}"));
    rows.push(compare(name, "rgb", zen, lib));

    if matches!(k, Kind::Lossy) {
        let zen = quiet_catch(|| {
            let cfg = DecodeConfig::default().upsampling(UpsamplingMethod::Simple);
            DecodeRequest::new(&cfg, data)
                .decode_rgba()
                .map_err(|e| format!("{e}"))
        });
        let lib = (|| -> Out {
            webpx::Decoder::new(data)
                .map_err(|e| format!("{e}"))?
                .config(webpx::DecoderConfig::new().no_fancy_upsampling(true))
                .decode_rgba_raw()
                .map_err(|e| format!("{e}"))
        })();
        rows.push(compare(name, "rgba_nofancy", zen, lib));

        let zen = quiet_catch(|| -> Out {
            let p = zenwebp::oneshot::decode_yuv420(data).map_err(|e| format!("{e}"))?;
            let (w, h) = (p.y_width as usize, p.y_height as usize);
            let packed = pack_planes(
                &p.y,
                w,
                &p.u,
                p.uv_width as usize,
                &p.v,
                p.uv_width as usize,
                w,
                h,
            )
            .ok_or("zen yuv plane short")?;
            Ok((packed, p.y_width, p.y_height))
        });
        let lib = (|| -> Out {
            let p = webpx::decode_yuv(data).map_err(|e| format!("{e}"))?;
            let packed = pack_planes(
                &p.y,
                p.y_stride,
                &p.u,
                p.u_stride,
                &p.v,
                p.v_stride,
                p.width as usize,
                p.height as usize,
            )
            .ok_or("lib yuv plane short")?;
            Ok((packed, p.width, p.height))
        })();
        rows.push(compare(name, "yuv", zen, lib));
    }
    rows
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

fn load_png_rgba8(path: &Path) -> Option<(Vec<u8>, u32, u32, bool)> {
    use zenpixels_convert::PixelBufferConvertTypedExt;
    use zenpng::PngDecodeConfig;
    let bytes = std::fs::read(path).ok()?;
    let output = zenpng::decode(&bytes, &PngDecodeConfig::default(), &zenwebp::Unstoppable).ok()?;
    let (w, h) = (output.info.width, output.info.height);
    let buf = output.pixels.to_rgba8();
    let slice = buf.as_slice();
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        out.extend_from_slice(&slice.row(y)[..(w as usize) * 4]);
    }
    let has_alpha = out.as_chunks::<4>().0.iter().any(|p| p[3] != 255);
    Some((out, w, h, has_alpha))
}

fn crop(src: &[u8], sw: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for r in y..y + h {
        let s = ((r * sw + x) * 4) as usize;
        out.extend_from_slice(&src[s..s + (w * 4) as usize]);
    }
    out
}

fn synth_alpha(px: &mut [u8], w: u32, h: u32, style: u32) {
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4 + 3) as usize;
            px[i] = match style {
                0 => ((x * 255) / w.max(1)) as u8, // gradient
                1 => {
                    if (x / 8 + y / 8) % 2 == 0 {
                        0
                    } else {
                        255
                    }
                } // hard checker
                2 => ((x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503)) >> 7) as u8, // noise
                _ => {
                    if x * x + y * y < (w * w) / 4 {
                        255
                    } else {
                        0
                    }
                } // disc, mostly 0
            };
        }
    }
}

struct Variant {
    tag: String,
    rgba: Vec<u8>,
    w: u32,
    h: u32,
    alpha: bool,
}

fn variants(
    name: &str,
    rgba: &[u8],
    w: u32,
    h: u32,
    has_alpha: bool,
    rng: &mut Rng,
) -> Vec<Variant> {
    let mut v = vec![Variant {
        tag: format!("{name}_full"),
        rgba: rgba.to_vec(),
        w,
        h,
        alpha: has_alpha,
    }];
    let sizes: [(u32, u32); 6] = [(1, 1), (3, 5), (17, 13), (33, 31), (127, 65), (250, 199)];
    for &(cw, ch) in &sizes {
        if cw > w || ch > h || !rng.chance(50) {
            continue;
        }
        let x = rng.range(0, w - cw);
        let y = rng.range(0, h - ch);
        v.push(Variant {
            tag: format!("{name}_c{cw}x{ch}"),
            rgba: crop(rgba, w, x, y, cw, ch),
            w: cw,
            h: ch,
            alpha: has_alpha,
        });
    }
    // Synthetic-alpha variant of a mid-size crop.
    let (cw, ch) = (w.min(161), h.min(97));
    let mut a = crop(rgba, w, 0, 0, cw, ch);
    let style = rng.range(0, 3);
    synth_alpha(&mut a, cw, ch, style);
    v.push(Variant {
        tag: format!("{name}_a{style}_{cw}x{ch}"),
        rgba: a,
        w: cw,
        h: ch,
        alpha: true,
    });
    v
}

fn to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

/// One encode of `v`, by libwebp or zenwebp, with a random knob draw.
fn encode_one(v: &Variant, rng: &mut Rng, by_lib: bool) -> Option<(String, Vec<u8>)> {
    let lossless = rng.chance(30);
    let use_alpha = v.alpha;
    let pixels_rgba;
    let pixels: &[u8] = if use_alpha {
        &v.rgba
    } else {
        pixels_rgba = to_rgb(&v.rgba);
        &pixels_rgba
    };
    let mut tag = String::new();
    let q = rng.range(0, 100);
    let m = rng.range(0, 6) as u8;
    if lossless {
        let nl = if rng.chance(30) {
            rng.range(0, 100) as u8
        } else {
            100
        };
        let exact = rng.chance(30);
        write!(tag, "ll_q{q}_m{m}_nl{nl}_ex{}", exact as u8).ok();
        if by_lib {
            let cfg = webpx::EncoderConfig::new_lossless()
                .quality(q as f32)
                .method(m)
                .near_lossless(nl)
                .exact(exact);
            let r = if use_alpha {
                cfg.encode_rgba(pixels, v.w, v.h, webpx::Unstoppable)
            } else {
                cfg.encode_rgb(pixels, v.w, v.h, webpx::Unstoppable)
            };
            return r.ok().map(|b| (tag, b));
        }
        let cfg = LosslessConfig::new()
            .with_quality(q as f32)
            .with_method(m)
            .with_near_lossless(nl)
            .with_exact(exact);
        let layout = if use_alpha {
            PixelLayout::Rgba8
        } else {
            PixelLayout::Rgb8
        };
        return EncodeRequest::lossless(&cfg, pixels, layout, v.w, v.h)
            .encode()
            .ok()
            .map(|b| (tag, b));
    }
    let segs = rng.range(1, 4) as u8;
    let sns = rng.range(0, 100) as u8;
    let flt = if rng.chance(20) {
        0
    } else {
        rng.range(0, 100) as u8
    };
    let sharp = rng.range(0, 7) as u8;
    let ftype = rng.range(0, 1) as u8;
    let parts = rng.range(0, 3) as u8;
    let syuv = rng.chance(20);
    let aq = rng.range(0, 100) as u8;
    let afilt = rng.range(0, 3);
    let acomp = rng.chance(85);
    write!(
        tag,
        "ly_q{q}_m{m}_s{segs}_sns{sns}_f{flt}_sh{sharp}_ft{ftype}_p{parts}_sy{}",
        syuv as u8
    )
    .ok();
    if use_alpha {
        write!(tag, "_aq{aq}_af{afilt}_ac{}", acomp as u8).ok();
    }
    if by_lib {
        let mut cfg = webpx::EncoderConfig::new()
            .quality(q as f32)
            .method(m)
            .segments(segs)
            .sns_strength(sns)
            .filter_strength(flt)
            .filter_sharpness(sharp)
            .filter_type(ftype)
            .partitions(parts)
            .sharp_yuv(syuv)
            .alpha_quality(aq)
            .alpha_compression(acomp)
            .alpha_filter(match afilt {
                0 => webpx::AlphaFilter::None,
                1 => webpx::AlphaFilter::Fast,
                _ => webpx::AlphaFilter::Best,
            });
        if rng.chance(15) {
            cfg = cfg.preprocessing(rng.range(0, 7) as u8);
            tag.push_str("_pre");
        }
        let r = if use_alpha {
            cfg.encode_rgba(pixels, v.w, v.h, webpx::Unstoppable)
        } else {
            cfg.encode_rgb(pixels, v.w, v.h, webpx::Unstoppable)
        };
        return r.ok().map(|b| (tag, b));
    }
    let mut cfg = LossyConfig::new()
        .with_quality(q as f32)
        .with_method(m)
        .with_segments(segs)
        .with_sns_strength(sns)
        .with_filter_strength(flt)
        .with_filter_sharpness(sharp)
        .with_sharp_yuv(syuv)
        .with_alpha_quality(aq);
    if rng.chance(30) {
        cfg = cfg.with_cost_model(zenwebp::CostModel::StrictLibwebpParity);
        tag.push_str("_strict");
    }
    let layout = if use_alpha {
        PixelLayout::Rgba8
    } else {
        PixelLayout::Rgb8
    };
    EncodeRequest::lossy(&cfg, pixels, layout, v.w, v.h)
        .encode()
        .ok()
        .map(|b| (tag, b))
}

/// A short animation built from shifted crops of `rgba`.
fn encode_anim(
    name: &str,
    rgba: &[u8],
    w: u32,
    h: u32,
    rng: &mut Rng,
    by_lib: bool,
) -> Option<(String, Vec<u8>)> {
    let (cw, ch) = (w.min(96), h.min(80));
    if cw < 8 || ch < 8 {
        return None;
    }
    let nframes = rng.range(2, 5);
    let mut frames = Vec::new();
    for i in 0..nframes {
        let x = (i * 7).min(w - cw);
        let y = (i * 5).min(h - ch);
        let mut f = crop(rgba, w, x, y, cw, ch);
        if rng.chance(40) {
            synth_alpha(&mut f, cw, ch, rng.range(0, 3));
        }
        frames.push(f);
    }
    let lossless = rng.chance(40);
    let q = rng.range(0, 100) as f32;
    let tag = format!(
        "{name}_anim{nframes}_{}_q{q}",
        if lossless { "ll" } else { "ly" }
    );
    if by_lib {
        let mut enc = webpx::AnimationEncoder::new(cw, ch).ok()?;
        enc.set_quality(q);
        enc.set_lossless(lossless);
        for (i, f) in frames.iter().enumerate() {
            enc.add_frame_rgba(f, (i as i32) * 100).ok()?;
        }
        return enc
            .finish(nframes as i32 * 100)
            .ok()
            .map(|b| (format!("lib_{tag}"), b));
    }
    let mut enc = AnimationEncoder::new(cw, ch, AnimationConfig::default()).ok()?;
    let cfg = if lossless {
        EncoderConfig::Lossless(LosslessConfig::new().with_quality(q))
    } else {
        EncoderConfig::Lossy(LossyConfig::new().with_quality(q))
    };
    for (i, f) in frames.iter().enumerate() {
        // Mix the delta-optimised path with explicit blend/dispose sub-frames.
        if i > 0 && rng.chance(40) {
            let (sw, sh) = (cw / 2, ch / 2);
            let sub = crop(f, cw, 0, 0, sw, sh);
            let ox = rng.range(0, (cw - sw) / 2) * 2;
            let oy = rng.range(0, (ch - sh) / 2) * 2;
            let blend = if rng.chance(50) {
                BlendMethod::AlphaBlend
            } else {
                BlendMethod::Overwrite
            };
            let dispose = if rng.chance(50) {
                DisposeMethod::Background
            } else {
                DisposeMethod::None
            };
            enc.add_frame_advanced(
                &sub,
                PixelLayout::Rgba8,
                sw,
                sh,
                ox,
                oy,
                i as u32 * 100,
                &cfg,
                dispose,
                blend,
            )
            .ok()?;
        } else {
            enc.add_frame(f, PixelLayout::Rgba8, i as u32 * 100, &cfg)
                .ok()?;
        }
    }
    enc.finalize(100).ok().map(|b| (format!("zen_{tag}"), b))
}

/// `--detail FILE`: per-frame / per-channel breakdown of an animated mismatch.
fn detail(path: &Path) {
    let data = std::fs::read(path).unwrap();
    let mut zd = AnimationDecoder::new(&data).unwrap();
    let zf = zd.decode_all().unwrap();
    let mut ld = webpx::AnimationDecoder::new(&data).unwrap();
    let lf = ld.decode_all().unwrap();
    println!("frames zen={} lib={}", zf.len(), lf.len());
    let demux = zenwebp::mux::WebPDemuxer::new(&data).unwrap();
    for (i, (z, l)) in zf.iter().zip(&lf).enumerate() {
        let fr = demux.frame(i as u32 + 1).unwrap();
        let z4: Vec<u8> = if z.data.len() == (z.width * z.height * 3) as usize {
            z.data
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect()
        } else {
            z.data.clone()
        };
        let mut cmax = [0u8; 4];
        let mut cn = [0u64; 4];
        let mut first = None;
        for (j, (a, b)) in z4.iter().zip(&l.data).enumerate() {
            let d = a.abs_diff(*b);
            if d > 0 {
                cmax[j % 4] = cmax[j % 4].max(d);
                cn[j % 4] += 1;
                if first.is_none() {
                    first = Some(j / 4);
                }
            }
        }
        print!(
            "frame {i}: off=({},{}) {}x{} lossy={} alpha={} blend={:?} dispose={:?} maxRGBA={cmax:?} n={cn:?}",
            fr.x_offset,
            fr.y_offset,
            fr.width,
            fr.height,
            fr.is_lossy,
            fr.has_alpha,
            fr.blend,
            fr.dispose
        );
        // DP_REF_PREFIX=<dir/name>: compare against <prefix>-<frame>.png too.
        if let Ok(prefix) = std::env::var("DP_REF_PREFIX")
            && let Some((r, rw, rh, _)) =
                load_png_rgba8(Path::new(&format!("{prefix}-{}.png", i + 1)))
        {
            let cnt = |a: &[u8]| a.iter().zip(&r).filter(|(x, y)| x != y).count();
            print!(
                " | ref {rw}x{rh}: zen-vs-ref diff={} lib-vs-ref diff={}",
                cnt(&z4),
                cnt(&l.data)
            );
        }
        if let Some(px) = first {
            let w = z.width as usize;
            println!(
                " first@({},{}) zen={:?} lib={:?}",
                px % w,
                px / w,
                &z4[px * 4..px * 4 + 4],
                &l.data[px * 4..px * 4 + 4]
            );
        } else {
            println!();
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn walk(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, ext, out);
        } else if p
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case(ext))
        {
            out.push(p);
        }
    }
}

fn main() {
    let mut files_dirs = Vec::new();
    let mut gen_dirs = Vec::new();
    let mut out_path = PathBuf::from("decode_parity.tsv");
    let mut mismatch_dir: Option<PathBuf> = None;
    let mut save_gen: Option<PathBuf> = None;
    let mut per_variant = 2u32;
    let mut gen_limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--files" => files_dirs.push(PathBuf::from(args.next().unwrap())),
            "--gen-src" => gen_dirs.push(PathBuf::from(args.next().unwrap())),
            "--out" => out_path = PathBuf::from(args.next().unwrap()),
            "--mismatch-dir" => mismatch_dir = Some(PathBuf::from(args.next().unwrap())),
            "--save-gen" => save_gen = Some(PathBuf::from(args.next().unwrap())),
            "--encodes-per-variant" => per_variant = args.next().unwrap().parse().unwrap(),
            "--detail" => return detail(Path::new(&args.next().unwrap())),
            "--gen-limit" => gen_limit = args.next().unwrap().parse().unwrap(),
            o => panic!("unknown arg {o}"),
        }
    }
    // Silence panic messages from catch_unwind; they are recorded as rows.
    std::panic::set_hook(Box::new(|_| {}));
    for d in [&mismatch_dir, &save_gen].into_iter().flatten() {
        std::fs::create_dir_all(d).unwrap();
    }

    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    let record = |name: &str, data: &[u8], rs: Vec<Row>| {
        let bad = rs.iter().any(|r| !matches!(r.status, "match" | "both_err"));
        if bad && let Some(d) = &mismatch_dir {
            std::fs::write(d.join(name.replace('/', "__")), data).ok();
        }
        rows.lock().unwrap().extend(rs);
    };

    // Phase 1: existing files.
    let mut files = Vec::new();
    for d in &files_dirs {
        walk(d, "webp", &mut files);
    }
    files.sort();
    files.dedup();
    eprintln!("phase 1: {} existing .webp files", files.len());
    files.par_iter().for_each(|p| {
        let Ok(data) = std::fs::read(p) else { return };
        let name = p.to_string_lossy().to_string();
        let rs = check_file(&name, &data);
        record(&name, &data, rs);
    });

    // Phase 2: generated.
    let mut srcs = Vec::new();
    for d in &gen_dirs {
        walk(d, "png", &mut srcs);
    }
    srcs.sort();
    srcs.truncate(gen_limit);
    eprintln!(
        "phase 2: {} PNG sources, {per_variant} encodes/variant/encoder",
        srcs.len()
    );
    let done = std::sync::atomic::AtomicUsize::new(0);
    srcs.par_iter().for_each(|p| {
        let Some((rgba, w, h, has_alpha)) = load_png_rgba8(p) else {
            return;
        };
        if w > 4096 || h > 4096 {
            return;
        }
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let mut rng = Rng(hash_str(&p.to_string_lossy()));
        let mut encoded: Vec<(String, Vec<u8>)> = Vec::new();
        for v in variants(&stem, &rgba, w, h, has_alpha, &mut rng) {
            for _ in 0..per_variant {
                for by_lib in [true, false] {
                    if let Some((t, b)) = encode_one(&v, &mut rng, by_lib) {
                        let who = if by_lib { "lib" } else { "zen" };
                        encoded.push((format!("gen/{who}_{}_{t}.webp", v.tag), b));
                    }
                }
            }
        }
        for by_lib in [true, false] {
            if let Some((t, b)) = encode_anim(&stem, &rgba, w, h, &mut rng, by_lib) {
                encoded.push((format!("gen/{t}.webp"), b));
            }
        }
        for (name, data) in &encoded {
            if let Some(d) = &save_gen {
                std::fs::write(d.join(name.trim_start_matches("gen/")), data).ok();
            }
            let rs = check_file(name, data);
            record(name, data, rs);
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_multiple_of(25) {
            eprintln!("  {n}/{} sources", srcs.len());
        }
    });

    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| (&a.file, a.mode).cmp(&(&b.file, b.mode)));
    let mut tsv = String::from("file\tmode\tstatus\tmax_diff\tdiff_bytes\ttotal_bytes\tnote\n");
    for r in &rows {
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.file, r.mode, r.status, r.max_diff, r.diff_bytes, r.total_bytes, r.note
        )
        .ok();
    }
    std::fs::write(&out_path, tsv).unwrap();

    // Summary: per (source-kind, mode) status counts.
    let mut summary: std::collections::BTreeMap<(String, &str, &str), usize> = Default::default();
    for r in &rows {
        let src = if r.file.starts_with("gen/lib_") {
            "gen-by-libwebp"
        } else if r.file.starts_with("gen/zen_") {
            "gen-by-zenwebp"
        } else {
            "existing"
        };
        *summary
            .entry((src.to_string(), r.mode, r.status))
            .or_default() += 1;
    }
    let files_n: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.file.as_str()).collect();
    println!("files: {}  comparisons: {}", files_n.len(), rows.len());
    for ((src, mode, status), n) in &summary {
        println!("{src:16} {mode:13} {status:10} {n}");
    }
    let bad: Vec<&Row> = rows
        .iter()
        .filter(|r| !matches!(r.status, "match" | "both_err"))
        .collect();
    println!("NON-MATCH: {}", bad.len());
    for r in bad.iter().take(60) {
        println!(
            "  {} [{}] {} max={} n={}/{} {}",
            r.file, r.mode, r.status, r.max_diff, r.diff_bytes, r.total_bytes, r.note
        );
    }
}
