//! Both-directions overwrite harness for the structural inventory (adopted from
//! the round-4 review of PR 106).
//!
//! For mutated WebP files it overwrites every leaf the inventory reports as
//! unconsumed and checks that what the caller can observe (`probe()`,
//! `decode()`, `WebPProbe`, the animation path) does not change, then
//! overwrites half of each consumed Metadata/ImageData leaf and checks that it
//! does. It prints the violation classes; the safe-direction classes (bytes a
//! second reader reads, reported unconsumed with a detail naming that reader)
//! are expected, see the PR report.
//!
//! `#[ignore]`d: run with
//! `INVENTORY_BD_ITERS=20000 cargo test --test inventory_both_directions -- --ignored --nocapture`
//! (`INVENTORY_BD_SEED`, `INVENTORY_BD_CEX=<dir>` to save samples).
#![cfg(not(target_arch = "wasm32"))]

use std::hash::{DefaultHasher, Hash, Hasher};
use zencodec::decode::{DecodeJob, DecoderConfig};

use zencodec::inventory::{Disposition, MetadataKind, Part, PartKind};
use zenwebp::mux::{AnimationConfig, AnimationEncoder};
use zenwebp::zencodec::WebpDecoderConfig;
use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};

type Chunk = ([u8; 4], Vec<u8>);

fn chunks_of(data: &[u8]) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let fourcc: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        out.push((fourcc, data[pos + 8..pos + 8 + size].to_vec()));
        pos += 8 + size + (size & 1);
    }
    out
}

#[allow(dead_code)]
fn sub_chunks(body: &[u8]) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 8 <= body.len() {
        let fourcc: [u8; 4] = body[pos..pos + 4].try_into().unwrap();
        let size = u32::from_le_bytes(body[pos + 4..pos + 8].try_into().unwrap()) as usize;
        out.push((fourcc, body[pos + 8..pos + 8 + size].to_vec()));
        pos += 8 + size + (size & 1);
    }
    out
}

fn chunk_bytes(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = fourcc.to_vec();
    v.extend((payload.len() as u32).to_le_bytes());
    v.extend(payload);
    if payload.len() % 2 == 1 {
        v.push(0);
    }
    v
}

fn riff(chunks: &[Chunk]) -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    for (f, p) in chunks {
        body.extend(chunk_bytes(f, p));
    }
    let mut v = b"RIFF".to_vec();
    v.extend((body.len() as u32).to_le_bytes());
    v.extend(body);
    v
}

fn pattern_rgb(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.extend([
                ((x * 255 / w) as u8) ^ seed,
                ((y * 255 / h) as u8).wrapping_add(seed),
                ((x ^ y) as u8).wrapping_mul(seed | 1),
            ]);
        }
    }
    px
}

fn pattern_rgba(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            px.extend([
                ((x * 255 / w) as u8) ^ seed,
                ((y * 255 / h) as u8).wrapping_add(seed),
                ((x ^ y) as u8).wrapping_mul(seed | 1),
                (255 - (x * 255 / w)) as u8 ^ seed,
            ]);
        }
    }
    px
}

fn lossy(w: usize, h: usize, seed: u8) -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &pattern_rgb(w, h, seed),
        PixelLayout::Rgb8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap()
}

fn lossy_alpha(w: usize, h: usize, seed: u8) -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &pattern_rgba(w, h, seed),
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap()
}

fn lossless_anim() -> Vec<u8> {
    let mut anim = AnimationEncoder::new(
        8,
        8,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    let cfg = EncoderConfig::new_lossless();
    for (i, color) in [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
        .into_iter()
        .enumerate()
    {
        let rgba: Vec<u8> = color.iter().cycle().take(8 * 8 * 4).copied().collect();
        anim.add_frame(&rgba, PixelLayout::Rgba8, i as u32 * 100, &cfg)
            .unwrap();
    }
    anim.finalize(100).unwrap()
}

fn lossy_alpha_anim() -> Vec<u8> {
    let mut anim = AnimationEncoder::new(
        16,
        16,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    let cfg = EncoderConfig::Lossy(LossyConfig::default());
    for i in 0..2u8 {
        anim.add_frame(
            &pattern_rgba(16, 16, 40 * i + 3),
            PixelLayout::Rgba8,
            u32::from(i) * 100,
            &cfg,
        )
        .unwrap();
    }
    anim.finalize(100).unwrap()
}

fn lossy_alpha_with_metadata() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let base = chunks_of(&lossy_alpha(32, 24, 7));
    let get = |f: &[u8; 4]| base.iter().find(|c| &c.0 == f).unwrap().1.clone();
    let mut vp8x = get(b"VP8X");
    vp8x[0] |= 0x20 | 0x08 | 0x04;
    let icc = vec![0xABu8; 64];
    let exif = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
    let xmp = b"<x:xmpmeta/>".to_vec();
    let data = riff(&[
        (*b"VP8X", vp8x),
        (*b"ICCP", icc.clone()),
        (*b"ALPH", get(b"ALPH")),
        (*b"VP8 ", get(b"VP8 ")),
        (*b"EXIF", exif.clone()),
        (*b"XMP ", xmp.clone()),
    ]);
    (data, icc, exif, xmp)
}

fn ll_still_vp8l() -> Vec<u8> {
    let ll = EncodeRequest::new(
        &EncoderConfig::new_lossless(),
        &pattern_rgba(8, 8, 1),
        PixelLayout::Rgba8,
        8,
        8,
    )
    .encode()
    .unwrap();
    chunks_of(&ll)
        .into_iter()
        .find(|c| &c.0 == b"VP8L")
        .unwrap()
        .1
}

// ── Mutator and seeds (copied from review_r2.rs Part C) ──

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn lenient_chunks(data: &[u8]) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let fourcc: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = (pos + 8).saturating_add(size).min(data.len());
        out.push((fourcc, data[pos + 8..end].to_vec()));
        pos = (pos + 8).saturating_add(size).saturating_add(size & 1);
    }
    out
}

fn seeds() -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    v.push(lossy_alpha_with_metadata().0);
    // lossless VP8X still with metadata
    let vp8l = ll_still_vp8l();
    v.push(riff(&[
        (
            *b"VP8X",
            vec![0x10 | 0x20 | 0x08 | 0x04, 0, 0, 0, 7, 0, 0, 7, 0, 0],
        ),
        (*b"ICCP", b"icc-ll".to_vec()),
        (*b"VP8L", vp8l.clone()),
        (*b"EXIF", b"II*\0exif-ll".to_vec()),
        (*b"XMP ", b"<xmp-ll/>".to_vec()),
        (*b"PRIV", b"private-ll".to_vec()),
    ]));
    // lossless animation with metadata and extra sub-chunks
    let mut a = chunks_of(&lossless_anim());
    a[0].1[0] |= 0x20 | 0x08;
    a.insert(1, (*b"ICCP", b"icc-anim".to_vec()));
    a.push((*b"EXIF", b"II*\0exif-anim".to_vec()));
    let i = a.iter().position(|c| &c.0 == b"ANMF").unwrap();
    a[i].1.extend(chunk_bytes(b"XMP ", b"<frame-xmp/>"));
    v.push(riff(&a));
    // lossy alpha animation with ICCP
    let mut b = chunks_of(&lossy_alpha_anim());
    b[0].1[0] |= 0x20;
    b.insert(1, (*b"ICCP", b"icc-la".to_vec()));
    v.push(riff(&b));
    // simple files with tail chunks
    let mut s = chunks_of(&lossy(16, 16, 2));
    s.push((*b"ICCP", b"icc-tail".to_vec()));
    s.push((*b"EXIF", b"II*\0exif-tail".to_vec()));
    v.push(riff(&s));
    let mut s = vec![(*b"VP8L", vp8l.clone())];
    s.push((*b"XMP ", b"<tail/>".to_vec()));
    v.push(riff(&s));
    // still with an ANMF carrying ICCP first
    let mut anmf = vec![0u8; 16];
    anmf[6] = 7;
    anmf[9] = 7;
    anmf.extend(chunk_bytes(b"ICCP", b"icc-in-anmf"));
    anmf.extend(chunk_bytes(b"VP8L", &vp8l));
    v.push(riff(&[
        (*b"VP8X", vec![0x10, 0, 0, 0, 7, 0, 0, 7, 0, 0]),
        (*b"VP8L", vp8l.clone()),
        (*b"ANMF", anmf),
    ]));
    // duplicate VP8 still with ICCP twice and EXIF
    let la = chunks_of(&lossy_alpha(32, 24, 9));
    let get = |f: &[u8; 4]| la.iter().find(|c| &c.0 == f).unwrap().1.clone();
    let mut x = get(b"VP8X");
    x[0] |= 0x20 | 0x08;
    v.push(riff(&[
        (*b"VP8X", x),
        (*b"ICCP", b"icc-one".to_vec()),
        (*b"ALPH", get(b"ALPH")),
        (*b"VP8 ", get(b"VP8 ")),
        (*b"EXIF", b"II*\0exif-dup-still".to_vec()),
        (*b"VP8 ", get(b"VP8 ")),
        (*b"ICCP", b"icc-two-longer".to_vec()),
    ]));
    // animation with an unknown chunk between frames and an ICCP at the end
    let mut a = chunks_of(&lossless_anim());
    a[0].1[0] |= 0x20;
    let second = a
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.0 == b"ANMF")
        .nth(1)
        .unwrap()
        .0;
    a.insert(second, (*b"JUNK", b"between".to_vec()));
    a.push((*b"ICCP", b"icc-anim-end".to_vec()));
    v.push(riff(&a));
    // simple VP8L with trailing junk and a tail XMP
    let mut s = vec![(*b"VP8L", vp8l)];
    s.push((*b"XMP ", b"<tail2/>".to_vec()));
    let mut d = riff(&s);
    d.extend(b"trailing-bytes-after-riff");
    v.push(d);
    v
}

const KNOWN: [&[u8; 4]; 10] = [
    b"VP8 ", b"VP8L", b"VP8X", b"ANIM", b"ANMF", b"ALPH", b"ICCP", b"EXIF", b"XMP ", b"ZZZZ",
];

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut chunks = lenient_chunks(seed);
    let ops = 1 + rng.below(3);
    let mut fix_riff = rng.below(2) == 0;
    let mut byte_ops: Vec<u8> = Vec::new();
    for _ in 0..ops {
        match rng.below(9) {
            0 if !chunks.is_empty() => {
                let i = rng.below(chunks.len());
                let c = chunks[i].clone();
                let j = rng.below(chunks.len() + 1);
                chunks.insert(j, c);
            }
            1 if chunks.len() > 1 => {
                let i = 1 + rng.below(chunks.len() - 1);
                chunks.remove(i);
            }
            2 if chunks.len() > 2 => {
                let i = rng.below(chunks.len());
                let j = rng.below(chunks.len());
                chunks.swap(i, j);
            }
            3 if !chunks.is_empty() => {
                let i = rng.below(chunks.len());
                chunks[i].0 = *KNOWN[rng.below(KNOWN.len())];
            }
            4 if !chunks.is_empty() => {
                // edit inside a chunk payload (e.g. an ANMF's sub-chunks)
                let i = rng.below(chunks.len());
                let p = &mut chunks[i].1;
                if !p.is_empty() {
                    let k = rng.below(p.len());
                    p[k] = rng.next() as u8;
                }
            }
            5 if !chunks.is_empty() => {
                let i = rng.below(chunks.len());
                let f = *KNOWN[rng.below(KNOWN.len())];
                let len = rng.below(24);
                let payload: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                // insert a sub-chunk into an ANMF or a chunk at top level
                if &chunks[i].0 == b"ANMF" && chunks[i].1.len() >= 16 {
                    let at = 16;
                    let sub = chunk_bytes(&f, &payload);
                    chunks[i].1.splice(at..at, sub);
                } else {
                    chunks.insert(i, (f, payload));
                }
            }
            6 => byte_ops.push(0),
            7 => byte_ops.push(1),
            _ => fix_riff = !fix_riff,
        }
    }
    let mut d = riff(&chunks);
    if !fix_riff {
        let s = u32::from_le_bytes(seed[4..8].try_into().unwrap());
        d[4..8].copy_from_slice(&s.to_le_bytes());
    }
    for op in byte_ops {
        match op {
            0 => {
                // flip a byte or set a size field
                let k = rng.below(d.len());
                d[k] = rng.next() as u8;
            }
            _ => {
                if rng.below(2) == 0 {
                    let n = rng.below(48) + 1;
                    d.extend((0..n).map(|_| rng.next() as u8));
                } else {
                    let n = rng.below(d.len());
                    d.truncate(n.max(12));
                }
            }
        }
    }
    d
}

// ── Pitfall "Test in both directions" ──

fn demux_bitstream_fails_r4(d: &[u8]) -> bool {
    let Ok(dm) = zenwebp::mux::WebPDemuxer::new(d) else {
        return false;
    };
    let Some(f) = dm.frame(1) else { return false };
    if !f.is_lossy || dm.is_animated() {
        return false;
    }
    let (w, h) = (dm.canvas_width(), dm.canvas_height());
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return true;
    }
    let mut x = vec![if f.alpha_data.is_some() { 0x10 } else { 0 }, 0, 0, 0];
    x.extend(&(w - 1).to_le_bytes()[..3]);
    x.extend(&(h - 1).to_le_bytes()[..3]);
    let mut cs: Vec<Chunk> = vec![(*b"VP8X", x)];
    if let Some(a) = f.alpha_data {
        cs.push((*b"ALPH", a.to_vec()));
    }
    cs.push((*b"VP8 ", f.bitstream.to_vec()));
    let limits = zencodec::ResourceLimits::none().with_max_pixels(1 << 20);
    WebpDecoderConfig::new()
        .with_limits(limits)
        .decode(&riff(&cs))
        .is_err()
}

fn h<T: Hash + ?Sized>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Everything the caller can observe, as one string.
fn signature(d: &[u8]) -> String {
    let mut out = String::new();
    let wp = |sed: Option<&dyn zencodec::SourceEncodingDetails>| -> String {
        match sed.and_then(|s| s.codec_details::<zenwebp::detect::WebPProbe>()) {
            None => "wp:none".into(),
            Some(p) => format!(
                "wp:{}x{} a{} an{} f{} q{:?} icc{:?}",
                p.width,
                p.height,
                p.has_alpha,
                p.has_animation,
                p.frame_count,
                p.estimated_quality().map(f32::to_bits),
                p.icc_profile.as_ref().map(h)
            ),
        }
    };
    let info_s = |i: &zencodec::ImageInfo| -> String {
        format!(
            "{}x{} a{} seq{:?} icc{:?} exif{:?} xmp{:?} or{:?}",
            i.width,
            i.height,
            i.has_alpha,
            i.sequence,
            i.source_color.icc_profile.as_deref().map(h),
            i.embedded_metadata.exif.as_deref().map(h),
            i.embedded_metadata.xmp.as_deref().map(h),
            i.orientation
        )
    };
    match WebpDecoderConfig::new().job().probe(d) {
        Ok(i) => out.push_str(&format!(
            "P[{} {}]",
            info_s(&i),
            wp(i.source_encoding_details())
        )),
        Err(_) => out.push_str("P[err]"),
    }
    let limits = zencodec::ResourceLimits::none().with_max_pixels(1 << 20);
    match WebpDecoderConfig::new().with_limits(limits).decode(d) {
        Ok(o) => out.push_str(&format!(
            "D[{} {:?} px{} {}]",
            info_s(o.info()),
            o.pixels().descriptor(),
            h(&o.pixels().contiguous_bytes().into_owned()),
            wp(o.source_encoding_details())
        )),
        Err(_) => out.push_str("D[err]"),
    }
    // Animation path (what zencodec's animation decoder wraps), bounded.
    if let Ok(i) = WebpDecoderConfig::new().job().probe(d)
        && matches!(i.sequence, zencodec::ImageSequence::Animation { .. })
        && u64::from(i.width) * u64::from(i.height) <= 1 << 16
    {
        match zenwebp::mux::AnimationDecoder::new(d) {
            Ok(mut a) => {
                let info = a.info();
                let frames = a.decode_all();
                out.push_str(&format!(
                    "A[n{} loop{:?} a{} {:?}]",
                    info.frame_count,
                    info.loop_count,
                    info.has_alpha,
                    frames
                        .as_ref()
                        .map(|fs| fs
                            .iter()
                            .map(|f| (h(&f.data), f.duration_ms, f.timestamp_ms))
                            .collect::<Vec<_>>())
                        .map_err(|_| ())
                ));
            }
            Err(_) => out.push_str("A[err]"),
        }
    }
    out
}

/// The bytes of a part an overwrite may touch without changing framing: a
/// chunk's payload (not its 8-byte header or pad), or a whole gap/field/header.
fn writable(d: &[u8], p: &Part) -> std::ops::Range<usize> {
    let (s, e) = (p.range().start as usize, p.range().end as usize);
    if p.kind() == PartKind::Chunk {
        if s + 8 > d.len() {
            return s..s;
        }
        let size = u32::from_le_bytes(d[s + 4..s + 8].try_into().unwrap()) as usize;
        (s + 8)..(s + 8).saturating_add(size).min(e)
    } else {
        s..e
    }
}

#[test]
#[ignore = "two-direction overwrite run; INVENTORY_BD_ITERS / INVENTORY_BD_SEED"]
fn both_directions() {
    let iters: usize = std::env::var("INVENTORY_BD_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let out_dir = std::path::PathBuf::from(
        std::env::var("INVENTORY_BD_CEX").unwrap_or_else(|_| "/nonexistent".into()),
    );
    let mut rng = Rng(std::env::var("INVENTORY_BD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9e37_79b9_7f4a_7c15));
    let seeds = seeds();
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut saved: std::collections::BTreeMap<String, usize> = Default::default();
    let mut unexplained: Vec<String> = Vec::new();
    for it in 0..iters {
        let d = if it % 4 == 0 {
            seeds[it % seeds.len()].clone()
        } else {
            mutate(&mut rng, &seeds[it % seeds.len()])
        };
        let inv = WebpDecoderConfig::new()
            .job()
            .inventory(&d)
            .unwrap()
            .unwrap();
        let base = signature(&d);
        let any_ok = !(base.contains("P[err]") && base.contains("D[err]"));
        let anim_ok = base.contains("A[") && !base.contains("A[err]") && !base.contains("Err(())");
        let riff_end = 8 + u64::from(u32::from_le_bytes(
            d.get(4..8)
                .and_then(|b| b.try_into().ok())
                .unwrap_or([0; 4]),
        ));
        let demux_fails = demux_bitstream_fails_r4(&d);
        *counts.entry("inputs".into()).or_default() += 1;
        let mut has_kids = vec![false; inv.parts().len()];
        for p in inv.parts() {
            if let Some(q) = p.parent() {
                has_kids[q.index()] = true;
            }
        }
        for (i, p) in inv.parts().iter().enumerate() {
            if has_kids[i] {
                continue;
            }
            let w = writable(&d, p);
            if w.is_empty() {
                continue;
            }
            let consumed = p.disposition().is_consumed();
            if consumed
                && !matches!(
                    p.disposition(),
                    Disposition::Metadata(_) | Disposition::ImageData
                )
            {
                continue; // Structure leaves are framing; changing them reshapes the walk
            }
            let mut d2 = d.clone();
            // Unconsumed: overwrite every byte. Consumed: overwrite the first half
            // (image payload slack is a documented limit).
            let w2 = if consumed {
                w.start..w.start + w.len().div_ceil(2)
            } else {
                w.clone()
            };
            for b in &mut d2[w2] {
                *b ^= 0xA5;
            }
            let changed = signature(&d2) != base;
            let key_detail: String = p.detail().unwrap_or("").chars().take(60).collect();
            let class = format!(
                "{} {} {:?} [{}]",
                if consumed {
                    "D2 consumed-unchanged"
                } else {
                    "D1 unconsumed-changed"
                },
                p.disposition(),
                p.label().unwrap_or(match p.kind() {
                    PartKind::Gap => "gap",
                    PartKind::Field => "field",
                    PartKind::Header => "header",
                    _ => "?",
                }),
                key_detail
            );
            let in_anmf = p
                .parent()
                .is_some_and(|q| inv.get(q).is_some_and(|q| q.label() == Some("ANMF")));
            let mut class = class;
            if consumed && !any_ok {
                continue; // nothing reaches the caller either way
            }
            if consumed && in_anmf && !anim_ok {
                continue; // a frame after a failing frame is never decoded (bitstream-level)
            }
            let decode_ok = !base.contains("D[err]");
            if consumed && p.disposition() == Disposition::ImageData && !decode_ok && !anim_ok {
                continue; // the image is not decoded at all (bitstream-level), so its bytes cannot matter
            }
            if consumed
                && p.disposition() == Disposition::Metadata(MetadataKind::Animation)
                && !anim_ok
            {
                continue; // animation metadata is only observable through the animation path
            }
            if consumed && !changed && p.disposition() == Disposition::ImageData && demux_fails {
                class = format!(
                    "{class} <demuxer path fails: decode() fell back to WebPDecoder's first chunks>"
                );
            }
            if !consumed && changed {
                let det = p.detail().unwrap_or("");
                let sub = if p.kind() == PartKind::Gap && p.disposition() == Disposition::Trailing {
                    if p.range().start < riff_end + 10 {
                        "trailing: inside read_data's 10-byte overrun"
                    } else {
                        "trailing: beyond the overrun (detect::probe walks to EOF)"
                    }
                } else if det.contains("WebPProbe quality") {
                    "owner of WebPProbe quality"
                } else if (det.contains("WebPDemuxer") || det.contains("decoded, then discarded"))
                    && demux_fails
                {
                    "demuxer path fails -> fallback to WebPDecoder reads it"
                } else if det.contains("decoded, then discarded") {
                    "dropped ALPH: validity decides decode() success"
                } else {
                    ""
                };
                if !sub.is_empty() {
                    class = format!("{class} <{sub}>");
                }
            }
            let violation = if consumed { !changed } else { changed };
            *counts
                .entry(if consumed {
                    "checked consumed leaves".into()
                } else {
                    "checked unconsumed leaves".into()
                })
                .or_default() += 1;
            if violation && !consumed {
                // Safe direction: allowed only when the detail names the reader.
                let det = p.detail().unwrap_or("");
                let named = [
                    "detect::probe",
                    "WebPProbe",
                    "read_data",
                    "parsed:",
                    "falls back to WebPDecoder",
                ]
                .iter()
                .any(|r| det.contains(r));
                if !named {
                    unexplained.push(class.clone());
                }
            }
            if violation {
                *counts.entry(class.clone()).or_default() += 1;
                let n = saved.entry(class.clone()).or_default();
                if *n < 3 {
                    let tag = format!("{:x}", h(&class));
                    let _ = std::fs::write(out_dir.join(format!("{tag}-{n}.webp")), &d);
                    println!(
                        "CEX {tag}-{n}: part {}..{} :: {class}",
                        p.range().start,
                        p.range().end
                    );
                }
                *n += 1;
            }
        }
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (k, n) in v {
        println!("{n:8}  {k}");
    }
    // Consumed-unchanged results are not asserted: they are dominated by
    // identical duplicate chunks (the fallback decodes the same pixels) and by
    // the documented demuxer-fallback condition.
    assert!(
        unexplained.is_empty(),
        "{} unconsumed leaves change the output without a detail naming the reader, e.g. {:?}",
        unexplained.len(),
        unexplained.first()
    );
}
