//! Structural inventory (`DecodeJob::inventory`): coverage, dispositions that
//! match what the decoder does, and the container quirks the walk documents.
//!
//! Fixtures are built here from encoder output and hand-assembled RIFF
//! chunks; nothing binary is committed.
#![cfg(not(target_arch = "wasm32"))]

use std::fmt::Write as _;

use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::inventory::{Disposition, Inventory, MetadataKind, PartKind, PartTag};
use zenwebp::mux::{AnimationConfig, AnimationEncoder, WebPMux};
use zenwebp::zencodec::WebpDecoderConfig;
use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};

// ── RIFF helpers ────────────────────────────────────────────────────────────

type Chunk = ([u8; 4], Vec<u8>);

/// Split a RIFF/WEBP file into its top-level chunks (test-side parser: assumes
/// a well-formed file).
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

fn chunk_bytes(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = fourcc.to_vec();
    v.extend((payload.len() as u32).to_le_bytes());
    v.extend(payload);
    if payload.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// RIFF/WEBP file from chunks, with the correct RIFF size.
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

fn set_riff_size(data: &mut [u8], size: u32) {
    data[4..8].copy_from_slice(&size.to_le_bytes());
}

// ── Encoder fixtures ────────────────────────────────────────────────────────

fn rgb_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.extend([
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x ^ y) & 0xff) as u8,
            ]);
        }
    }
    px
}

fn rgba_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            px.extend([
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x ^ y) & 0xff) as u8,
                (255 - (x * 255 / w)) as u8,
            ]);
        }
    }
    px
}

fn lossy_rgb() -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &rgb_gradient(32, 24),
        PixelLayout::Rgb8,
        32,
        24,
    )
    .encode()
    .unwrap()
}

fn lossless_rgb() -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::new_lossless(),
        &rgb_gradient(32, 24),
        PixelLayout::Rgb8,
        32,
        24,
    )
    .encode()
    .unwrap()
}

fn lossy_alpha() -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &rgba_gradient(32, 24),
        PixelLayout::Rgba8,
        32,
        24,
    )
    .encode()
    .unwrap()
}

fn animation() -> Vec<u8> {
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

// ── Inspection helpers ──────────────────────────────────────────────────────

fn inventory_of(data: &[u8]) -> Inventory {
    let job = WebpDecoderConfig::new().job();
    let inv = job.inventory(data).unwrap().expect("inventory declared");
    inv.validate()
        .unwrap_or_else(|e| panic!("invalid inventory: {e}\n{inv}"));
    inv
}

/// One line per part: depth-indented `kind tag range disposition label [detail]`.
fn render(inv: &Inventory) -> String {
    let mut depth = vec![0usize; inv.parts().len()];
    let mut out = String::new();
    for (i, p) in inv.parts().iter().enumerate() {
        if let Some(parent) = p.parent {
            depth[i] = depth[parent.index()] + 1;
        }
    }
    // File order, parents before children.
    let mut order: Vec<usize> = (0..inv.parts().len()).collect();
    order.sort_by_key(|&i| {
        let mut chain = vec![];
        let mut cur = Some(i);
        while let Some(c) = cur {
            let p = &inv.parts()[c];
            chain.push((p.range.start, p.range.end));
            cur = p.parent.map(|x| x.index());
        }
        chain.reverse();
        chain
    });
    for i in order {
        let p = &inv.parts()[i];
        let _ = write!(
            out,
            "{}{:?} {} {}..{} {}",
            "  ".repeat(depth[i]),
            p.kind,
            match &p.tag {
                PartTag::FourCc(f) => format!("{:?}", String::from_utf8_lossy(f)),
                other => format!("{other:?}"),
            },
            p.range.start,
            p.range.end,
            p.disposition
        );
        if let Some(l) = &p.label {
            let _ = write!(out, " label={l:?}");
        }
        if let Some(d) = &p.detail {
            let _ = write!(out, " detail={d:?}");
        }
        out.push('\n');
    }
    out
}

fn top_level_chunks(inv: &Inventory) -> Vec<(String, Disposition)> {
    inv.children(None)
        .into_iter()
        .map(|id| inv.get(id).unwrap())
        .filter(|p| p.kind == PartKind::Chunk)
        .map(|p| (p.label.as_deref().unwrap_or("").to_string(), p.disposition))
        .collect()
}

fn find<'a>(inv: &'a Inventory, label: &str) -> Vec<&'a zencodec::inventory::Part> {
    inv.parts()
        .iter()
        .filter(|p| p.label.as_deref() == Some(label))
        .collect()
}

// ── Conformance: encoder output, corpus ─────────────────────────────────────

#[test]
fn check_inventory_on_encoder_output() {
    for (name, data) in [
        ("lossy", lossy_rgb()),
        ("lossless", lossless_rgb()),
        ("lossy+alpha", lossy_alpha()),
        ("animation", animation()),
    ] {
        zencodec_testkit::check_inventory(WebpDecoderConfig::new(), &data)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn capability_is_declared() {
    // `check_all` is not usable for zenwebp yet: its cross-path check streams a
    // lossless file and its pixel round trip is exact-only (both predate the
    // inventory work). The inventory checks run directly on encoder output in
    // `check_inventory_on_encoder_output`.
    assert!(WebpDecoderConfig::capabilities().inventory());
}

#[test]
fn corpus_webp_conformance_valid() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let dir = corpus
        .get("webp-conformance/valid")
        .expect("webp-conformance/valid missing from codec-corpus");
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("webp") {
            continue;
        }
        let data = std::fs::read(&path).unwrap();
        zencodec_testkit::check_inventory(WebpDecoderConfig::new(), &data)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        n += 1;
    }
    assert!(n >= 200, "expected the 225-file conformance set, saw {n}");
}

// ── Dispositions follow the decoder ─────────────────────────────────────────

#[test]
fn simple_lossy_and_lossless_are_one_image_chunk() {
    for data in [lossy_rgb(), lossless_rgb()] {
        let inv = inventory_of(&data);
        let top = top_level_chunks(&inv);
        assert_eq!(top.len(), 1, "{}", inv);
        assert_eq!(top[0].1, Disposition::ImageData);
        assert_eq!(inv.parts()[0].kind, PartKind::Header);
        assert_eq!(inv.parts()[0].range, 0..12);
    }
}

/// Extended still with every chunk type: ICCP, ALPH, VP8, EXIF (odd), XMP, an
/// unknown chunk, a duplicate EXIF and junk after the RIFF end.
fn everything_still() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut data, icc, exif, xmp) = everything_still_no_junk();
    data.extend(b"junk");
    (data, icc, exif, xmp)
}

fn everything_still_no_junk() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let base = chunks_of(&lossy_alpha());
    let vp8x = base.iter().find(|c| &c.0 == b"VP8X").unwrap().1.clone();
    let alph = base.iter().find(|c| &c.0 == b"ALPH").unwrap().1.clone();
    let vp8 = base.iter().find(|c| &c.0 == b"VP8 ").unwrap().1.clone();
    let mut vp8x = vp8x;
    vp8x[0] |= 0x20 | 0x10 | 0x08 | 0x04; // ICC, alpha, EXIF, XMP
    // Smallest legal TIFF header with orientation 6 (odd total length: 0x2b).
    let exif = {
        let mut e = b"II*\0\x08\0\0\0".to_vec();
        e.extend(1u16.to_le_bytes());
        e.extend([0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0]);
        e.extend([0, 0, 0, 0]);
        e.push(0x7f);
        e
    };
    let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>".to_vec();
    let icc = vec![0xAB; 130];
    let chunks: Vec<Chunk> = vec![
        (*b"VP8X", vp8x),
        (*b"ICCP", icc.clone()),
        (*b"ALPH", alph),
        (*b"VP8 ", vp8),
        (*b"EXIF", exif.clone()),
        (*b"XMP ", xmp.clone()),
        (*b"ZEN1", b"serial=12345 secret".to_vec()),
        (*b"EXIF", b"II*\0second-exif-ignored".to_vec()),
    ];
    (riff(&chunks), icc, exif, xmp)
}

#[test]
fn extended_still_dispositions_match_decoder_output() {
    let (data, icc, exif, xmp) = everything_still();
    let inv = inventory_of(&data);
    let got = top_level_chunks(&inv);
    use Disposition::*;
    assert_eq!(
        got,
        vec![
            ("VP8X".to_string(), Structure),
            ("ICCP".to_string(), Metadata(MetadataKind::Icc)),
            ("ALPH".to_string(), ImageData),
            ("VP8 ".to_string(), ImageData),
            ("EXIF".to_string(), Metadata(MetadataKind::Exif)),
            ("XMP ".to_string(), Metadata(MetadataKind::Xmp)),
            ("ZEN1".to_string(), Unknown),
            ("EXIF".to_string(), Skipped),
        ],
        "{inv}"
    );
    // The bytes the inventory calls metadata are what the decoder reports.
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.source_color.icc_profile.as_deref(), Some(&icc[..]));
    assert_eq!(info.embedded_metadata.exif.as_deref(), Some(&exif[..]));
    assert_eq!(info.embedded_metadata.xmp.as_deref(), Some(&xmp[..]));
    // Unknown chunk: label is the FourCC, detail shows the readable payload start.
    let zen = find(&inv, "ZEN1");
    assert_eq!(zen.len(), 1);
    assert!(zen[0].detail.as_deref().unwrap().contains("serial=12345"));
    // Trailing junk is the last part and is not consumed.
    let last = inv.children(None).pop().unwrap();
    let last = inv.get(last).unwrap();
    assert_eq!(last.disposition, Disposition::Trailing);
    assert_eq!(last.range.end, data.len() as u64);
    zencodec_testkit::check_inventory(WebpDecoderConfig::new(), &data).unwrap();
}

#[test]
fn metadata_flags_do_not_gate_the_decoders_metadata_reads() {
    // VP8X flags clear, EXIF chunk present: WebPDecoder still reports it
    // (`icc_profile`/`exif_metadata`/`xmp_metadata` ignore the flags), so the
    // inventory must say Metadata.
    let (mut data, _, exif, _) = everything_still();
    data[20] = 0x10; // alpha only
    let inv = inventory_of(&data);
    let exif_part = find(&inv, "EXIF")[0];
    assert_eq!(
        exif_part.disposition,
        Disposition::Metadata(MetadataKind::Exif)
    );
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.embedded_metadata.exif.as_deref(), Some(&exif[..]));
}

#[test]
fn alph_is_image_data_only_when_the_decoder_uses_it() {
    // Alpha flag clear: the decoder never reads ALPH for a lossy still.
    let (mut data, ..) = everything_still();
    data[20] &= !0x10;
    let inv = inventory_of(&data);
    // `decode()` still parses ALPH through the demuxer, then drops the channel.
    let alph = find(&inv, "ALPH")[0];
    assert_eq!(alph.disposition, Disposition::Dropped, "{inv}");
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert!(!info.has_alpha);
}

#[test]
fn chunks_after_a_simple_image_are_not_read() {
    let base = chunks_of(&lossless_rgb());
    let mut chunks = base.clone();
    chunks.push((*b"EXIF", b"MM\0*after-image".to_vec()));
    chunks.push((*b"Priv", b"camera-serial-ABC".to_vec()));
    let data = riff(&chunks);
    let inv = inventory_of(&data);
    let top = top_level_chunks(&inv);
    assert_eq!(
        top,
        vec![
            ("VP8L".to_string(), Disposition::ImageData),
            ("EXIF".to_string(), Disposition::Skipped),
            ("Priv".to_string(), Disposition::Unknown),
        ]
    );
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert!(
        info.embedded_metadata.exif.is_none(),
        "decoder must not report EXIF after a simple-format image"
    );
}

#[test]
fn animation_frames_header_and_subchunks() {
    let data = animation();
    let inv = inventory_of(&data);
    let anmf = find(&inv, "ANMF");
    assert_eq!(anmf.len(), 3);
    for a in &anmf {
        assert_eq!(a.disposition, Disposition::Structure);
        assert!(a.body.is_some());
    }
    let headers: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.kind == PartKind::Header && p.range.end - p.range.start == 16)
        .collect();
    assert_eq!(headers.len(), 3);
    assert!(
        headers
            .iter()
            .all(|h| h.disposition == Disposition::Metadata(MetadataKind::Animation))
    );
    let frames: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.parent.is_some() && p.label.as_deref() == Some("VP8L"))
        .collect();
    assert_eq!(frames.len(), 3);
    assert!(
        frames
            .iter()
            .all(|f| f.disposition == Disposition::ImageData)
    );
    // ANIM: background colour is native-only, loop count reaches zencodec.
    let anim = find(&inv, "ANIM")[0];
    assert_eq!(anim.body.as_ref().map(|b| b.end - b.start), Some(6));
    let anim_id = inv
        .children(None)
        .into_iter()
        .find(|&id| inv.get(id).unwrap().label.as_deref() == Some("ANIM"))
        .unwrap();
    let kids = inv.children(Some(anim_id));
    let kinds: Vec<_> = kids
        .iter()
        .map(|&k| inv.get(k).unwrap().disposition)
        .collect();
    assert_eq!(
        kinds,
        vec![
            Disposition::Dropped,
            Disposition::Metadata(MetadataKind::Animation)
        ]
    );
}

#[test]
fn later_anmf_subchunks_are_not_read() {
    // Frame 1 gets an extra unknown and an extra EXIF sub-chunk after its image.
    let data = animation();
    let mut chunks = chunks_of(&data);
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    chunks[i].1.extend(chunk_bytes(b"EXIF", b"MM\0*frame-exif"));
    chunks[i].1.extend(chunk_bytes(b"FPRV", b"frame-private"));
    let data = riff(&chunks);
    let inv = inventory_of(&data);
    let exif = find(&inv, "EXIF");
    assert_eq!(exif.len(), 1);
    assert_eq!(exif[0].disposition, Disposition::Skipped);
    assert!(exif[0].parent.is_some());
    let fprv = find(&inv, "FPRV");
    assert_eq!(fprv[0].disposition, Disposition::Unknown);
    zencodec_testkit::check_inventory(WebpDecoderConfig::new(), &data).unwrap();
}

// ── Container quirks (decoder behaviour the inventory mirrors) ──────────────

/// Decoder quirk: the VP8X chunk walk runs ten bytes past the RIFF end, so a
/// chunk header starting right after the declared RIFF end is still read. If
/// the decoder is fixed to stop at the RIFF end, this test and the walk's
/// `past_end` handling change in the same commit.
#[test]
fn current_behaviour_vp8x_walk_reads_a_chunk_past_the_riff_end() {
    let (data, ..) = everything_still();
    let chunks = chunks_of(&data[..data.len() - 4]); // drop the junk
    let exif = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
    let mut kept: Vec<Chunk> = chunks
        .into_iter()
        .filter(|c| !matches!(&c.0, b"EXIF" | b"XMP " | b"ZEN1"))
        .collect();
    kept[0].1[0] &= !0x0c; // no EXIF/XMP flags
    let mut data = riff(&kept);
    let riff_end = data.len();
    data.extend(chunk_bytes(b"EXIF", &exif)); // outside the RIFF size
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(
        info.embedded_metadata.exif.as_deref(),
        Some(&exif[..]),
        "decoder reads an EXIF chunk that starts at the RIFF end"
    );
    let inv = inventory_of(&data);
    let part = find(&inv, "EXIF")[0];
    assert_eq!(
        part.disposition,
        Disposition::Metadata(MetadataKind::Exif),
        "{inv}"
    );
    assert_eq!(part.range.start, riff_end as u64);
    assert!(
        part.detail
            .as_deref()
            .unwrap()
            .contains("past the RIFF end")
    );
}

/// Decoder quirk: `read_frame` advances by the unrounded ANMF size and expects
/// the next ANMF header there. An unknown chunk between ANMFs, or an ANMF with
/// a pad byte, stops decoding at that frame.
#[test]
fn current_behaviour_unknown_chunk_between_anmf_stops_the_frame_chain() {
    let data = animation();
    let mut chunks = chunks_of(&data);
    let second = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.0 == b"ANMF")
        .nth(1)
        .unwrap()
        .0;
    chunks.insert(second, (*b"JUNK", vec![1, 2, 3, 4]));
    let data = riff(&chunks);
    let all = zenwebp::mux::AnimationDecoder::new(&data).and_then(|mut d| d.decode_all());
    assert!(all.is_err(), "decoder is expected to fail at frame 2");
    let inv = inventory_of(&data);
    let anmf = find(&inv, "ANMF");
    assert_eq!(anmf.len(), 3);
    assert!(
        !anmf[0]
            .detail
            .as_deref()
            .unwrap_or("")
            .contains("unreachable")
    );
    for later in &anmf[1..] {
        assert!(
            later.detail.as_deref().unwrap().contains("unreachable"),
            "{inv}"
        );
    }
    // Unreachable frames: only the first sub-chunk's header (and the VP8L
    // alpha word) is read; the pixel payload is Skipped.
    let header_only = inv
        .parts()
        .iter()
        .filter(|p| p.parent.is_some() && p.label.as_deref() == Some("VP8L"))
        .filter(|p| {
            p.disposition == Disposition::Structure
                && p.detail
                    .as_deref()
                    .is_some_and(|d| d.contains("header read by read_data"))
        })
        .count();
    assert_eq!(header_only, 2, "{inv}");
    let skipped_payload = inv
        .parts()
        .iter()
        .filter(|p| p.kind == PartKind::Gap && p.disposition == Disposition::Skipped)
        .count();
    assert_eq!(skipped_payload, 2, "{inv}");
}

#[test]
fn mux_roundtrip_drops_unknown_and_duplicate_chunks() {
    // `WebPMux::from_data` keeps only chunk types it models, once each. The
    // inventory shows what a mux round trip would silently lose.
    let (data, ..) = everything_still();
    let data = &data[..data.len() - 4];
    let mux = WebPMux::from_data(data).unwrap();
    let out = mux.assemble().unwrap();
    let names: Vec<_> = chunks_of(&out).into_iter().map(|c| c.0).collect();
    assert!(
        !names.contains(b"ZEN1"),
        "unknown chunk should be dropped: {names:?}"
    );
    assert_eq!(names.iter().filter(|n| *n == b"EXIF").count(), 1);
}

// ── Malformed input never errors ────────────────────────────────────────────

#[test]
fn malformed_inputs_yield_valid_inventories() {
    let (good, ..) = everything_still();
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("three bytes", b"RIF".to_vec()),
        ("not riff", vec![0x42; 200]),
        ("riff only", b"RIFF\x04\0\0\0WEBP".to_vec()),
        ("riff size huge", {
            let mut d = good.clone();
            set_riff_size(&mut d, u32::MAX);
            d
        }),
        ("riff size zero", {
            let mut d = good.clone();
            set_riff_size(&mut d, 0);
            d
        }),
        ("chunk size huge", {
            let mut d = good.clone();
            d[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
            d
        }),
    ];
    // Non-zero pad byte on the odd-sized EXIF chunk.
    let mut padded = good.clone();
    let inv = inventory_of(&good);
    let exif = find(&inv, "EXIF")[0];
    let pad = exif.range.end as usize - 1;
    assert_eq!(padded[pad], 0, "fixture has a zero pad byte to flip");
    padded[pad] = 0x5a;
    cases.push(("non-zero pad", padded.clone()));
    for (name, d) in &cases {
        let inv = inventory_of(d);
        assert_eq!(inv.input_len(), d.len() as u64, "{name}");
    }
    let inv = inventory_of(&padded);
    let nz: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.disposition == Disposition::Padding)
        .collect();
    assert_eq!(nz.len(), 1, "{inv}");
    assert!(nz[0].detail.as_deref().unwrap().contains("0x5a"));
}

// ── Pinned part list ────────────────────────────────────────────────────────

/// Hand-assembled animated VP8X file with every chunk type the format defines,
/// an unknown chunk (with a non-zero pad byte), a duplicate, sub-chunks after
/// the frame image, and bytes after the RIFF end. Payloads are not valid
/// bitstreams: the walk never decodes.
fn pinned_fixture() -> Vec<u8> {
    let mut vp8x = vec![0x3e, 0, 0, 0];
    vp8x.extend([3, 0, 0, 3, 0, 0]); // canvas 4x4
    let mut anmf = vec![0u8; 16];
    anmf[6] = 3; // frame width - 1
    anmf[9] = 3;
    anmf[12] = 40; // duration
    anmf.extend(chunk_bytes(b"ALPH", &[1, 2, 3]));
    anmf.extend(chunk_bytes(b"VP8 ", &[9, 8, 7, 6]));
    anmf.extend(chunk_bytes(b"EXIF", &[0xEE, 0xEE]));
    anmf.extend(chunk_bytes(b"QQQQ", b"hi"));
    let mut data = riff(&[
        (*b"VP8X", vp8x),
        (*b"ICCP", vec![0xC0, 0xC1, 0xC2]),
        (*b"ANIM", vec![1, 2, 3, 4, 5, 0]),
        (*b"ANMF", anmf),
        (*b"VP8 ", vec![0xAA, 0xBB]),
        (*b"ZEN1", b"hello".to_vec()),
        (*b"EXIF", vec![0x4D, 0x4D, 0, 0x2A]),
        (*b"XMP ", b"<x/>".to_vec()),
        (*b"ICCP", vec![0xDD, 0xDD]),
    ]);
    let zen_pad = data.windows(4).position(|w| w == b"ZEN1").unwrap() + 8 + 5;
    assert_eq!(data[zen_pad], 0);
    data[zen_pad] = 0x41;
    data.extend(b"TAIL");
    data
}

#[test]
fn pinned_part_list() {
    let data = pinned_fixture();
    let inv = inventory_of(&data);
    let got = render(&inv);
    if let Ok(path) = std::env::var("BLESS_PINNED") {
        std::fs::write(path, &got).unwrap();
    }
    let expected = r####"Header "RIFF" 0..12 structure label="RIFF/WEBP" detail="riff_size=174; 4 bytes follow the declared RIFF end at 182"
Chunk "VP8X" 12..30 structure label="VP8X" detail="flags=0x3e (icc=1 alpha=1 exif=1 xmp=1 anim=1), canvas 4x4"
  Field Name("flags") 20..21 structure
  Field Name("reserved") 21..24 dropped detail="read into _reserved_bytes and discarded"
  Field Name("canvas") 24..30 structure
Chunk "ICCP" 30..42 metadata(icc) label="ICCP" detail="first occurrence"
Chunk "ANIM" 42..56 structure label="ANIM"
  Field Name("background_color") 50..54 dropped detail="native WebPDecoder::background_color_hint only; the zencodec path does not report it"
  Field Name("loop_count") 54..56 metadata(animation)
Chunk "ANMF" 56..124 structure label="ANMF"
  Header Name("frame header") 64..80 metadata(animation) detail="frame x/y/size, duration, blend and dispose flags"
  Chunk "ALPH" 80..92 image-data label="ALPH" detail="frame alpha plane; bytes after the stream's own end are not distinguished"
  Chunk "VP8 " 92..104 image-data label="VP8 " detail="frame image; bytes after the bitstream's own end are not distinguished"
  Chunk "EXIF" 104..114 skipped label="EXIF" detail="read_frame reads only the frame's image chunk(s)"
  Chunk "QQQQ" 114..124 unknown label="QQQQ" detail="unrecognised chunk type; payload starts \"hi\""
Chunk "VP8 " 124..134 skipped label="VP8 " detail="animated file: frames are read from ANMF; detect::probe fails on this chunk's frame header, so no WebPProbe is attached"
Chunk "ZEN1" 134..147 unknown label="ZEN1" detail="unrecognised chunk type; payload starts \"hello\""
Gap None 147..148 padding detail="non-zero RIFF pad byte 0x41"
Chunk "EXIF" 148..160 metadata(exif) label="EXIF" detail="first occurrence"
Chunk "XMP " 160..172 metadata(xmp) label="XMP " detail="first occurrence"
Chunk "ICCP" 172..182 skipped label="ICCP" detail="duplicate chunk"
Gap None 182..186 trailing
"####;
    assert_eq!(got, expected);
}

// ── More decoder quirks the walk mirrors ────────────────────────────────────

/// Decoder quirk: after the VP8X chunk walk's ten-byte overrun, 8 or more
/// bytes of data following the RIFF end are parsed as a chunk header whose
/// size is then seeked over, so the whole decode fails. The inventory itself
/// still reports the bytes as `Trailing`.
#[test]
fn current_behaviour_trailing_data_after_a_vp8x_file_fails_the_decode() {
    let (mut data, ..) = everything_still_no_junk();
    assert!(
        zenwebp::decoder::decode_rgba(&data).is_ok(),
        "fixture decodes as built"
    );
    data.extend(b"trailing junk, longer than a chunk header");
    assert!(
        zenwebp::decoder::decode_rgba(&data).is_err(),
        "decoder is expected to reject trailing data after a VP8X file"
    );
    let inv = inventory_of(&data);
    let last = inv.get(inv.children(None).pop().unwrap()).unwrap();
    assert_eq!(last.disposition, Disposition::Trailing);
    assert_eq!(last.range.end, data.len() as u64);
    // The same bytes after a simple-format file are harmless.
    let mut simple = lossless_rgb();
    simple.extend(b"trailing junk, longer than a chunk header");
    assert!(zenwebp::decoder::decode_rgba(&simple).is_ok());
}

/// Decoder quirk: `read_frame` advances by the unrounded ANMF size, so an ANMF
/// whose size is odd (pad byte after it) is followed by a read at the pad byte.
#[test]
fn current_behaviour_odd_sized_anmf_breaks_the_next_frame() {
    let base = chunks_of(&animation());
    let mut chunks: Vec<Chunk> = Vec::new();
    for (f, p) in &base {
        if f != b"ANMF" {
            chunks.push((*f, p.clone()));
            continue;
        }
        // Rebuild the frame with an unpadded, odd-sized VP8L sub-chunk.
        let mut sub = chunks_of_sub(&p[16..]);
        if sub.len().is_multiple_of(2) {
            sub.push(0);
        }
        let mut payload = p[..16].to_vec();
        payload.extend(b"VP8L");
        payload.extend((sub.len() as u32).to_le_bytes());
        payload.extend(&sub);
        assert_eq!(payload.len() % 2, 1);
        chunks.push((*f, payload));
    }
    let data = riff(&chunks);
    let all = zenwebp::mux::AnimationDecoder::new(&data).and_then(|mut d| d.decode_all());
    assert!(all.is_err(), "decoder is expected to fail at frame 2");
    let inv = inventory_of(&data);
    let anmf = find(&inv, "ANMF");
    assert!(
        !anmf[0]
            .detail
            .as_deref()
            .unwrap_or("")
            .contains("unreachable")
    );
    assert!(
        anmf[1].detail.as_deref().unwrap().contains("unreachable"),
        "{inv}"
    );
    assert!(
        anmf[2].detail.as_deref().unwrap().contains("unreachable"),
        "{inv}"
    );
}

/// Payload of the first sub-chunk in `body` (test-side, well-formed input).
fn chunks_of_sub(body: &[u8]) -> Vec<u8> {
    let size = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    body[8..8 + size].to_vec()
}

// ── Cross-check against exiftool -v3 ────────────────────────────────────────

/// Top-level `(fourcc, payload size, payload offset)` as `exiftool -v3` lists
/// them. The offset is the first hex-dump line of the chunk, absent for empty
/// chunks.
fn exiftool_chunks(bin: &str, path: &std::path::Path) -> Vec<(String, u64, Option<u64>)> {
    let out = std::process::Command::new(bin)
        .arg("-v3")
        .arg(path)
        .output()
        .expect("run exiftool");
    assert!(
        out.status.success(),
        "exiftool failed on {}",
        path.display()
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut chunks: Vec<(String, u64, Option<u64>)> = Vec::new();
    let mut want_offset = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RIFF '") {
            let (name, rest) = rest.split_once("' chunk (").expect("chunk header line");
            let size: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .expect("chunk size");
            chunks.push((name.to_string(), size, None));
            want_offset = true;
        } else if want_offset
            && let Some(hex) = line.strip_prefix("      ")
            && let Some((off, _)) = hex.split_once(": ")
            && let Ok(off) = u64::from_str_radix(off, 16)
        {
            chunks.last_mut().unwrap().2 = Some(off);
            want_offset = false;
        }
    }
    chunks
}

/// Every top-level chunk exiftool lists must be a chunk part of ours with the
/// same FourCC, payload offset and payload size, in the same order.
fn compare_with_exiftool(bin: &str, path: &std::path::Path, data: &[u8]) -> Result<usize, String> {
    let inv = inventory_of(data);
    let ours: Vec<_> = inv
        .children(None)
        .into_iter()
        .map(|id| inv.get(id).unwrap())
        .filter(|p| p.kind == PartKind::Chunk)
        .collect();
    let theirs = exiftool_chunks(bin, path);
    if ours.len() != theirs.len() {
        return Err(format!(
            "exiftool lists {} chunks, the inventory has {}",
            theirs.len(),
            ours.len()
        ));
    }
    for (i, (p, (name, size, off))) in ours.iter().zip(&theirs).enumerate() {
        let label = p.label.as_deref().unwrap_or("");
        if label != name {
            return Err(format!("chunk {i}: exiftool {name:?}, inventory {label:?}"));
        }
        let span = p.range.end - p.range.start - 8; // payload (+ zero pad)
        if !(span == *size || span == *size + 1 && size % 2 == 1) {
            return Err(format!(
                "chunk {i} {name}: exiftool size {size}, inventory span {span}"
            ));
        }
        if let Some(off) = off
            && *off != p.range.start + 8
        {
            return Err(format!(
                "chunk {i} {name}: exiftool payload offset {off}, inventory {}",
                p.range.start + 8
            ));
        }
    }
    Ok(ours.len())
}

/// Run with `INVENTORY_ORACLE_EXIFTOOL=exiftool` (see `just inventory-oracle`).
#[test]
#[ignore = "needs INVENTORY_ORACLE_EXIFTOOL; run via `just inventory-oracle`"]
fn oracle_exiftool_chunk_list() {
    let bin = std::env::var("INVENTORY_ORACLE_EXIFTOOL")
        .expect("set INVENTORY_ORACLE_EXIFTOOL to the exiftool binary");
    let tmp = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let dir = corpus.get("webp-conformance/valid").expect("corpus dir");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("webp"))
        .collect();
    names.sort();
    for p in names {
        files.push((
            format!("corpus/{}", p.file_name().unwrap().to_string_lossy()),
            p,
        ));
    }
    for (name, data) in [
        ("synthetic/everything_still", everything_still_no_junk().0),
        ("synthetic/animation", animation()),
        ("synthetic/lossy_alpha", lossy_alpha()),
    ] {
        let path = tmp.join(name.replace('/', "_") + ".webp");
        std::fs::write(&path, data).unwrap();
        files.push((name.to_string(), path));
    }
    assert!(files.len() >= 20);
    let mut failures = Vec::new();
    let mut chunk_total = 0;
    for (name, path) in &files {
        let data = std::fs::read(path).unwrap();
        match compare_with_exiftool(&bin, path, &data) {
            Ok(n) => {
                chunk_total += n;
                println!("oracle ok   {n:2} chunks  {name}");
            }
            Err(e) => {
                println!("oracle DIFF {name}: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    println!(
        "oracle: {} files, {chunk_total} chunks matched",
        files.len()
    );
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
}

/// The three container walkers disagree on metadata: `WebPDecoder` (the zencodec
/// path, which the inventory follows) keeps the first ICCP/EXIF/XMP and ignores
/// the VP8X flags; `WebPDemuxer` keeps the last one and only when the flag is
/// set.
#[test]
fn current_behaviour_decoder_and_demuxer_disagree_on_metadata() {
    let (mut data, _, exif, _) = everything_still_no_junk();
    let second_exif = b"II*\0second-exif-ignored".to_vec();
    // Flags say EXIF present: decoder keeps the first, demuxer the last.
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.embedded_metadata.exif.as_deref(), Some(&exif[..]));
    let demux = zenwebp::mux::WebPDemuxer::new(&data).unwrap();
    assert_eq!(demux.exif(), Some(&second_exif[..]));
    // Flag cleared: decoder still reports EXIF, demuxer reports none.
    data[20] &= !0x08;
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.embedded_metadata.exif.as_deref(), Some(&exif[..]));
    let demux = zenwebp::mux::WebPDemuxer::new(&data).unwrap();
    assert_eq!(demux.exif(), None);
}

// ── Review round 1: dispositions per zencodec entry point ───────────────────

fn seeded_rgb(w: usize, h: usize, seed: u8) -> Vec<u8> {
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

fn seeded_rgba(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 4);
    for (i, c) in seeded_rgb(w, h, seed).chunks(3).enumerate() {
        let x = i % w;
        px.extend([c[0], c[1], c[2], (255 - (x * 255 / w)) as u8 ^ seed]);
    }
    px
}

fn lossy_seed(w: usize, h: usize, seed: u8) -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &seeded_rgb(w, h, seed),
        PixelLayout::Rgb8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap()
}

fn lossy_alpha_seed(w: usize, h: usize, seed: u8) -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(LossyConfig::default()),
        &seeded_rgba(w, h, seed),
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap()
}

fn chunk_payload(chunks: &[Chunk], f: &[u8; 4]) -> Vec<u8> {
    chunks.iter().find(|c| &c.0 == f).unwrap().1.clone()
}

fn zc_pixels(data: &[u8]) -> Vec<u8> {
    let out = WebpDecoderConfig::new()
        .decode(data)
        .expect("zencodec decode");
    out.pixels().contiguous_bytes().into_owned()
}

fn probe_details(info: &zencodec::ImageInfo) -> zenwebp::detect::WebPProbe {
    webp_probe(info.source_encoding_details())
}

fn webp_probe(d: Option<&dyn zencodec::SourceEncodingDetails>) -> zenwebp::detect::WebPProbe {
    d.and_then(|d| d.codec_details::<zenwebp::detect::WebPProbe>())
        .cloned()
        .expect("WebPProbe details")
}

fn vp8x_still(w: u32, h: u32, flags: u8) -> Vec<u8> {
    let mut v = vec![flags, 0, 0, 0];
    v.extend((w - 1).to_le_bytes()[..3].iter());
    v.extend((h - 1).to_le_bytes()[..3].iter());
    v
}

/// Review F1: `decode()` takes a lossy VP8X still's pixels from `WebPDemuxer`,
/// which keeps the LAST VP8 and the LAST ALPH; the native decoder keeps the first.
#[test]
fn review_f1_duplicate_image_chunks_follow_decode() {
    let a = lossy_seed(32, 24, 0);
    let b = lossy_seed(32, 24, 0x5a);
    let vp8_a = chunk_payload(&chunks_of(&a), b"VP8 ");
    let vp8_b = chunk_payload(&chunks_of(&b), b"VP8 ");
    let data = riff(&[
        (*b"VP8X", vp8x_still(32, 24, 0)),
        (*b"VP8 ", vp8_a),
        (*b"VP8 ", vp8_b),
    ]);
    assert_eq!(
        zc_pixels(&data),
        zc_pixels(&b),
        "decode() uses the last VP8"
    );
    assert_ne!(zc_pixels(&data), zc_pixels(&a));
    let inv = inventory_of(&data);
    let vp8 = find(&inv, "VP8 ");
    assert_eq!(vp8[0].disposition, Disposition::Skipped, "{inv}");
    assert_eq!(vp8[1].disposition, Disposition::ImageData, "{inv}");
    assert!(
        vp8[1]
            .detail
            .as_deref()
            .unwrap()
            .contains("native WebPDecoder would use the first")
    );

    // Duplicate ALPH: the last plane is applied.
    let (la, lb) = (lossy_alpha_seed(32, 24, 0), lossy_alpha_seed(32, 24, 0x5a));
    let (ca, cb) = (chunks_of(&la), chunks_of(&lb));
    let two = riff(&[
        (*b"VP8X", chunk_payload(&ca, b"VP8X")),
        (*b"ALPH", chunk_payload(&ca, b"ALPH")),
        (*b"ALPH", chunk_payload(&cb, b"ALPH")),
        (*b"VP8 ", chunk_payload(&ca, b"VP8 ")),
    ]);
    let second_only = riff(&[
        (*b"VP8X", chunk_payload(&ca, b"VP8X")),
        (*b"ALPH", chunk_payload(&cb, b"ALPH")),
        (*b"VP8 ", chunk_payload(&ca, b"VP8 ")),
    ]);
    assert_eq!(zc_pixels(&two), zc_pixels(&second_only));
    let inv = inventory_of(&two);
    let alph = find(&inv, "ALPH");
    assert_eq!(alph[0].disposition, Disposition::Skipped, "{inv}");
    assert!(alph[1].disposition.is_consumed(), "{inv}");

    // [VP8L, VP8 ]: the native decoder rejects the file, the demuxer path decodes the VP8.
    let ll = chunk_payload(&chunks_of(&lossless_rgb()), b"VP8L");
    let vp8_32 = chunk_payload(&chunks_of(&a), b"VP8 ");
    let mixed = riff(&[
        (*b"VP8X", vp8x_still(32, 24, 0)),
        (*b"VP8L", ll),
        (*b"VP8 ", vp8_32),
    ]);
    assert!(WebpDecoderConfig::new().decode(&mixed).is_ok());
    let inv = inventory_of(&mixed);
    assert_eq!(
        find(&inv, "VP8L")[0].disposition,
        Disposition::Skipped,
        "{inv}"
    );
    assert_eq!(
        find(&inv, "VP8 ")[0].disposition,
        Disposition::ImageData,
        "{inv}"
    );
}

/// Review F2: 8+ trailing bytes after a VP8X lossy still make `probe()` fail while
/// `decode()` succeeds without ICC/EXIF/XMP in `ImageInfo`.
#[test]
fn review_f2_trailing_bytes_drop_metadata_in_decode() {
    let base = chunks_of(&lossy_alpha_seed(32, 24, 7));
    let mut vp8x = chunk_payload(&base, b"VP8X");
    vp8x[0] |= 0x20 | 0x08 | 0x04;
    let mut data = riff(&[
        (*b"VP8X", vp8x),
        (*b"ICCP", vec![0xAB; 64]),
        (*b"ALPH", chunk_payload(&base, b"ALPH")),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
        (*b"EXIF", b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec()),
        (*b"XMP ", b"<x:xmpmeta/>".to_vec()),
    ]);
    assert!(
        WebpDecoderConfig::new()
            .decode(&data)
            .unwrap()
            .info()
            .embedded_metadata
            .exif
            .is_some()
    );
    data.extend(b"trailing junk, longer than a chunk header");
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    let out = WebpDecoderConfig::new()
        .decode(&data)
        .expect("decode survives");
    assert!(out.info().source_color.icc_profile.is_none());
    assert!(out.info().embedded_metadata.exif.is_none());
    assert!(out.info().embedded_metadata.xmp.is_none());
    // ...but WebPProbe, attached to the output, still carries the ICC profile.
    assert_eq!(
        webp_probe(out.source_encoding_details())
            .icc_profile
            .as_deref(),
        Some(&[0xAB; 64][..])
    );
    let inv = inventory_of(&data);
    assert_eq!(
        find(&inv, "EXIF")[0].disposition,
        Disposition::Dropped,
        "{inv}"
    );
    assert_eq!(
        find(&inv, "XMP ")[0].disposition,
        Disposition::Dropped,
        "{inv}"
    );
    // The ICCP: dropped from ImageInfo, still exposed through WebPProbe.
    assert_eq!(
        find(&inv, "ICCP")[0].disposition,
        Disposition::Metadata(MetadataKind::Icc)
    );
    assert!(
        inv.get(inv.children(None)[0])
            .unwrap()
            .detail
            .as_deref()
            .unwrap()
            .contains("probe() fails")
    );
    let last = inv.get(inv.children(None).pop().unwrap()).unwrap();
    assert_eq!(last.disposition, Disposition::Trailing);
}

/// Review F3: `detect::probe` (source_encoding_details) reads chunks the decoder skips.
#[test]
fn review_f3_probe_view_of_skipped_chunks() {
    // (a) ICCP after a simple image, and past the RIFF end.
    let vp8l = chunk_payload(&chunks_of(&lossless_rgb()), b"VP8L");
    let icc = b"ICC-after-simple-image-serial-777".to_vec();
    let after = riff(&[(*b"VP8L", vp8l.clone()), (*b"ICCP", icc.clone())]);
    let mut past_end = riff(&[(*b"VP8L", vp8l)]);
    past_end.extend(chunk_bytes(b"ICCP", &icc));
    for data in [after, past_end] {
        let out = WebpDecoderConfig::new().decode(&data).unwrap();
        assert!(out.info().source_color.icc_profile.is_none());
        assert_eq!(
            webp_probe(out.source_encoding_details())
                .icc_profile
                .as_deref(),
            Some(&icc[..])
        );
        let inv = inventory_of(&data);
        let part = find(&inv, "ICCP")[0];
        assert_eq!(
            part.disposition,
            Disposition::Metadata(MetadataKind::Icc),
            "{inv}"
        );
        assert!(
            part.detail
                .as_deref()
                .unwrap()
                .contains("source_encoding_details")
        );
    }
    // (b) duplicate ICCP in a VP8X still: ImageInfo keeps the first, WebPProbe the last.
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    let data = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0x20)),
        (*b"ICCP", b"first-icc".to_vec()),
        (*b"VP8 ", vp8.clone()),
        (*b"ICCP", b"second-icc-skipped".to_vec()),
    ]);
    let out = WebpDecoderConfig::new().decode(&data).unwrap();
    assert_eq!(
        out.info().source_color.icc_profile.as_deref(),
        Some(&b"first-icc"[..])
    );
    assert_eq!(
        webp_probe(out.source_encoding_details())
            .icc_profile
            .as_deref(),
        Some(&b"second-icc-skipped"[..])
    );
    let inv = inventory_of(&data);
    let iccs = find(&inv, "ICCP");
    assert_eq!(
        iccs[0].disposition,
        Disposition::Metadata(MetadataKind::Icc)
    );
    assert_eq!(
        iccs[1].disposition,
        Disposition::Metadata(MetadataKind::Icc),
        "{inv}"
    );
    assert!(iccs[1].detail.as_deref().unwrap().contains("duplicate"));
    // (c) a second VP8X feeds WebPProbe's size and alpha.
    let data = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0)),
        (*b"VP8 ", vp8),
        (*b"VP8X", vec![0x10, 0, 0, 0, 0xff, 0x0f, 0, 0xff, 0x0f, 0]),
    ]);
    let p = probe_details(&WebpDecoderConfig::new().job().probe(&data).unwrap());
    assert_eq!((p.width, p.has_alpha), (4096, true));
    let inv = inventory_of(&data);
    assert!(
        find(&inv, "VP8X")[1]
            .detail
            .as_deref()
            .unwrap()
            .contains("WebPProbe")
    );
    // (d) animated file with a top-level VP8 before the first ANMF: WebPProbe's quality comes from it.
    let mut chunks = chunks_of(&animation());
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    chunks.insert(
        i,
        (
            *b"VP8 ",
            chunk_payload(&chunks_of(&lossy_seed(8, 8, 3)), b"VP8 "),
        ),
    );
    let data = riff(&chunks);
    let inv = inventory_of(&data);
    let vp8 = find(&inv, "VP8 ")[0];
    assert_eq!(vp8.disposition, Disposition::Skipped);
    assert!(vp8.detail.as_deref().unwrap().contains("WebPProbe quality"));
}

fn set_vp8l_alpha_bit(anmf: &mut [u8], on: bool) {
    assert_eq!(&anmf[16..20], b"VP8L");
    let b = &mut anmf[16 + 8 + 4];
    if on { *b |= 0x10 } else { *b &= !0x10 }
}

/// Review F4: an unreachable ANMF's header and first sub-chunk header still feed `has_alpha`.
#[test]
fn review_f4_unreachable_frame_bytes_that_the_decoder_reads() {
    let mut chunks = chunks_of(&animation());
    chunks[0].1[0] &= !0x10;
    for c in chunks.iter_mut().filter(|c| &c.0 == b"ANMF") {
        set_vp8l_alpha_bit(&mut c.1, false);
    }
    let second = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.0 == b"ANMF")
        .nth(1)
        .unwrap()
        .0;
    chunks.insert(second, (*b"JUNK", vec![1, 2, 3, 4]));
    let base = riff(&chunks);
    assert!(
        !WebpDecoderConfig::new()
            .job()
            .probe(&base)
            .unwrap()
            .has_alpha
    );
    let third = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| &c.0 == b"ANMF")
        .nth(2)
        .unwrap()
        .0;
    set_vp8l_alpha_bit(&mut chunks[third].1, true);
    let data = riff(&chunks);
    assert!(
        WebpDecoderConfig::new()
            .job()
            .probe(&data)
            .unwrap()
            .has_alpha,
        "a bit in the unreachable frame 3 switches the output to RGBA"
    );
    let inv = inventory_of(&data);
    // Every frame header is read, reachable or not.
    let headers: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.kind == PartKind::Header && p.range.end - p.range.start == 16)
        .collect();
    assert_eq!(headers.len(), 3);
    assert!(
        headers
            .iter()
            .all(|h| h.disposition == Disposition::Metadata(MetadataKind::Animation))
    );
    // Frame 3's VP8L: header and the alpha word are Structure; the pixels are Skipped.
    let vp8l = find(&inv, "VP8L")
        .into_iter()
        .filter(|p| p.parent.is_some())
        .nth(2)
        .unwrap();
    assert_eq!(vp8l.disposition, Disposition::Structure, "{inv}");
    let word = inv
        .parts()
        .iter()
        .find(|p| {
            matches!(&p.tag, PartTag::Name(n) if n == "alpha_is_used header")
                && p.range.start == vp8l.range.start + 8
        })
        .expect("alpha_is_used field of frame 3");
    assert_eq!(word.disposition, Disposition::Structure);
    assert_eq!(word.range.start, vp8l.range.start + 8);
    assert_eq!(word.range.end, vp8l.range.start + 13);
    assert!(
        inv.parts()
            .iter()
            .any(|p| p.kind == PartKind::Gap && p.disposition == Disposition::Skipped)
    );
}

/// Review F5: ICCP/EXIF/XMP at the start of the first ANMF reach `ImageInfo`.
#[test]
fn review_f5_metadata_in_first_anmf_reaches_image_info() {
    let icc = vec![0x11u8; 40];
    let exif = b"II*\0\x08\0\0\0\0\0\0\0\0\0frame-exif-serial-XYZ".to_vec();
    let xmp = b"<x:xmpmeta>frame</x:xmpmeta>".to_vec();
    for (fourcc, payload) in [(*b"ICCP", &icc), (*b"EXIF", &exif), (*b"XMP ", &xmp)] {
        for variant in 0..2 {
            let mut prefix = Vec::new();
            if variant == 1 {
                prefix.extend(chunk_bytes(b"ZERO", &[]));
            }
            prefix.extend(chunk_bytes(&fourcc, payload));
            let mut chunks = chunks_of(&animation());
            let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
            let mut p = chunks[i].1[..16].to_vec();
            p.extend(&prefix);
            p.extend(&chunks[i].1[16..]);
            chunks[i].1 = p;
            let data = riff(&chunks);
            let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
            let reported = match &fourcc {
                b"ICCP" => info.source_color.icc_profile.as_deref().map(<[u8]>::to_vec),
                b"EXIF" => info.embedded_metadata.exif.as_deref().map(<[u8]>::to_vec),
                _ => info.embedded_metadata.xmp.as_deref().map(<[u8]>::to_vec),
            };
            assert_eq!(reported.as_deref(), Some(&payload[..]), "variant {variant}");
            let inv = inventory_of(&data);
            let part = find(&inv, std::str::from_utf8(&fourcc).unwrap())[0];
            assert!(
                matches!(part.disposition, Disposition::Metadata(_)),
                "{} v{variant}: {inv}",
                String::from_utf8_lossy(&fourcc)
            );
        }
    }
}

/// Review F6: files the decoder rejects report nothing as consumed beyond the headers.
#[test]
fn review_f6_rejected_files_report_nothing_consumed() {
    // (a) an ANMF smaller than 24 bytes: read_data fails.
    let mut chunks = chunks_of(&animation());
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    let anmf = chunks[i].1.clone();
    chunks.insert(i, (*b"ANMF", anmf[..20].to_vec()));
    let data = riff(&chunks);
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    let inv = inventory_of(&data);
    let consumed: Vec<_> = top_level_chunks(&inv)
        .into_iter()
        .filter(|(l, d)| d.is_consumed() && l != "VP8X")
        .collect();
    assert!(consumed.is_empty(), "{consumed:?}\n{inv}");
    assert!(
        inv.parts()[0]
            .detail
            .as_deref()
            .unwrap()
            .contains("decoder rejects")
    );

    // (b) lossless VP8X still with the EXIF flag set and no EXIF chunk: ChunkMissing.
    let vp8l = chunk_payload(&chunks_of(&lossless_rgb()), b"VP8L");
    let data = riff(&[
        (*b"VP8X", vp8x_still(32, 24, 0x08 | 0x20)),
        (*b"ICCP", vec![0x42; 20]),
        (*b"VP8L", vp8l),
    ]);
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    let inv = inventory_of(&data);
    assert_eq!(
        find(&inv, "ICCP")[0].disposition,
        Disposition::Skipped,
        "{inv}"
    );
    assert!(!find(&inv, "VP8L")[0].disposition.is_consumed());

    // (c) the same flag error on a LOSSY still still decodes (demuxer path), without metadata.
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(32, 24, 0)), b"VP8 ");
    let data = riff(&[
        (*b"VP8X", vp8x_still(32, 24, 0x08 | 0x20)),
        (*b"ICCP", vec![0x42; 20]),
        (*b"VP8 ", vp8),
    ]);
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    let out = WebpDecoderConfig::new().decode(&data).unwrap();
    assert!(out.info().source_color.icc_profile.is_none());
    let inv = inventory_of(&data);
    assert_eq!(
        find(&inv, "VP8 ")[0].disposition,
        Disposition::ImageData,
        "{inv}"
    );
    assert_eq!(
        find(&inv, "ICCP")[0].disposition,
        Disposition::Metadata(MetadataKind::Icc)
    );
}

/// Review F7: a header cut short inside the RIFF range is Malformed, not Trailing.
#[test]
fn review_f7_truncated_header_inside_a_vp8x_riff() {
    let mut chunks = chunks_of(&lossy_alpha_seed(32, 24, 7));
    chunks.push((*b"EXIF", b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec()));
    let full = riff(&chunks);
    let exif_at = full.windows(4).rposition(|w| w == b"EXIF").unwrap();
    let inv = inventory_of(&full[..exif_at + 3]);
    let last = inv.get(inv.children(None).pop().unwrap()).unwrap();
    assert_eq!(last.disposition, Disposition::Malformed, "{inv}");
    assert!(
        last.detail
            .as_deref()
            .unwrap()
            .contains("truncated chunk header")
    );
}

/// Review F8: a VP8X payload under 10 bytes is parsed anyway.
#[test]
fn review_f8_short_vp8x_is_parsed_by_the_decoder() {
    let base = chunks_of(&lossy_alpha_seed(4, 4, 9));
    let mut data = riff(&[
        (*b"VP8X", vec![0x10, 0, 0, 0]),
        ([3, 0, 0, 3], vec![]),
        (*b"ALPH", chunk_payload(&base, b"ALPH")),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
    ]);
    let riff_size = (data.len() - 8) as u32;
    data[4..8].copy_from_slice(&riff_size.to_le_bytes());
    let out = WebpDecoderConfig::new().decode(&data).unwrap();
    assert!(out.info().has_alpha);
    let inv = inventory_of(&data);
    let vp8x = find(&inv, "VP8X")[0];
    assert_eq!(vp8x.disposition, Disposition::Structure, "{inv}");
    assert!(vp8x.detail.as_deref().unwrap().contains("decoder reads 10"));
    assert_eq!(
        find(&inv, "ALPH")[0].disposition,
        Disposition::ImageData,
        "{inv}"
    );
    assert_eq!(
        find(&inv, "VP8 ")[0].disposition,
        Disposition::ImageData,
        "{inv}"
    );
}

/// Review F9: the chunk after ALPH in an ANMF is decoded as VP8 whatever its FourCC.
#[test]
fn review_f9_chunk_after_alph_is_decoded_as_vp8() {
    let mut anim = AnimationEncoder::new(
        16,
        16,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    for (i, seed) in [1u8, 2].into_iter().enumerate() {
        anim.add_frame(
            &seeded_rgba(16, 16, seed),
            PixelLayout::Rgba8,
            i as u32 * 100,
            &EncoderConfig::Lossy(LossyConfig::default()),
        )
        .unwrap();
    }
    let data = anim.finalize(100).unwrap();
    let orig = zenwebp::mux::AnimationDecoder::new(&data)
        .unwrap()
        .decode_all()
        .unwrap();
    let mut chunks = chunks_of(&data);
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    let alph_len = u32::from_le_bytes(chunks[i].1[20..24].try_into().unwrap()) as usize;
    let off = 16 + 8 + alph_len + (alph_len & 1);
    assert_eq!(&chunks[i].1[off..off + 4], b"VP8 ");
    chunks[i].1[off..off + 4].copy_from_slice(b"ZZZZ");
    let data = riff(&chunks);
    let renamed = zenwebp::mux::AnimationDecoder::new(&data)
        .unwrap()
        .decode_all()
        .unwrap();
    assert_eq!(renamed[0].data, orig[0].data, "decoded identically");
    let inv = inventory_of(&data);
    let zz = find(&inv, "ZZZZ")[0];
    assert_eq!(zz.disposition, Disposition::ImageData, "{inv}");
    assert!(
        zz.detail
            .as_deref()
            .unwrap()
            .contains("whatever its FourCC")
    );
}

/// Review F10 / COMMON item 4: no unread bytes hidden inside consumed parts that can be found cheaply.
#[test]
fn review_f10_reserved_bytes_and_alpha_slack() {
    let (data, ..) = everything_still_no_junk();
    let inv = inventory_of(&data);
    let reserved: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| matches!(&p.tag, PartTag::Name(n) if n == "reserved"))
        .collect();
    assert_eq!(reserved.len(), 1);
    assert_eq!(reserved[0].disposition, Disposition::Dropped);
    assert_eq!(reserved[0].range, 21..24);

    // Uncompressed ALPH (compression 0) with 9 bytes past width*height.
    let base = chunks_of(&lossy_alpha_seed(8, 8, 4));
    let mut alph = vec![0u8]; // header: no compression, no filter
    alph.extend((0..64u8).map(|i| 255 - i));
    let exact = riff(&[
        (*b"VP8X", chunk_payload(&base, b"VP8X")),
        (*b"ALPH", alph.clone()),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
    ]);
    alph.extend(b"secret-123");
    let slack = riff(&[
        (*b"VP8X", chunk_payload(&base, b"VP8X")),
        (*b"ALPH", alph),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
    ]);
    assert_eq!(
        zc_pixels(&exact),
        zc_pixels(&slack),
        "slack does not change the pixels"
    );
    let inv = inventory_of(&slack);
    let plane = inv
        .parts()
        .iter()
        .find(|p| matches!(&p.tag, PartTag::Name(n) if n == "alpha plane"))
        .expect("alpha plane child");
    assert_eq!(plane.disposition, Disposition::ImageData);
    assert_eq!(plane.range.end - plane.range.start, 65);
    let tail = inv
        .parts()
        .iter()
        .find(|p| p.kind == PartKind::Gap && p.disposition == Disposition::Unreferenced)
        .expect("slack gap");
    assert_eq!(tail.range.end - tail.range.start, 10);
}

// ── Review round 2 ──────────────────────────────────────────────────────────

fn lossless_rgba_vp8l(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let ll = EncodeRequest::new(
        &EncoderConfig::new_lossless(),
        &seeded_rgba(w, h, seed),
        PixelLayout::Rgba8,
        w as u32,
        h as u32,
    )
    .encode()
    .unwrap();
    chunk_payload(&chunks_of(&ll), b"VP8L")
}

fn lossy_alpha_animation() -> Vec<u8> {
    let mut anim = AnimationEncoder::new(
        16,
        16,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    for (i, seed) in [1u8, 2].into_iter().enumerate() {
        anim.add_frame(
            &seeded_rgba(16, 16, seed),
            PixelLayout::Rgba8,
            i as u32 * 100,
            &EncoderConfig::Lossy(LossyConfig::default()),
        )
        .unwrap();
    }
    anim.finalize(100).unwrap()
}

fn first_top_part(inv: &Inventory) -> &zencodec::inventory::Part {
    inv.get(inv.children(None)[0]).unwrap()
}

/// R8: `Limits::default()` (16384 px per side, 120 MP, 10,000 frames) rejects in
/// `read_data`; the model calls the decoder's own checks.
#[test]
fn review2_r8_default_limits_are_modelled() {
    // (a) 11000x11000 canvas lossy still: probe() fails, decode() succeeds via the
    // demuxer without ImageInfo metadata or alpha.
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(8, 8, 1)), b"VP8 ");
    let mut vp8x = vec![0x20 | 0x08, 0, 0, 0];
    vp8x.extend((11000u32 - 1).to_le_bytes()[..3].iter());
    vp8x.extend((11000u32 - 1).to_le_bytes()[..3].iter());
    let data = riff(&[
        (*b"VP8X", vp8x),
        (*b"ICCP", b"icc-big".to_vec()),
        (*b"VP8 ", vp8),
        (*b"EXIF", b"II*\0exif-big".to_vec()),
    ]);
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    let out = WebpDecoderConfig::new().decode(&data).unwrap();
    assert!(out.info().embedded_metadata.exif.is_none());
    assert!(out.info().source_color.icc_profile.is_none());
    let inv = inventory_of(&data);
    assert_eq!(
        find(&inv, "EXIF")[0].disposition,
        Disposition::Dropped,
        "{inv}"
    );
    assert!(
        first_top_part(&inv)
            .detail
            .as_deref()
            .unwrap()
            .contains("total pixels")
    );
    // ICCP stays reachable through WebPProbe, and says so.
    assert_eq!(
        find(&inv, "ICCP")[0].disposition,
        Disposition::Metadata(MetadataKind::Icc)
    );

    // (b) 10,001 frames: probe() and decode() both fail.
    let mut chunks = chunks_of(&animation());
    let anmf = chunks.iter().find(|c| &c.0 == b"ANMF").unwrap().clone();
    chunks.retain(|c| &c.0 != b"ANMF");
    for _ in 0..10_001 {
        chunks.push(anmf.clone());
    }
    let data = riff(&chunks);
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    let inv = inventory_of(&data);
    assert!(
        first_top_part(&inv)
            .detail
            .as_deref()
            .unwrap()
            .contains("frame count 10001")
    );
    assert!(
        !inv.parts().iter().any(|p| matches!(
            p.disposition,
            Disposition::ImageData | Disposition::Metadata(_)
        )),
        "a rejected file consumes nothing"
    );

    // (c) canvas over 16384 on a side.
    let ll = lossless_rgba_vp8l(8, 8, 1);
    let mut vp8x = vec![0x08, 0, 0, 0];
    vp8x.extend((32u32 - 1).to_le_bytes()[..3].iter());
    vp8x.extend((57624u32 - 1).to_le_bytes()[..3].iter());
    let data = riff(&[
        (*b"VP8X", vp8x),
        (*b"VP8L", ll),
        (*b"EXIF", b"II*\0e".to_vec()),
    ]);
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    let inv = inventory_of(&data);
    assert!(!find(&inv, "EXIF")[0].disposition.is_consumed(), "{inv}");

    // (d) simple VP8L whose header declares 16384x16384 (268 MP).
    let mut ll = lossless_rgba_vp8l(8, 8, 1);
    let mut header = u32::from_le_bytes(ll[1..5].try_into().unwrap());
    header = (header & !0x0FFF_FFFF) | 0x3FFF | (0x3FFF << 14);
    ll[1..5].copy_from_slice(&header.to_le_bytes());
    let data = riff(&[(*b"VP8L", ll), (*b"ICCP", b"icc-after".to_vec())]);
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    let inv = inventory_of(&data);
    let image = find(&inv, "VP8L")[0];
    assert_eq!(image.disposition, Disposition::Malformed, "{inv}");
    assert!(image.detail.as_deref().unwrap().contains("exceeds limit"));
    assert!(!find(&inv, "ICCP")[0].disposition.is_consumed());
}

/// R1: an ANMF in a non-animated file still feeds the first-ANMF scan.
#[test]
fn review2_r1_anmf_in_a_still_files_metadata() {
    let vp8l = lossless_rgba_vp8l(8, 8, 1);
    let icc = b"ICC-inside-an-ANMF-of-a-still".to_vec();
    let mut anmf = vec![0u8; 16];
    anmf[6] = 7;
    anmf[9] = 7;
    anmf.extend(chunk_bytes(b"ICCP", &icc));
    anmf.extend(chunk_bytes(b"VP8L", &vp8l));
    let data = riff(&[
        (*b"VP8X", vec![0x10u8, 0, 0, 0, 7, 0, 0, 7, 0, 0]),
        (*b"VP8L", vp8l.clone()),
        (*b"ANMF", anmf),
    ]);
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.source_color.icc_profile.as_deref(), Some(&icc[..]));
    assert_eq!(
        WebpDecoderConfig::new()
            .decode(&data)
            .unwrap()
            .info()
            .source_color
            .icc_profile
            .as_deref(),
        Some(&icc[..])
    );
    let inv = inventory_of(&data);
    let iccp = find(&inv, "ICCP")[0];
    assert_eq!(
        iccp.disposition,
        Disposition::Metadata(MetadataKind::Icc),
        "{inv}"
    );
    assert!(iccp.parent.is_some());
    let anmf = find(&inv, "ANMF")[0];
    assert_eq!(anmf.disposition, Disposition::Skipped);
    assert!(
        anmf.detail
            .as_deref()
            .unwrap()
            .contains("animation flag is clear")
    );
    let header = inv
        .parts()
        .iter()
        .find(|p| p.kind == PartKind::Header && p.range.end - p.range.start == 16)
        .unwrap();
    assert_eq!(header.disposition, Disposition::Dropped);
}

/// R2: the first-ANMF scan's misread can file another chunk's payload.
#[test]
fn review2_r2_misread_header_files_another_chunks_payload() {
    let secret = b"<x:xmpmeta>owner=someone serial=42</x:xmpmeta>".to_vec();
    let mut chunks = chunks_of(&animation());
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    let mut misread = b"XMP ".to_vec();
    misread.extend((secret.len() as u32).to_le_bytes());
    let mut p = chunks[i].1[..16].to_vec();
    p.extend(chunk_bytes(b"ABCD", &misread));
    p.extend(chunk_bytes(b"ZEN2", &secret));
    p.extend(&chunks[i].1[16..]);
    chunks[i].1 = p;
    let data = riff(&chunks);
    let probe = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(probe.embedded_metadata.xmp.as_deref(), Some(&secret[..]));
    let inv = inventory_of(&data);
    let zen2 = find(&inv, "ZEN2")[0];
    assert_eq!(
        zen2.disposition,
        Disposition::Metadata(MetadataKind::Xmp),
        "{inv}"
    );
    assert!(
        zen2.detail
            .as_deref()
            .unwrap()
            .contains("misreads a header")
    );

    // A range inside a chunk's payload becomes a child of that chunk.
    let mut inner = vec![0u8; 3];
    inner.extend(b"XMP ");
    inner.extend(5u32.to_le_bytes());
    inner.extend(b"INNER!");
    let mut chunks = chunks_of(&animation());
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    let mut p = chunks[i].1[..16].to_vec();
    p.extend(chunk_bytes(b"ABCD", &[]));
    p.extend(chunk_bytes(b"ZEN3", &inner));
    p.extend(&chunks[i].1[16..]);
    chunks[i].1 = p;
    let data = riff(&chunks);
    let inv = inventory_of(&data);
    if let Some(x) = WebpDecoderConfig::new()
        .job()
        .probe(&data)
        .unwrap()
        .embedded_metadata
        .xmp
    {
        assert!(
            inv.parts()
                .iter()
                .any(|p| p.disposition == Disposition::Metadata(MetadataKind::Xmp)),
            "{x:?}\n{inv}"
        );
    }
}

/// R3: when `from_webp` fails, `decode()` outputs RGB: the alpha plane is discarded.
#[test]
fn review2_r3_alph_is_dropped_when_from_webp_fails() {
    let base = chunks_of(&lossy_alpha_seed(32, 24, 7));
    let mut data = riff(&[
        (*b"VP8X", chunk_payload(&base, b"VP8X")),
        (*b"ALPH", chunk_payload(&base, b"ALPH")),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
    ]);
    let clean = WebpDecoderConfig::new().decode(&data).unwrap();
    assert!(clean.has_alpha());
    assert_eq!(
        find(&inventory_of(&data), "ALPH")[0].disposition,
        Disposition::ImageData
    );
    data.extend(b"trailing junk, longer than a chunk header");
    let out = WebpDecoderConfig::new().decode(&data).unwrap();
    assert!(
        !out.has_alpha(),
        "decode() outputs RGB when from_webp fails"
    );
    let inv = inventory_of(&data);
    let alph = find(&inv, "ALPH")[0];
    assert_eq!(alph.disposition, Disposition::Dropped, "{inv}");
    assert!(alph.detail.as_deref().unwrap().contains("outputs RGB"));
}

fn lossless_anim_with_trailing_vp8() -> Vec<u8> {
    let mut chunks = chunks_of(&animation());
    chunks.push((
        *b"VP8 ",
        chunk_payload(&chunks_of(&lossy_seed(8, 8, 3)), b"VP8 "),
    ));
    riff(&chunks)
}

/// R4: when `detect::probe` fails, no `WebPProbe` is attached and nothing is
/// reported through it.
#[test]
fn review2_r4_probe_icc_not_claimed_when_detect_probe_fails() {
    let mut chunks = chunks_of(&lossy_alpha_animation());
    chunks[0].1[0] |= 0x20;
    let i = chunks.iter().position(|c| &c.0 == b"ANIM").unwrap();
    chunks.insert(i, (*b"ICCP", b"first-icc".to_vec()));
    chunks.push((*b"ICCP", b"second-icc-owner".to_vec()));
    let data = riff(&chunks);
    assert!(zenwebp::detect::probe(&data).is_err());
    assert!(
        WebpDecoderConfig::new()
            .job()
            .probe(&data)
            .unwrap()
            .source_encoding_details()
            .is_none()
    );
    let inv = inventory_of(&data);
    let iccs = find(&inv, "ICCP");
    assert_eq!(
        iccs[0].disposition,
        Disposition::Metadata(MetadataKind::Icc)
    );
    assert_eq!(iccs[1].disposition, Disposition::Skipped, "{inv}");

    // A VP8 whose uncompressed header detect::probe rejects: no WebPProbe either.
    let mut vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    vp8[3] = 0; // break the 9d 01 2a start code
    let data = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0x20)),
        (*b"ICCP", b"first-icc".to_vec()),
        (*b"VP8 ", vp8),
        (*b"ICCP", b"last-icc".to_vec()),
    ]);
    assert!(zenwebp::detect::probe(&data).is_err());
    let inv = inventory_of(&data);
    assert!(!inv.parts().iter().any(|p| {
        p.detail
            .as_deref()
            .is_some_and(|d| d.contains("reported through source_encoding_details"))
    }));
}

/// R5: the `WebPProbe` bitstream owner is the first of top-level VP8/VP8L and an
/// ANMF's first sub-chunk, in file order (the same computation as R4).
#[test]
fn review2_r5_probe_bitstream_owner_in_file_order() {
    let data = lossless_anim_with_trailing_vp8();
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    let d = info.source_encoding_details().unwrap();
    assert!(d.is_lossless(), "WebPProbe takes the first ANMF's VP8L");
    let inv = inventory_of(&data);
    let vp8 = find(&inv, "VP8 ")[0];
    assert!(
        !vp8.detail.as_deref().unwrap().contains("WebPProbe quality"),
        "{inv}"
    );
    let anmf = find(&inv, "ANMF")[0];
    assert!(
        anmf.detail
            .as_deref()
            .unwrap()
            .contains("WebPProbe quality/lossless"),
        "{inv}"
    );
    // With the VP8 first, it owns the bitstream.
    let mut chunks = chunks_of(&animation());
    let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
    chunks.insert(
        i,
        (
            *b"VP8 ",
            chunk_payload(&chunks_of(&lossy_seed(8, 8, 3)), b"VP8 "),
        ),
    );
    let inv = inventory_of(&riff(&chunks));
    assert!(
        find(&inv, "VP8 ")[0]
            .detail
            .as_deref()
            .unwrap()
            .contains("WebPProbe quality")
    );
}

/// R7: a truncated final ICCP is clamped and delivered through `WebPProbe`.
#[test]
fn review2_r7_truncated_final_iccp_is_still_metadata() {
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    let mut simple = riff(&[(*b"VP8 ", vp8.clone()), (*b"ICCP", vec![0x77; 8])]);
    simple.pop(); // 7 of the declared 8 payload bytes remain
    let mut ext = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0x20)),
        (*b"VP8 ", vp8),
        (*b"ICCP", vec![0x77; 8]),
    ]);
    ext.pop();
    for data in [simple, ext] {
        let out = WebpDecoderConfig::new().decode(&data).unwrap();
        assert_eq!(
            webp_probe(out.source_encoding_details())
                .icc_profile
                .as_deref(),
            Some(&[0x77u8; 7][..])
        );
        let inv = inventory_of(&data);
        let iccp = find(&inv, "ICCP")[0];
        assert_eq!(
            iccp.disposition,
            Disposition::Metadata(MetadataKind::Icc),
            "{inv}"
        );
        assert!(
            iccp.detail
                .as_deref()
                .unwrap()
                .contains("only 7 are present")
        );
    }
}

/// R9: `read_data`'s first-chunk header checks are modelled for simple files.
#[test]
fn review2_r9_simple_file_header_checks() {
    let good = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("start code", {
            let mut v = good.clone();
            v[3] = 0;
            v
        }),
        ("not a keyframe", {
            let mut v = good.clone();
            v[0] |= 1;
            v
        }),
        ("zero dimensions", {
            let mut v = good.clone();
            v[6..10].copy_from_slice(&[0, 0, 0, 0]);
            v
        }),
    ];
    for (name, vp8) in cases {
        let data = riff(&[(*b"VP8 ", vp8), (*b"ICCP", b"icc-after".to_vec())]);
        assert!(WebpDecoderConfig::new().decode(&data).is_err(), "{name}");
        let inv = inventory_of(&data);
        assert_eq!(
            find(&inv, "VP8 ")[0].disposition,
            Disposition::Malformed,
            "{name}\n{inv}"
        );
        assert!(
            !find(&inv, "ICCP")[0].disposition.is_consumed(),
            "{name}\n{inv}"
        );
    }
    let mut ll = lossless_rgba_vp8l(8, 8, 1);
    ll[0] = 0x2e;
    let data = riff(&[(*b"VP8L", ll)]);
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    assert_eq!(
        find(&inventory_of(&data), "VP8L")[0].disposition,
        Disposition::Malformed
    );
}

/// R10 / round 3 N1: a clamped last chunk is not a rejected file. The demuxer
/// clamps it; whether the bytes present suffice is a bitstream question.
#[test]
fn review3_n1_truncated_last_chunk_is_image_data_with_a_conditional_note() {
    let base = chunks_of(&lossy_alpha_seed(32, 24, 7));
    let mut vp8x = chunk_payload(&base, b"VP8X");
    vp8x[0] |= 0x20;
    let build = |alph_last: bool| {
        let mut chunks = vec![
            (*b"VP8X", vp8x.clone()),
            (*b"ICCP", b"icc-bytes-serial-9".to_vec()),
        ];
        if alph_last {
            chunks.push((*b"VP8 ", chunk_payload(&base, b"VP8 ")));
            chunks.push((*b"ALPH", chunk_payload(&base, b"ALPH")));
        } else {
            chunks.push((*b"ALPH", chunk_payload(&base, b"ALPH")));
            chunks.push((*b"VP8 ", chunk_payload(&base, b"VP8 ")));
        }
        riff(&chunks)
    };
    let patch_size = |data: &mut Vec<u8>, fourcc: &[u8; 4], add: u32| {
        let at = data.windows(4).rposition(|w| w == fourcc).unwrap() + 4;
        let n = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) + add;
        data[at..at + 4].copy_from_slice(&n.to_le_bytes());
    };
    let mut cases: Vec<(&str, Vec<u8>, &[u8; 4])> = Vec::new();
    let mut d = build(false);
    d.pop();
    cases.push(("VP8 cut 1 byte", d, b"VP8 "));
    let mut d = build(false);
    patch_size(&mut d, b"VP8 ", 100);
    cases.push(("VP8 size +100", d, b"VP8 "));
    let mut d = build(true);
    patch_size(&mut d, b"ALPH", 50);
    cases.push(("ALPH size +50", d, b"ALPH"));
    for (name, data, label) in cases {
        let out = WebpDecoderConfig::new().decode(&data).expect(name);
        assert_eq!(
            webp_probe(out.source_encoding_details())
                .icc_profile
                .as_deref(),
            Some(&b"icc-bytes-serial-9"[..]),
            "{name}"
        );
        let inv = inventory_of(&data);
        assert!(
            !inv.parts()[0]
                .detail
                .as_deref()
                .unwrap()
                .contains("decoder rejects"),
            "{name}\n{inv}"
        );
        let l = std::str::from_utf8(label).unwrap();
        let part = find(&inv, l).into_iter().next_back().unwrap();
        // ALPH is decoded then dropped here: from_webp fails, so the output is RGB.
        assert!(
            part.disposition.is_consumed() || part.disposition == Disposition::Dropped,
            "{name}\n{inv}"
        );
        let d = part.detail.as_deref().unwrap();
        assert!(
            d.contains("only") && d.contains("else it falls back to WebPDecoder"),
            "{name}: {d}"
        );
        assert_eq!(
            find(&inv, "ICCP")[0].disposition,
            Disposition::Metadata(MetadataKind::Icc)
        );
    }

    // Cut 20 bytes: decode fails, and the wording still says "when they suffice".
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(32, 24, 2)), b"VP8 ");
    let mut data = riff(&[
        (*b"VP8X", vp8x_still(32, 24, 0x20)),
        (*b"ICCP", b"icc".to_vec()),
        (*b"VP8 ", vp8.clone()),
        (*b"VP8 ", vp8),
    ]);
    data.truncate(data.len() - 20);
    assert!(WebpDecoderConfig::new().decode(&data).is_err());
    let inv = inventory_of(&data);
    let last = find(&inv, "VP8 ").into_iter().next_back().unwrap();
    assert_eq!(last.disposition, Disposition::ImageData, "{inv}");
    assert!(
        last.detail
            .as_deref()
            .unwrap()
            .contains("when they suffice")
    );

    // Simple files: a truncated VP8 still decodes (ImageData); a truncated VP8L fails (Malformed).
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(64, 64, 4)), b"VP8 ");
    let mut simple = riff(&[(*b"VP8 ", vp8)]);
    let n = u32::from_le_bytes(simple[16..20].try_into().unwrap()) + 100;
    simple[16..20].copy_from_slice(&n.to_le_bytes()); // size overshoots, every bitstream byte is present
    assert!(WebpDecoderConfig::new().decode(&simple).is_ok());
    let inv = inventory_of(&simple);
    assert_eq!(
        find(&inv, "VP8 ")[0].disposition,
        Disposition::ImageData,
        "{inv}"
    );
    let mut simple_ll = riff(&[(*b"VP8L", lossless_rgba_vp8l(16, 16, 1))]);
    simple_ll.truncate(simple_ll.len() - 2);
    assert!(WebpDecoderConfig::new().decode(&simple_ll).is_err());
    assert_eq!(
        find(&inventory_of(&simple_ll), "VP8L")[0].disposition,
        Disposition::Malformed
    );
}

/// F10 documentation gaps: the "not distinguished" note on frame images, and the
/// right detail for an uncompressed ALPH without slack.
#[test]
fn review2_f10_details() {
    let inv = inventory_of(&animation());
    let frame = inv
        .parts()
        .iter()
        .find(|p| p.parent.is_some() && p.label.as_deref() == Some("VP8L"))
        .unwrap();
    assert!(
        frame
            .detail
            .as_deref()
            .unwrap()
            .contains("not distinguished")
    );
    let inv = inventory_of(&lossy_alpha_animation());
    let alph = find(&inv, "ALPH")[0];
    assert!(
        alph.detail
            .as_deref()
            .unwrap()
            .contains("not distinguished")
    );

    let base = chunks_of(&lossy_alpha_seed(8, 8, 4));
    let mut plane = vec![0u8];
    plane.extend((0..64u8).map(|i| 255 - i));
    let data = riff(&[
        (*b"VP8X", chunk_payload(&base, b"VP8X")),
        (*b"ALPH", plane),
        (*b"VP8 ", chunk_payload(&base, b"VP8 ")),
    ]);
    let alph = find(&inventory_of(&data), "ALPH")[0].clone();
    let d = alph.detail.as_deref().unwrap();
    assert!(
        d.contains("no bytes after it") && !d.contains("; compressed alpha"),
        "{d}"
    );
}

/// Review round 2, R8 companion: `read_extended_header` rejects a canvas whose
/// pixel count overflows u32 before any limit is consulted.
#[test]
fn review2_r8_canvas_product_overflow() {
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(8, 8, 1)), b"VP8 ");
    let mut vp8x = vec![0x08, 0, 0, 0];
    vp8x.extend([0xff, 0xff, 0x0f]); // width 1,048,576
    vp8x.extend([0xff, 0x0f, 0x00]); // height 4096
    let data = riff(&[
        (*b"VP8X", vp8x),
        (*b"VP8 ", vp8),
        (*b"EXIF", b"II*\0e".to_vec()),
    ]);
    assert!(WebpDecoderConfig::new().job().probe(&data).is_err());
    let inv = inventory_of(&data);
    assert!(
        first_top_part(&inv)
            .detail
            .as_deref()
            .unwrap()
            .contains("probe() fails here")
    );
    assert_eq!(
        find(&inv, "EXIF")[0].disposition,
        Disposition::Dropped,
        "{inv}"
    );
}

/// Round 3 N2: `detect::probe` also fails when the VP8 first partition (clamped to
/// the data) is under 2 bytes.
#[test]
fn review3_n2_first_partition_under_two_bytes() {
    let mut vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    // Frame tag: bit 0 keyframe(0), bits 1-3 version, bit 4 show_frame, bits 5.. first_part_size.
    let tag = u32::from(vp8[0]) | u32::from(vp8[1]) << 8 | u32::from(vp8[2]) << 16;
    let tag = (tag & 0x1f) | (1 << 5);
    vp8[..3].copy_from_slice(&tag.to_le_bytes()[..3]);
    let simple = riff(&[(*b"VP8 ", vp8.clone()), (*b"ICCP", b"icc-last".to_vec())]);
    assert!(zenwebp::detect::probe(&simple).is_err());
    let inv = inventory_of(&simple);
    assert_eq!(
        find(&inv, "ICCP")[0].disposition,
        Disposition::Skipped,
        "{inv}"
    );

    // Two VP8 chunks in a VP8X still: decode() uses the last, detect::probe fails on the first.
    let good = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    let data = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0x20)),
        (*b"ICCP", b"icc-first".to_vec()),
        (*b"VP8 ", vp8),
        (*b"VP8 ", good),
        (*b"ICCP", b"icc-dup".to_vec()),
    ]);
    assert!(zenwebp::detect::probe(&data).is_err());
    assert!(WebpDecoderConfig::new().decode(&data).is_ok());
    let inv = inventory_of(&data);
    let iccs = find(&inv, "ICCP");
    assert_eq!(iccs[1].disposition, Disposition::Skipped, "{inv}");
}

/// Round 3 N3: a first-ANMF scan range that leaves the ANMF reaches `ImageInfo`
/// over top-level chunks and over bytes past the RIFF end.
#[test]
fn review3_n3_misread_range_leaves_the_anmf() {
    let build = |n: u32, trailing: &[u8]| {
        let mut chunks = chunks_of(&animation());
        let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
        let mut misread = b"XMP ".to_vec();
        misread.extend(n.to_le_bytes());
        let mut p = chunks[i].1[..16].to_vec();
        p.extend(chunk_bytes(b"ABCD", &misread));
        p.extend(chunk_bytes(b"ZERO", &[]));
        p.extend(&chunks[i].1[16..]);
        chunks[i].1 = p;
        chunks.insert(
            i + 1,
            (*b"ZEN3", b"owner=someone serial=42 gps=47.6N".to_vec()),
        );
        let mut d = riff(&chunks);
        d.extend(trailing);
        (d, chunks)
    };
    // Where the scan's second range starts: 16 (frame header) + 16 (ABCD) + 8 (ZERO) into the ANMF payload.
    let (probe_data, chunks) = build(0, &[]);
    let anmf_at = probe_data.windows(4).position(|w| w == b"ANMF").unwrap() as u64 + 8;
    let start = anmf_at + 40;
    let zen3_at = probe_data.windows(4).position(|w| w == b"ZEN3").unwrap() as u64;
    let zen3_end = zen3_at + 8 + chunks[2].1.len() as u64 + (chunks[2].1.len() as u64 & 1);

    // (a) the range ends exactly at the end of the private ZEN3 chunk.
    let (data, _) = build((zen3_end - start) as u32, &[]);
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    let xmp = info
        .embedded_metadata
        .xmp
        .as_deref()
        .expect("xmp from the misread");
    assert_eq!(xmp, &data[start as usize..zen3_end as usize]);
    let inv = inventory_of(&data);
    let zen3 = find(&inv, "ZEN3")[0];
    assert_eq!(
        zen3.disposition,
        Disposition::Metadata(MetadataKind::Xmp),
        "{inv}"
    );
    assert!(zen3.detail.as_deref().unwrap().contains("first-ANMF scan"));

    // (b) the range runs past the RIFF end into trailing bytes.
    // Trailing bytes that parse as a whole chunk, so the decoder's walk (which
    // overruns the RIFF end, #107) does not fail on them.
    let mut tail = b"TAIL".to_vec();
    tail.extend(24u32.to_le_bytes());
    tail.extend(b"secret-bytes-0123456789");
    tail.extend([0, 0]);
    let tail = &tail[..];
    let (data, _) = build(0, tail);
    let riff_end = data.len() as u64 - tail.len() as u64;
    let (data, _) = build((riff_end + 20 - start) as u32, tail);
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(
        info.embedded_metadata.xmp.as_deref().unwrap(),
        &data[start as usize..(riff_end + 20) as usize]
    );
    let inv = inventory_of(&data);
    let span = inv
        .parts()
        .iter()
        .find(|p| matches!(&p.tag, PartTag::Name(n) if n == "first-ANMF scan range"))
        .expect("a top-level part over the trailing bytes");
    assert_eq!(span.disposition, Disposition::Metadata(MetadataKind::Xmp));
    assert_eq!(span.range, riff_end..riff_end + 20);
    assert!(
        inv.parts()
            .iter()
            .any(|p| p.disposition == Disposition::Trailing)
    );
}

// ── Review round 4 ──────────────────────────────────────────────────────────

fn still_with_meta(meta: &[([u8; 4], Vec<u8>)], flags: u8) -> Vec<u8> {
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 3)), b"VP8 ");
    let mut cs: Vec<Chunk> = vec![(*b"VP8X", vp8x_still(16, 16, flags))];
    cs.extend(meta.iter().cloned());
    cs.push((*b"VP8 ", vp8));
    riff(&cs)
}

#[derive(Clone, Copy)]
struct AlwaysStop;
impl enough::Stop for AlwaysStop {
    fn check(&self) -> Result<(), enough::StopReason> {
        Err(enough::StopReason::Cancelled)
    }
}

fn decode_with(
    job: zenwebp::zencodec::WebpDecodeJob,
    data: &[u8],
) -> Result<zencodec::decode::DecodeOutput, String> {
    job.decoder(std::borrow::Cow::Borrowed(data), &[])
        .and_then(zencodec::decode::Decode::decode)
        .map_err(|e| e.to_string())
}

/// Q1: the job's policy, limits and stop token change the answer.
#[test]
fn review4_q1_job_settings() {
    use zencodec::decode::DecodePolicy;
    let data = still_with_meta(
        &[
            (*b"ICCP", b"icc-policy".to_vec()),
            (*b"EXIF", b"II*\0exif-policy".to_vec()),
            (*b"XMP ", b"<xmp-policy/>".to_vec()),
        ],
        0x20 | 0x08 | 0x04,
    );
    // (a) metadata policy: EXIF/XMP Dropped; the ICCP stays Metadata because
    // WebPProbe (source_encoding_details) still carries it.
    let strip = DecodePolicy::none()
        .with_allow_icc(false)
        .with_allow_exif(false)
        .with_allow_xmp(false);
    let job = || WebpDecoderConfig::new().job().with_policy(strip);
    let info = job().probe(&data).unwrap();
    assert!(info.source_color.icc_profile.is_none());
    assert!(info.embedded_metadata.exif.is_none() && info.embedded_metadata.xmp.is_none());
    assert_eq!(
        probe_details(&info).icc_profile.as_deref(),
        Some(&b"icc-policy"[..])
    );
    let inv = job().inventory(&data).unwrap().unwrap();
    inv.validate().unwrap();
    assert_eq!(
        find(&inv, "EXIF")[0].disposition,
        Disposition::Dropped,
        "{inv}"
    );
    assert_eq!(find(&inv, "XMP ")[0].disposition, Disposition::Dropped);
    assert!(
        find(&inv, "EXIF")[0]
            .detail
            .as_deref()
            .unwrap()
            .contains("DecodePolicy strips it")
    );
    let iccp = find(&inv, "ICCP")[0];
    assert_eq!(iccp.disposition, Disposition::Metadata(MetadataKind::Icc));
    assert!(iccp.detail.as_deref().unwrap().contains("WebPProbe"));

    // (b) animation denied: decode() returns frame 1, the animation decoder refuses.
    let anim = animation();
    let job = || {
        WebpDecoderConfig::new()
            .job()
            .with_policy(DecodePolicy::none().with_allow_animation(false))
    };
    assert!(
        job()
            .animation_frame_decoder(std::borrow::Cow::Borrowed(&anim[..]), &[])
            .is_err()
    );
    assert!(decode_with(job(), &anim).is_ok());
    let inv = job().inventory(&anim).unwrap().unwrap();
    inv.validate().unwrap();
    let frames: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.parent.is_some() && p.label.as_deref() == Some("VP8L"))
        .map(|p| p.disposition)
        .collect();
    assert_eq!(
        frames,
        vec![
            Disposition::ImageData,
            Disposition::Skipped,
            Disposition::Skipped
        ],
        "{inv}"
    );
    let loop_count = inv
        .parts()
        .iter()
        .find(|p| matches!(&p.tag, PartTag::Name(n) if n == "loop_count"))
        .unwrap();
    assert_eq!(loop_count.disposition, Disposition::Dropped);

    // (c) max_input_bytes: the same error as decode().
    let job = || {
        WebpDecoderConfig::new()
            .job()
            .with_limits(zencodec::ResourceLimits::none().with_max_input_bytes(64))
    };
    let dec = decode_with(job(), &data).unwrap_err();
    let inv = job().inventory(&data).unwrap_err().to_string();
    assert_eq!(inv, dec);

    // (c') max_pixels: decode() fails, probe() does not apply the job's limits.
    let job = || {
        WebpDecoderConfig::new()
            .job()
            .with_limits(zencodec::ResourceLimits::none().with_max_pixels(100))
    };
    assert!(decode_with(job(), &data).is_err());
    assert!(job().probe(&data).is_ok());
    let inv = job().inventory(&data).unwrap().unwrap();
    inv.validate().unwrap();
    let vp8 = find(&inv, "VP8 ")[0];
    assert_eq!(vp8.disposition, Disposition::Skipped, "{inv}");
    assert!(vp8.detail.as_deref().unwrap().contains("job's limits"));
    assert_eq!(
        find(&inv, "EXIF")[0].disposition,
        Disposition::Metadata(MetadataKind::Exif)
    );
    // ...and max_frames on an animation.
    let job = || {
        WebpDecoderConfig::new()
            .job()
            .with_limits(zencodec::ResourceLimits::none().with_max_frames(2))
    };
    assert!(decode_with(job(), &anim).is_err());
    let inv = job().inventory(&anim).unwrap().unwrap();
    assert!(
        !inv.parts()
            .iter()
            .any(|p| p.disposition == Disposition::ImageData),
        "{inv}"
    );

    // (d) a stopped token: inventory() fails like decode().
    let job = || {
        WebpDecoderConfig::new()
            .job()
            .with_stop(zencodec::StopToken::new(AlwaysStop))
    };
    assert!(decode_with(job(), &data).is_err());
    assert!(job().inventory(&data).is_err());
}

fn icc_with_tail(declared: u32, tail: &[u8]) -> Vec<u8> {
    let mut h = vec![0u8; 132];
    h[0..4].copy_from_slice(&declared.to_be_bytes());
    h[8..12].copy_from_slice(&[4, 0x30, 0, 0]);
    h[12..16].copy_from_slice(b"mntr");
    h[16..20].copy_from_slice(b"RGB ");
    h[20..24].copy_from_slice(b"XYZ ");
    h[36..40].copy_from_slice(b"acsp");
    h.extend(tail);
    h
}

fn field_children<'a>(
    inv: &'a Inventory,
    parent: &zencodec::inventory::Part,
) -> Vec<&'a zencodec::inventory::Part> {
    inv.parts()
        .iter()
        .filter(|p| {
            p.parent
                .and_then(|q| inv.get(q))
                .is_some_and(|q| std::ptr::eq(q, parent))
        })
        .collect()
}

/// Q2: tails the caller still receives are their own Metadata children.
#[test]
fn review4_q2_icc_and_xmp_tails() {
    let tail = b"TAIL-past-the-ICC-declared-size owner=someone serial=42".to_vec();
    let icc = icc_with_tail(132, &tail);
    let xmp = b"<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/><?xpacket end=\"w\"?>TAIL-after-the-xpacket-trailer gps=47.6N".to_vec();
    let data = still_with_meta(
        &[(*b"ICCP", icc.clone()), (*b"XMP ", xmp.clone())],
        0x20 | 0x04,
    );
    let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
    assert_eq!(info.source_color.icc_profile.as_deref(), Some(&icc[..]));
    assert_eq!(info.embedded_metadata.xmp.as_deref(), Some(&xmp[..]));
    let inv = inventory_of(&data);
    let iccp = find(&inv, "ICCP")[0];
    let kids = field_children(&inv, iccp);
    assert_eq!(kids.len(), 2, "{inv}");
    assert_eq!(kids[0].range.end - kids[0].range.start, 132);
    assert_eq!(kids[1].range.end - kids[1].range.start, tail.len() as u64);
    assert!(
        kids.iter()
            .all(|k| k.disposition == Disposition::Metadata(MetadataKind::Icc))
    );
    assert!(kids[1].detail.as_deref().unwrap().contains("declared size"));
    let xmpp = find(&inv, "XMP ")[0];
    let kids = field_children(&inv, xmpp);
    assert_eq!(kids.len(), 2, "{inv}");
    let cut = xmp.windows(2).position(|w| w == b"?>").map(|_| ()).and(
        xmp.windows(14)
            .position(|w| w == b"<?xpacket end=")
            .map(|i| i + xmp[i..].windows(2).position(|w| w == b"?>").unwrap() + 2),
    );
    assert_eq!(kids[0].range.end - kids[0].range.start, cut.unwrap() as u64);
    assert!(
        kids.iter()
            .all(|k| k.disposition == Disposition::Metadata(MetadataKind::Xmp))
    );
    // Without a packet wrapper the end is </x:xmpmeta>; EXIF stays whole.
    let bare = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>trailing-note".to_vec();
    let data = still_with_meta(
        &[
            (*b"XMP ", bare.clone()),
            (*b"EXIF", b"II*\0exif-and-more".to_vec()),
        ],
        0x04 | 0x08,
    );
    let inv = inventory_of(&data);
    let kids = field_children(&inv, find(&inv, "XMP ")[0]);
    assert_eq!(
        kids[1].range.end - kids[1].range.start,
        b"trailing-note".len() as u64
    );
    assert!(field_children(&inv, find(&inv, "EXIF")[0]).is_empty());
    // A profile whose declared size matches the payload is not split.
    let data = still_with_meta(&[(*b"ICCP", icc_with_tail(132, &[]))], 0x20);
    let inv = inventory_of(&data);
    assert!(field_children(&inv, find(&inv, "ICCP")[0]).is_empty());
}

/// Q4: a misread range that covers part of a leaf marks only that part.
#[test]
fn review4_q4_partial_cover() {
    let secret = b"owner=someone serial=42 gps=47.6N rest-is-not-delivered".to_vec();
    for (cover, whole) in [(8 + 10u32, false), (8 + secret.len() as u32 + 1, true)] {
        let mut chunks = chunks_of(&animation());
        let i = chunks.iter().position(|c| &c.0 == b"ANMF").unwrap();
        let mut misread = b"XMP ".to_vec();
        misread.extend(cover.to_le_bytes());
        let mut p = chunks[i].1[..16].to_vec();
        p.extend(chunk_bytes(b"ABCD", &misread));
        p.extend(chunk_bytes(b"ZERO", &[]));
        chunks[i].1 = p;
        chunks.insert(i + 1, (*b"ZEN3", secret.clone()));
        let data = riff(&chunks);
        let info = WebpDecoderConfig::new().job().probe(&data).unwrap();
        assert_eq!(
            info.embedded_metadata.xmp.as_ref().map(|v| v.len()),
            Some(cover as usize)
        );
        let inv = inventory_of(&data);
        let z = find(&inv, "ZEN3")[0];
        if whole {
            assert_eq!(
                z.disposition,
                Disposition::Metadata(MetadataKind::Xmp),
                "{inv}"
            );
        } else {
            assert_eq!(z.disposition, Disposition::Unknown, "{inv}");
            let kids = field_children(&inv, z);
            assert_eq!(kids.len(), 1, "{inv}");
            assert_eq!(
                kids[0].disposition,
                Disposition::Metadata(MetadataKind::Xmp)
            );
            assert_eq!(kids[0].range.end - kids[0].range.start, u64::from(cover));
        }
    }
}

/// Q3: bytes other readers read carry a note (or a small Structure child).
#[test]
fn review4_q3_other_readers() {
    // (b) a second VP8X: detect::probe reads its 10 bytes.
    let vp8 = chunk_payload(&chunks_of(&lossy_seed(16, 16, 2)), b"VP8 ");
    let data = riff(&[
        (*b"VP8X", vp8x_still(16, 16, 0)),
        (*b"VP8 ", vp8.clone()),
        (*b"VP8X", vec![0x10, 0, 0, 0, 0xff, 0x0f, 0, 0xff, 0x0f, 0]),
    ]);
    let inv = inventory_of(&data);
    let second = find(&inv, "VP8X")[1];
    let kids = field_children(&inv, second);
    assert_eq!(kids.len(), 1, "{inv}");
    assert_eq!(kids[0].disposition, Disposition::Structure);
    assert_eq!(kids[0].range.end - kids[0].range.start, 10);
    // ... also after a simple image.
    let data = riff(&[
        (*b"VP8 ", vp8.clone()),
        (*b"VP8X", vec![0x10, 0, 0, 0, 7, 0, 0, 7, 0, 0]),
    ]);
    let inv = inventory_of(&data);
    assert_eq!(
        field_children(&inv, find(&inv, "VP8X")[0]).len(),
        1,
        "{inv}"
    );

    // (c) trailing bytes after a VP8X file name the readers.
    let mut data = riff(&[(*b"VP8X", vp8x_still(16, 16, 0)), (*b"VP8 ", vp8.clone())]);
    data.extend(b"ZZZZ\x04\0\0\0abcd");
    let inv = inventory_of(&data);
    let last = inv.get(inv.children(None).pop().unwrap()).unwrap();
    assert_eq!(last.disposition, Disposition::Trailing);
    let d = last.detail.as_deref().unwrap();
    assert!(
        d.contains("read_data") && d.contains("detect::probe"),
        "{d}"
    );

    // (d) a truncated simple VP8L: the 5-byte header is Structure.
    let mut ll = riff(&[(*b"VP8L", lossless_rgba_vp8l(16, 16, 1))]);
    ll.truncate(ll.len() - 2);
    let inv = inventory_of(&ll);
    let img = find(&inv, "VP8L")[0];
    assert_eq!(img.disposition, Disposition::Malformed);
    let kids = field_children(&inv, img);
    assert_eq!(kids.len(), 1, "{inv}");
    assert_eq!(kids[0].range.end - kids[0].range.start, 5);
    assert_eq!(kids[0].disposition, Disposition::Structure);

    // (g) a simple VP8 chunk shorter than its 10-byte frame header.
    // The chunk declares 5 payload bytes; the frame header continues past it
    // (no pad byte), so read_data still finds a valid header in the file.
    let mut body = b"WEBPVP8 ".to_vec();
    body.extend(5u32.to_le_bytes());
    body.extend(&vp8[..12]);
    let mut short = b"RIFF".to_vec();
    short.extend((body.len() as u32).to_le_bytes());
    short.extend(body);
    assert!(WebpDecoderConfig::new().job().probe(&short).is_ok());
    let inv = inventory_of(&short);
    assert!(
        find(&inv, "VP8 ")[0]
            .detail
            .as_deref()
            .unwrap()
            .contains("10-byte frame header from the file"),
        "{inv}"
    );

    // (e)(f) ALPH notes on the demuxer path.
    let (la, lb) = (lossy_alpha_seed(32, 24, 0), lossy_alpha_seed(32, 24, 0x5a));
    let (ca, cb) = (chunks_of(&la), chunks_of(&lb));
    let two = riff(&[
        (*b"VP8X", chunk_payload(&ca, b"VP8X")),
        (*b"ALPH", chunk_payload(&ca, b"ALPH")),
        (*b"ALPH", chunk_payload(&cb, b"ALPH")),
        (*b"VP8 ", chunk_payload(&ca, b"VP8 ")),
    ]);
    let inv = inventory_of(&two);
    let alph = find(&inv, "ALPH");
    assert!(
        alph[0]
            .detail
            .as_deref()
            .unwrap()
            .contains("falls back to WebPDecoder")
    );
    let mut cleared = two.clone();
    cleared[20] &= !0x10;
    let inv = inventory_of(&cleared);
    let last = find(&inv, "ALPH").into_iter().next_back().unwrap();
    assert_eq!(last.disposition, Disposition::Dropped);
    assert!(
        last.detail
            .as_deref()
            .unwrap()
            .contains("a corrupt ALPH makes decode() fail")
    );
}
