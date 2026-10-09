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
    let alph = find(&inv, "ALPH")[0];
    assert_eq!(alph.disposition, Disposition::Skipped, "{inv}");
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
    assert_ne!(
        anmf[0]
            .detail
            .as_deref()
            .unwrap_or("")
            .contains("unreachable"),
        true
    );
    for later in &anmf[1..] {
        assert!(
            later.detail.as_deref().unwrap().contains("unreachable"),
            "{inv}"
        );
    }
    let skipped_images = inv
        .parts()
        .iter()
        .filter(|p| p.parent.is_some() && p.label.as_deref() == Some("VP8L"))
        .filter(|p| p.disposition == Disposition::Skipped)
        .count();
    assert_eq!(skipped_images, 2, "{inv}");
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
    let expected = r####"Header "RIFF" 0..12 structure label="RIFF/WEBP" detail="riff_size=174; 4 bytes follow the declared RIFF end at 182"
Chunk "VP8X" 12..30 structure label="VP8X" detail="flags=0x3e (icc=1 alpha=1 exif=1 xmp=1 anim=1), canvas 4x4; reserved bits are ignored"
  Field Name("flags+canvas") 20..30 structure
Chunk "ICCP" 30..42 metadata(icc) label="ICCP" detail="first occurrence"
Chunk "ANIM" 42..56 structure label="ANIM"
  Field Name("background_color") 50..54 dropped detail="native WebPDecoder::background_color_hint only; the zencodec path does not report it"
  Field Name("loop_count") 54..56 metadata(animation)
Chunk "ANMF" 56..124 structure label="ANMF"
  Header Name("frame header") 64..80 metadata(animation) detail="frame x/y/size, duration, blend and dispose flags"
  Chunk "ALPH" 80..92 image-data label="ALPH" detail="frame alpha plane"
  Chunk "VP8 " 92..104 image-data label="VP8 " detail="frame image"
  Chunk "EXIF" 104..114 skipped label="EXIF" detail="read_frame reads only the frame's image chunk(s)"
  Chunk "QQQQ" 114..124 unknown label="QQQQ" detail="unrecognised chunk type; payload starts \"hi\""
Chunk "VP8 " 124..134 skipped label="VP8 " detail="animated file: frames are read from ANMF, not from top-level image chunks"
Chunk "ZEN1" 134..147 unknown label="ZEN1" detail="unrecognised chunk type; payload starts \"hello\""
Gap None 147..148 padding detail="non-zero RIFF pad byte 0x41"
Chunk "EXIF" 148..160 metadata(exif) label="EXIF" detail="first occurrence"
Chunk "XMP " 160..172 metadata(xmp) label="XMP " detail="first occurrence"
Chunk "ICCP" 172..182 skipped label="ICCP" detail="duplicate chunk: the decoder keeps the first one"
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
        if sub.len() % 2 == 0 {
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
