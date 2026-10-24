//! RIFF container walk regressions (#107, #108, #109), each checked against
//! libwebp (via webpx) as the oracle.
//!
//! - #107: the VP8X chunk walk ran ten bytes past the RIFF end, so a chunk
//!   header after the declared RIFF size was parsed (and trailing data failed
//!   the decode).
//! - #108: `read_frame` stepped to the next frame by the unpadded ANMF size.
//! - #109: a non-ANMF chunk between frames ended the frame chain.
//!
//! Fixtures are built from encoder output and hand-assembled chunks; nothing
//! binary is committed.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder};
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, WebPDecoder};

type Chunk = ([u8; 4], Vec<u8>);

/// Top-level chunks of a well-formed RIFF/WEBP file.
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

/// Deterministic RGBA texture; `alpha` false gives an opaque image.
fn pattern(w: u32, h: u32, seed: u32, alpha: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let n =
                (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ seed.wrapping_mul(97))
                    >> 9;
            v.extend_from_slice(&[
                ((x * 255 / w) as u8).wrapping_add((n & 31) as u8),
                ((y * 255 / h) as u8).wrapping_add((n >> 5 & 31) as u8),
                (n >> 10) as u8,
                if alpha {
                    ((x * 7 + y * 3 + seed) % 200 + 40) as u8
                } else {
                    255
                },
            ]);
        }
    }
    v
}

/// Three frames alternating lossless and lossy, so the ANMF sub-chunks have
/// mixed sizes; `alpha` adds ALPH chunks to the lossy frames.
fn animation(alpha: bool) -> Vec<u8> {
    let (w, h) = (24, 20);
    let mut enc = AnimationEncoder::new(
        w,
        h,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    for i in 0..3u32 {
        let cfg = if i % 2 == 0 {
            EncoderConfig::new_lossless()
        } else {
            EncoderConfig::new_lossy()
        };
        enc.add_frame(
            &pattern(w, h, i + 1, alpha),
            PixelLayout::Rgba8,
            i * 80,
            &cfg,
        )
        .unwrap();
    }
    enc.finalize(80).unwrap()
}

/// Every frame from zenwebp and libwebp, as RGBA, and require equality.
/// Returns zenwebp's frames.
fn frames_match_libwebp(label: &str, data: &[u8]) -> Vec<Vec<u8>> {
    let lib = webpx::AnimationDecoder::new(data)
        .unwrap_or_else(|e| panic!("{label}: libwebp rejects the fixture: {e:?}"))
        .decode_all()
        .unwrap_or_else(|e| panic!("{label}: libwebp decode_all: {e:?}"));
    let mut dec = AnimationDecoder::new(data).unwrap();
    let has_alpha = dec.info().has_alpha;
    let zen = dec
        .decode_all()
        .unwrap_or_else(|e| panic!("{label}: zenwebp decode_all: {e:?}"));
    assert_eq!(zen.len(), lib.len(), "{label}: frame count");
    zen.iter()
        .zip(&lib)
        .enumerate()
        .map(|(i, (z, l))| {
            let z4: Vec<u8> = if has_alpha {
                z.data.clone()
            } else {
                z.data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect()
            };
            assert!(z4 == l.data, "{label}: frame {i} differs from libwebp");
            assert_eq!(z.duration_ms, l.duration_ms, "{label}: frame {i} duration");
            z4
        })
        .collect()
}

/// Native `read_frame` loop, then `reset_animation` and a second pass: both
/// must yield `expect`.
fn read_frame_twice(label: &str, data: &[u8], expect: &[Vec<u8>]) {
    let mut dec = WebPDecoder::new(data).unwrap();
    assert_eq!(
        dec.num_frames() as usize,
        expect.len(),
        "{label}: num_frames"
    );
    let mut buf = vec![0u8; dec.output_buffer_size().unwrap()];
    for pass in 0..2 {
        for (i, want) in expect.iter().enumerate() {
            dec.read_frame(&mut buf)
                .unwrap_or_else(|e| panic!("{label}: pass {pass} frame {i}: {e:?}"));
            let got: Vec<u8> = if dec.has_alpha() {
                buf.clone()
            } else {
                buf.as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect()
            };
            assert!(&got == want, "{label}: pass {pass} frame {i} differs");
        }
        dec.reset_animation().unwrap();
    }
}

/// Rewrite every ANMF so its last sub-chunk has an odd size and no padding of
/// its own: the ANMF size is then odd and the ANMF's pad byte pads both.
fn with_odd_anmfs(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (f, p) in chunks_of(data) {
        if &f != b"ANMF" {
            out.push((f, p));
            continue;
        }
        let subs = chunks_of_body(&p[16..]);
        let (last, init) = subs.split_last().unwrap();
        let mut payload = p[..16].to_vec();
        for (sf, sp) in init {
            payload.extend(chunk_bytes(sf, sp));
        }
        let mut last_payload = last.1.clone();
        if last_payload.len() % 2 == 0 {
            // Trailing bytes after a VP8/VP8L bitstream are ignored by both
            // decoders; one makes the size odd.
            last_payload.push(0);
        }
        payload.extend(&last.0);
        payload.extend((last_payload.len() as u32).to_le_bytes());
        payload.extend(&last_payload);
        assert_eq!(payload.len() % 2, 1);
        out.push((f, payload));
    }
    riff(&out)
}

/// Sub-chunks of an ANMF body (after the 16-byte frame header).
fn chunks_of_body(body: &[u8]) -> Vec<Chunk> {
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

/// Insert `extra` before every ANMF after the first.
fn with_chunks_between_frames(data: &[u8], extra: &Chunk) -> Vec<u8> {
    let mut out = Vec::new();
    let mut seen = 0;
    for c in chunks_of(data) {
        if &c.0 == b"ANMF" {
            if seen > 0 {
                out.push(extra.clone());
            }
            seen += 1;
        }
        out.push(c);
    }
    riff(&out)
}

/// #108: an ANMF with an odd size is followed by a pad byte; the next frame
/// starts after it.
#[test]
fn odd_sized_anmf_frames_all_decode() {
    for alpha in [false, true] {
        let base = animation(alpha);
        let expect = frames_match_libwebp("base", &base);
        let odd = with_odd_anmfs(&base);
        let label = format!("odd ANMF (alpha={alpha})");
        assert_eq!(frames_match_libwebp(&label, &odd), expect, "{label}");
        read_frame_twice(&label, &odd, &expect);
    }
}

/// #109: chunks between ANMF frames (unknown, odd-sized, or a known type that
/// has no meaning there) are skipped, as libwebp's demuxer does.
#[test]
fn non_anmf_chunks_between_frames_are_skipped() {
    for alpha in [false, true] {
        let base = animation(alpha);
        let expect = frames_match_libwebp("base", &base);
        for extra in [
            (*b"JUNK", vec![1, 2, 3, 4]),
            (*b"ODD!", vec![9, 9, 9]),
            (*b"ZERO", vec![]),
        ] {
            let data = with_chunks_between_frames(&base, &extra);
            let label = format!(
                "{} between frames (alpha={alpha})",
                String::from_utf8_lossy(&extra.0)
            );
            assert_eq!(frames_match_libwebp(&label, &data), expect, "{label}");
            read_frame_twice(&label, &data, &expect);
        }
        // Both quirks at once.
        let data = with_chunks_between_frames(&with_odd_anmfs(&base), &(*b"ODD!", vec![7; 5]));
        assert_eq!(frames_match_libwebp("odd + junk", &data), expect);
        read_frame_twice("odd + junk", &data, &expect);
    }
}

/// VP8X still with ICCP + EXIF + XMP, lossy with alpha.
fn extended_still() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (24, 20);
    let px = pattern(w, h, 5, true);
    let plain = EncodeRequest::new(&EncoderConfig::new_lossy(), &px, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    let chunks = chunks_of(&plain);
    assert_eq!(&chunks[0].0, b"VP8X");
    let icc = b"icc-profile-bytes".to_vec();
    let exif = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
    let xmp = b"<x:xmpmeta/>".to_vec();
    let mut vp8x = chunks[0].1.clone();
    vp8x[0] |= 0x20 | 0x08 | 0x04;
    let mut out = vec![(*b"VP8X", vp8x), (*b"ICCP", icc.clone())];
    out.extend(chunks[1..].iter().cloned());
    out.push((*b"EXIF", exif.clone()));
    out.push((*b"XMP ", xmp.clone()));
    (riff(&out), icc, exif, xmp)
}

/// #107: data after the RIFF end is not part of the file. It must neither
/// fail the decode nor be read as a chunk.
#[test]
fn trailing_data_after_vp8x_file_is_ignored() {
    let (still, icc, exif, xmp) = extended_still();
    let anim = animation(true);
    let (want_px, ..) = webpx::decode_rgba(&still).unwrap();
    let want_frames = frames_match_libwebp("anim", &anim);
    for trailer in [
        b"trailing junk, longer than a chunk header".to_vec(),
        // A chunk header whose size runs far past the end of the data.
        chunk_bytes(b"JUNK", &[])
            .iter()
            .take(4)
            .copied()
            .chain(0xFFFF_FFF0u32.to_le_bytes())
            .collect(),
        vec![0u8; 8],
        vec![0u8; 17],
        // A well-formed chunk straight after the RIFF end.
        chunk_bytes(b"ICCP", b"not-the-profile"),
    ] {
        let mut data = still.clone();
        data.extend(&trailer);
        let (px, ..) = zenwebp::decoder::decode_rgba(&data)
            .unwrap_or_else(|e| panic!("still + {} trailing bytes: {e:?}", trailer.len()));
        assert!(
            px == want_px,
            "still + {} trailing bytes: pixels",
            trailer.len()
        );
        let mut dec = WebPDecoder::new(&data).unwrap();
        assert_eq!(dec.icc_profile().unwrap(), Some(icc.clone()));
        assert_eq!(dec.exif_metadata().unwrap(), Some(exif.clone()));
        assert_eq!(dec.xmp_metadata().unwrap(), Some(xmp.clone()));
        assert_eq!(webpx::get_icc_profile(&data).unwrap(), Some(icc.clone()));

        let mut data = anim.clone();
        data.extend(&trailer);
        assert_eq!(frames_match_libwebp("anim + trailer", &data), want_frames);
        read_frame_twice("anim + trailer", &data, &want_frames);
    }
}

/// #107: a chunk that starts at the declared RIFF end is outside the file,
/// even when it would otherwise be the file's only EXIF chunk.
#[test]
fn chunk_after_riff_end_is_not_read() {
    let (still, ..) = extended_still();
    let mut chunks: Vec<Chunk> = chunks_of(&still)
        .into_iter()
        .filter(|c| !matches!(&c.0, b"EXIF" | b"XMP " | b"ICCP"))
        .collect();
    chunks[0].1[0] &= !(0x20 | 0x08 | 0x04);
    let mut data = riff(&chunks);
    let riff_len = data.len();
    data.extend(chunk_bytes(b"EXIF", b"II*\0\x08\0\0\0outside"));
    data.extend(chunk_bytes(b"ICCP", b"outside"));
    assert_eq!(webpx::get_exif(&data).unwrap(), None, "libwebp oracle");
    assert_eq!(
        webpx::get_icc_profile(&data).unwrap(),
        None,
        "libwebp oracle"
    );
    let mut dec = WebPDecoder::new(&data).unwrap();
    assert_eq!(dec.exif_metadata().unwrap(), None);
    assert_eq!(dec.icc_profile().unwrap(), None);
    let (px, ..) = zenwebp::decoder::decode_rgba(&data).unwrap();
    let (want, ..) = zenwebp::decoder::decode_rgba(&data[..riff_len]).unwrap();
    assert!(px == want);

    // A RIFF size that ends mid-file: the chunks after it are not counted
    // as frames either.
    let anim = animation(false);
    let mut chunks = chunks_of(&anim);
    let last = chunks.pop().unwrap();
    assert_eq!(&last.0, b"ANMF");
    let mut data = riff(&chunks);
    data.extend(chunk_bytes(&last.0, &last.1));
    let dec = WebPDecoder::new(&data).unwrap();
    assert_eq!(dec.num_frames(), 2);
    let lib = webpx::AnimationDecoder::new(&data)
        .unwrap()
        .decode_all()
        .unwrap();
    assert_eq!(lib.len(), 2, "libwebp oracle");
    frames_match_libwebp("riff ends before frame 3", &data);
}
