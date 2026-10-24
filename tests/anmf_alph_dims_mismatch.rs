//! Regression for the animated ALPH-frame dimension mismatch (6c63e9e on main).
//!
//! `read_frame`'s ALPH arm sized the alpha plane from the ANMF frame
//! dimensions but reconstructed over the embedded VP8's own dimensions. An
//! animation whose ANMF frame declares 4x4 while its VP8 keyframe is 8x8
//! indexed the alpha plane out of bounds and panicked. It must now return an
//! error instead.

fn chunk(data: &[u8], fourcc: &[u8; 4]) -> Vec<u8> {
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = pos + 8 + size + (size & 1);
        if &data[pos..pos + 4] == fourcc {
            return data[pos..end.min(data.len())].to_vec();
        }
        pos = end;
    }
    panic!("no {} chunk", String::from_utf8_lossy(fourcc));
}

fn u24(v: u32) -> [u8; 3] {
    [v as u8, (v >> 8) as u8, (v >> 16) as u8]
}

/// An 8x8 animation canvas holding one ANMF frame that declares
/// `anmf_w`x`anmf_h` and carries a raw (uncompressed) `anmf_w`x`anmf_h` ALPH
/// chunk plus an 8x8 VP8 keyframe. The raw alpha is sized to the ANMF dims so
/// it passes `read_alpha_chunk`; only the VP8 disagrees.
fn mismatched_animation(anmf_w: u32, anmf_h: u32) -> Vec<u8> {
    let (w, h) = (8u32, 8u32);
    let rgba: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            [
                (i * 4) as u8,
                (i * 3) as u8,
                (i * 7) as u8,
                (i * 5 + 1) as u8,
            ]
        })
        .collect();
    let cfg = zenwebp::LossyConfig::new();
    let still = zenwebp::EncodeRequest::lossy(&cfg, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    // ALPH header byte 0: no preprocessing, no filtering, raw compression.
    let n_alpha = (anmf_w * anmf_h) as usize;
    let mut alph = Vec::new();
    alph.extend_from_slice(b"ALPH");
    alph.extend_from_slice(&((1 + n_alpha) as u32).to_le_bytes());
    alph.push(0);
    alph.extend((0..n_alpha).map(|i| (i * 13) as u8));
    if (1 + n_alpha) % 2 == 1 {
        alph.push(0);
    }
    let vp8 = chunk(&still, b"VP8 ");

    let mut anmf_payload = Vec::new();
    anmf_payload.extend_from_slice(&u24(0)); // x / 2
    anmf_payload.extend_from_slice(&u24(0)); // y / 2
    anmf_payload.extend_from_slice(&u24(anmf_w - 1));
    anmf_payload.extend_from_slice(&u24(anmf_h - 1));
    anmf_payload.extend_from_slice(&u24(100)); // duration
    anmf_payload.push(0); // blend/dispose flags
    anmf_payload.extend_from_slice(&alph);
    anmf_payload.extend_from_slice(&vp8);

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF\0\0\0\0WEBP");
    out.extend_from_slice(b"VP8X");
    out.extend_from_slice(&10u32.to_le_bytes());
    out.push(0x02 | 0x10); // animation + alpha
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&u24(w - 1));
    out.extend_from_slice(&u24(h - 1));
    out.extend_from_slice(b"ANIM");
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    out.extend_from_slice(b"ANMF");
    out.extend_from_slice(&(anmf_payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&anmf_payload);
    if anmf_payload.len() % 2 == 1 {
        out.push(0);
    }
    let riff = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff.to_le_bytes());
    out
}

#[test]
fn smaller_anmf_than_vp8_is_an_error_not_a_panic() {
    let data = mismatched_animation(4, 4);

    let mut decoder = zenwebp::WebPDecoder::new(&data).expect("container parses");
    assert!(decoder.is_animated());
    let (cw, ch) = decoder.dimensions();
    let mut buf = vec![0u8; (cw * ch * 4) as usize];
    assert!(
        decoder.read_frame(&mut buf).is_err(),
        "ANMF 4x4 holding an 8x8 VP8 must be rejected"
    );

    let mut anim = zenwebp::mux::AnimationDecoder::new(&data).expect("container parses");
    assert!(anim.next_frame().is_err());
}

#[test]
fn matching_dims_still_decode() {
    let data = mismatched_animation(8, 8);
    let mut anim = zenwebp::mux::AnimationDecoder::new(&data).expect("container parses");
    let frame = anim.next_frame().expect("decodes").expect("one frame");
    assert_eq!(frame.width * frame.height * 4, frame.data.len() as u32);
}
