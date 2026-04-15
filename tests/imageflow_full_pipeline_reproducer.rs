//! Reproduce the imageflow `test_transparent_webp_to_webp` divergence by
//! mimicking the full pipeline: decode 1_webp_ll.webp, (no resize — the
//! `_native` variant of the failing imageflow test), re-encode via the
//! zencodec dyn-dispatch path with BGRA8_SRGB, and decode back for comparison.
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zencodec::encode::{DynEncodeJob as _, DynEncoder as _, DynEncoderConfig as _};
use zenpixels::{PixelDescriptor, PixelSlice};

fn rgba_to_bgra_inplace(bytes: &mut [u8]) {
    for p in bytes.chunks_exact_mut(4) {
        p.swap(0, 2);
    }
}

#[test]
fn reencode_rose_lossless_bgra_dispatch() {
    let data = std::fs::read("tests/images/gallery2/1_webp_ll.webp").expect("read rose");
    let (mut rgba, w, h) = zenwebp::oneshot::decode_rgba(&data).expect("decode rose");
    eprintln!("decoded rose: {w}x{h}, {} bytes", rgba.len());

    // Sample: what does the decoded image look like? Pick a pixel we expect
    // to be inside the rose (reddish/pink petal) — the rose covers the upper
    // portion. Pixel (200, 60) is in the petal region.
    let probe_ix = (60 * w as usize + 200) * 4;
    let probe = &rgba[probe_ix..probe_ix + 4];
    eprintln!(
        "RGBA[200,60] after decode: R={} G={} B={} A={}",
        probe[0], probe[1], probe[2], probe[3]
    );

    // Convert to BGRA in-place (the way imageflow's bitmaps store pixels).
    let mut bgra = rgba.clone();
    rgba_to_bgra_inplace(&mut bgra);
    let probe_bgra = &bgra[probe_ix..probe_ix + 4];
    eprintln!(
        "BGRA[200,60] (imageflow bitmap layout): B={} G={} R={} A={}",
        probe_bgra[0], probe_bgra[1], probe_bgra[2], probe_bgra[3]
    );

    // Now replicate imageflow's zen_encoder path: pass BGRA8_SRGB via PixelSlice,
    // dispatch through dyn_job().into_encoder().encode(ps).
    let config = zenwebp::zencodec::WebpEncoderConfig::lossless().with_quality(85.0);
    let job = config.dyn_job();
    let encoder = job.into_encoder().expect("into_encoder");
    let desc = PixelDescriptor::BGRA8_SRGB;
    let stride = w as usize * 4;
    let ps = PixelSlice::new(&bgra, w, h, stride, desc).expect("bgra pixel slice");
    let output = encoder.encode(ps).expect("encode bgra");
    let webp = output.data();

    // Decode the re-encoded output back to RGBA and compare to original.
    let (redecoded, dw, dh) = zenwebp::oneshot::decode_rgba(webp).expect("decode output");
    assert_eq!((dw, dh), (w, h));

    let probe_re = &redecoded[probe_ix..probe_ix + 4];
    eprintln!(
        "RGBA[200,60] after re-encode+decode: R={} G={} B={} A={}",
        probe_re[0], probe_re[1], probe_re[2], probe_re[3]
    );

    // Compare: expect byte-exact on visible pixels, α=0 may have zeroed RGB.
    let mut mismatch = 0usize;
    let mut total_dr: i64 = 0;
    let mut total_dg: i64 = 0;
    let mut total_db: i64 = 0;
    let mut max_d = [0u16; 4];
    let mut first_diff: Option<(usize, [u8; 4], [u8; 4])> = None;

    for i in 0..(w * h) as usize {
        let src = &rgba[i * 4..i * 4 + 4];
        let dst = &redecoded[i * 4..i * 4 + 4];
        if src[3] == 0 {
            continue; // exact=false may zero RGB under transparent
        }
        for c in 0..4 {
            let d = (src[c] as i16 - dst[c] as i16).unsigned_abs();
            max_d[c] = max_d[c].max(d);
        }
        total_dr += (src[0] as i64 - dst[0] as i64).abs();
        total_dg += (src[1] as i64 - dst[1] as i64).abs();
        total_db += (src[2] as i64 - dst[2] as i64).abs();
        if src != dst {
            mismatch += 1;
            if first_diff.is_none() {
                first_diff = Some((
                    i,
                    [src[0], src[1], src[2], src[3]],
                    [dst[0], dst[1], dst[2], dst[3]],
                ));
            }
        }
    }

    eprintln!(
        "total pixels diff: {}, maxΔ R={} G={} B={} A={}, sumΔ R={} G={} B={}",
        mismatch, max_d[0], max_d[1], max_d[2], max_d[3], total_dr, total_dg, total_db
    );
    if let Some((ix, s, d)) = first_diff {
        eprintln!("first differing pixel @{ix}: src={s:?} dst={d:?}");
    }

    // Bit-exact lossless must hold for α>0 pixels.
    assert_eq!(
        mismatch, 0,
        "Re-encoded rose diverged from source: {mismatch} visible pixels differ, \
         maxΔ[R={} G={} B={}]",
        max_d[0], max_d[1], max_d[2]
    );
}

#[test]
fn reencode_rose_lossless_rgba_direct() {
    // Control: same rose, but via EncodeRequest::lossless RGBA — should pass
    // (this is what the golden test exercises).
    let data = std::fs::read("tests/images/gallery2/1_webp_ll.webp").expect("read rose");
    let (rgba, w, h) = zenwebp::oneshot::decode_rgba(&data).expect("decode rose");

    use zenwebp::{EncodeRequest, PixelLayout};
    let cfg = zenwebp::LosslessConfig::new();
    let webp = EncodeRequest::lossless(&cfg, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .expect("encode rgba");
    let (redecoded, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).expect("decode output");
    assert_eq!((dw, dh), (w, h));

    let mut mismatch = 0usize;
    for i in 0..(w * h) as usize {
        let src = &rgba[i * 4..i * 4 + 4];
        let dst = &redecoded[i * 4..i * 4 + 4];
        if src[3] == 0 {
            continue;
        }
        if src != dst {
            mismatch += 1;
        }
    }
    assert_eq!(
        mismatch, 0,
        "RGBA direct rose roundtrip drifted ({mismatch} pixels)"
    );
}
