//! Golden codec hashes, verified against libwebp.
//!
//! For every case in `tests/golden/cases.rs`: encode with zenwebp, decode the
//! bitstream with zenwebp AND libwebp in every mode, require identical
//! per-mode digests, then compare the encoded-bytes hash and folded decode
//! hash with `tests/golden/codec_golden.tsv`.
//!
//! `ZENWEBP_GOLDEN_BLESS=1` rewrites the golden file instead of comparing —
//! only after the libwebp check passed for every case, so a blessed decode
//! hash is always libwebp's output. The encode hash pins zenwebp's encoder
//! (no libwebp equivalent); review encoder changes before blessing.
//!
//! The lib unit test `golden_tests::golden_codec_hashes` asserts the same file
//! on every target (wasm32 included) without libwebp.
#![cfg(not(target_arch = "wasm32"))]

extern crate alloc;

#[path = "golden/cases.rs"]
mod cases;

use cases::{Case, Kind};

fn pack_yuv(p: &webpx::YuvPlanes) -> Vec<u8> {
    let (w, h) = (p.width as usize, p.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for r in 0..h {
        out.extend_from_slice(&p.y[r * p.y_stride..][..w]);
    }
    for r in 0..ch {
        out.extend_from_slice(&p.u[r * p.u_stride..][..cw]);
    }
    for r in 0..ch {
        out.extend_from_slice(&p.v[r * p.v_stride..][..cw]);
    }
    out
}

/// libwebp `WebPDecode` in `mode`'s colourspace / options, copied out tightly.
fn lib_decode(mode: &str, data: &[u8]) -> (Vec<u8>, u32, u32) {
    use libwebp_sys::*;
    match mode {
        "yuv" => {
            let p = webpx::decode_yuv(data).unwrap();
            return (pack_yuv(&p), p.width, p.height);
        }
        "anim" => {
            let frames = webpx::AnimationDecoder::new(data)
                .unwrap()
                .decode_all()
                .unwrap();
            let (w, h) = (frames[0].width, frames[0].height);
            let all: Vec<u8> = frames.iter().flat_map(|f| f.data.iter().copied()).collect();
            return (all, w, h * frames.len() as u32);
        }
        _ => {}
    }
    let (csp, bpp, no_fancy, dither) = match mode {
        "rgba" => (WEBP_CSP_MODE::MODE_RGBA, 4, 0, 0),
        "rgb" => (WEBP_CSP_MODE::MODE_RGB, 3, 0, 0),
        "bgra" => (WEBP_CSP_MODE::MODE_BGRA, 4, 0, 0),
        "bgr" => (WEBP_CSP_MODE::MODE_BGR, 3, 0, 0),
        "argb" => (WEBP_CSP_MODE::MODE_ARGB, 4, 0, 0),
        "rgbA" => (WEBP_CSP_MODE::MODE_rgbA, 4, 0, 0),
        "bgrA" => (WEBP_CSP_MODE::MODE_bgrA, 4, 0, 0),
        "Argb" => (WEBP_CSP_MODE::MODE_Argb, 4, 0, 0),
        "rgb565" => (WEBP_CSP_MODE::MODE_RGB_565, 2, 0, 0),
        "rgba4444" => (WEBP_CSP_MODE::MODE_RGBA_4444, 2, 0, 0),
        "rgba_nofancy" => (WEBP_CSP_MODE::MODE_RGBA, 4, 1, 0),
        "rgba_dither50" => (WEBP_CSP_MODE::MODE_RGBA, 4, 0, 50),
        other => panic!("unknown mode {other}"),
    };
    // SAFETY: standard libwebp advanced-API use; the libwebp-owned output is
    // read within its stride/height, then freed.
    unsafe {
        let mut config = WebPDecoderConfig::new().unwrap();
        config.options.no_fancy_upsampling = no_fancy;
        config.options.dithering_strength = dither;
        config.output.colorspace = csp;
        let st = WebPDecode(data.as_ptr(), data.len(), &mut config);
        assert_eq!(st, VP8StatusCode::VP8_STATUS_OK, "libwebp {mode}");
        let b = &config.output.u.RGBA;
        let (w, h) = (config.output.width as usize, config.output.height as usize);
        let mut px = Vec::with_capacity(w * h * bpp);
        for y in 0..h {
            px.extend_from_slice(std::slice::from_raw_parts(
                b.rgba.add(y * b.stride as usize),
                w * bpp,
            ));
        }
        WebPFreeDecBuffer(&mut config.output);
        (px, w as u32, h as u32)
    }
}

#[test]
fn golden_codec_hashes_match_libwebp() {
    let all: Vec<Case> = cases::cases();
    let mut lines = vec![cases::FORMAT_VERSION.to_string()];
    let mut parity = Vec::new();
    let mut kinds = [0usize; 3];
    for case in &all {
        let (webp, enc, ds) = cases::zen_hashes(case);
        let k = cases::kind(&webp);
        kinds[match k {
            Kind::Lossy => 0,
            Kind::Lossless => 1,
            Kind::Anim => 2,
        }] += 1;
        for &(mode, zen) in &ds {
            let (b, w, h) = lib_decode(mode, &webp);
            let lib = cases::digest(&b, w, h);
            if lib != zen {
                parity.push(format!(
                    "{} [{mode}] zen {zen:016x} libwebp {lib:016x}",
                    case.name
                ));
            }
        }
        lines.push(cases::format_line(&case.name, enc, cases::fold(&ds)));
    }
    eprintln!(
        "{} cases: {} lossy, {} lossless, {} animated",
        all.len(),
        kinds[0],
        kinds[1],
        kinds[2]
    );
    assert!(
        parity.is_empty(),
        "{} decode digests differ from libwebp:\n{}",
        parity.len(),
        parity
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/codec_golden.tsv");
    let text = lines.join("\n") + "\n";
    if std::env::var_os("ZENWEBP_GOLDEN_BLESS").is_some() {
        std::fs::write(path, &text).unwrap();
        eprintln!("blessed {path}");
        return;
    }
    let want = std::fs::read_to_string(path).unwrap();
    let golden = cases::parse_golden(&want);
    let mut changed = Vec::new();
    for (line, (name, enc, dec)) in lines[1..].iter().zip(&golden) {
        if *line != cases::format_line(name, *enc, *dec) {
            changed.push(format!(
                "now {line}\n was {}",
                cases::format_line(name, *enc, *dec)
            ));
        }
    }
    assert_eq!(golden.len(), all.len(), "case count changed; re-bless");
    assert!(
        changed.is_empty(),
        "{} golden cases changed (decode still matches libwebp):\n{}",
        changed.len(),
        changed
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
