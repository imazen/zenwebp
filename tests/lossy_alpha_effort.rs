//! Lossy-alpha operating point: the tuned default caps the alpha plane's
//! VP8L at `quality = 8·effort` (no q100 escalation at m6 — that cruncher
//! cost ~40x for ~25% of the ALPH chunk, debug-wedge 2026-08-20), and
//! `LossyConfig::with_alpha_effort` decouples alpha effort from `method`.
//! At `alpha_quality = 100` decoded alpha must stay bit-exact regardless.

use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LossyConfig, PixelLayout,
};

/// 96x96 logo-like RGBA: antialiased disc edge -> many distinct alpha levels.
fn test_rgba(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 4];
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let r_out = w.min(h) as f32 * 0.45;
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            let a = ((r_out - d + 1.5) / 3.0).clamp(0.0, 1.0);
            let i = (y * w + x) * 4;
            px[i] = (x * 255 / w) as u8;
            px[i + 1] = (y * 255 / h) as u8;
            px[i + 2] = ((x ^ y) & 0xff) as u8;
            px[i + 3] = (a * 255.0) as u8;
        }
    }
    px
}

fn encode(cfg: &LossyConfig, rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    EncodeRequest::new(
        &EncoderConfig::Lossy(cfg.clone()),
        rgba,
        PixelLayout::Rgba8,
        w,
        h,
    )
    .encode()
    .unwrap()
}

fn decoded_alpha(webp: &[u8]) -> Vec<u8> {
    let config = DecodeConfig::default();
    let (pixels, _, _, layout) = DecodeRequest::new(&config, webp).decode().unwrap();
    assert_eq!(layout, PixelLayout::Rgba8);
    pixels.chunks(4).map(|p| p[3]).collect()
}

#[test]
fn tuned_m6_alpha_values_exact() {
    let (w, h) = (96usize, 96usize);
    let rgba = test_rgba(w, h);
    let src_alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    let cfg = LossyConfig::new().with_quality(80.0).with_method(6);
    let out = encode(&cfg, &rgba, w as u32, h as u32);
    assert_eq!(
        decoded_alpha(&out),
        src_alpha,
        "alpha_quality=100 must keep decoded alpha bit-exact at m6"
    );
}

#[test]
fn alpha_effort_knob_honored() {
    let (w, h) = (96usize, 96usize);
    let rgba = test_rgba(w, h);
    let src_alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    let base = LossyConfig::new().with_quality(80.0).with_method(6);
    let out_default = encode(&base, &rgba, w as u32, h as u32);
    let out_e0 = encode(
        &base.clone().with_alpha_effort(0),
        &rgba,
        w as u32,
        h as u32,
    );
    // Same VP8 image plane, different alpha operating point -> the outputs
    // must differ (the knob reached the ALPH coder)...
    assert_ne!(
        out_default, out_e0,
        "alpha_effort must change the ALPH payload"
    );
    // ...and both stay bit-exact on decoded alpha values.
    assert_eq!(decoded_alpha(&out_default), src_alpha);
    assert_eq!(decoded_alpha(&out_e0), src_alpha);
}
