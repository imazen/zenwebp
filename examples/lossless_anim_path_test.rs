use std::fs;
use zencodec::encode::{AnimationFrameEncoder, DynEncoderConfig, DynEncodeJob};
use zenpixels::{PixelDescriptor, PixelSlice};

fn dump_chunks(label: &str, d: &[u8]) {
    println!("--- {label} ({} bytes) ---", d.len());
    let mut i = 12usize;
    while i + 8 <= d.len() {
        let cid = std::str::from_utf8(&d[i..i+4]).unwrap_or("???");
        let sz = u32::from_le_bytes([d[i+4], d[i+5], d[i+6], d[i+7]]);
        println!("  Chunk {cid:?} @ {i}, size={sz}");
        i += 8 + sz as usize;
        if sz % 2 == 1 { i += 1; }
    }
}

fn main() {
    let bytes = fs::read("/home/lilith/work/imageflow-zen-v3/.image-cache/sources/imageflow-resources/test_inputs/1_webp_ll.webp").unwrap();
    let (pixels, w, h) = zenwebp::oneshot::decode_rgba(&bytes).unwrap();
    println!("Decoded: {w}x{h}, {} bytes", pixels.len());

    // Path A: animation frame encoder (what imageflow zen uses)
    let cfg_a = zenwebp::zencodec::WebpEncoderConfig::lossless().with_quality(85.0);
    let job_a = cfg_a.dyn_job();
    let mut frame_enc = job_a.into_animation_frame_encoder().unwrap();
    let ps_a = PixelSlice::new(&pixels, w, h, w as usize * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    frame_enc.push_frame(ps_a, 100, None).unwrap();
    let out_a = frame_enc.finish(None).unwrap();
    fs::write("/tmp/zenwebp_via_anim_path.webp", out_a.data()).unwrap();
    dump_chunks("animation path", out_a.data());

    // Path B: single-frame encoder
    let cfg_b = zenwebp::zencodec::WebpEncoderConfig::lossless().with_quality(85.0);
    let encoder = cfg_b.dyn_job().into_encoder().unwrap();
    let ps_b = PixelSlice::new(&pixels, w, h, w as usize * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let out_b = encoder.encode(ps_b).unwrap();
    fs::write("/tmp/zenwebp_via_single_path.webp", out_b.data()).unwrap();
    dump_chunks("single-frame path", out_b.data());
}
