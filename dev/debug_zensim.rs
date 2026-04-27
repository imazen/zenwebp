use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

fn main() {
    let w: u32 = 256;
    let h: u32 = 256;
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    let mut state: u32 = 0xC0FFEE_u32;
    for y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                let r = ((x * 255) / (w / 2)) as u8;
                let g = ((y * 255) / h) as u8;
                let b = (((x + y) * 255) / (w / 2 + h)) as u8;
                buf.extend_from_slice(&[r, g, b]);
            } else {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let r = (state >> 24) as u8;
                let g = (state >> 16) as u8;
                let b = (state >> 8) as u8;
                buf.extend_from_slice(&[r, g, b]);
            }
        }
    }
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest());
    let mut src_chunks: Vec<[u8; 3]> = Vec::with_capacity((w * h) as usize);
    for px in buf.chunks_exact(3) {
        src_chunks.push([px[0], px[1], px[2]]);
    }
    let slice = zensim::RgbSlice::new(&src_chunks, w as usize, h as usize);
    let pre = z.precompute_reference(&slice).unwrap();

    for q in [40.0_f32, 60.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0] {
        let cfg = LossyConfig::new().with_quality(q).with_method(4);
        let req = EncodeRequest::lossy(&cfg, &buf, PixelLayout::Rgb8, w, h);
        let webp = req.encode().unwrap();
        let (rgb, w2, h2) = zenwebp::oneshot::decode_rgb(&webp).unwrap();
        assert_eq!((w2, h2), (w, h));
        let mut dec_chunks: Vec<[u8; 3]> = Vec::with_capacity((w2 * h2) as usize);
        for px in rgb.chunks_exact(3) {
            dec_chunks.push([px[0], px[1], px[2]]);
        }
        let dec_slice = zensim::RgbSlice::new(&dec_chunks, w2 as usize, h2 as usize);
        let res = z.compute_with_ref(&pre, &dec_slice).unwrap();
        println!("q={:.1} bytes={} score={:.2}", q, webp.len(), res.score());
    }
}
