#![cfg(not(target_arch = "wasm32"))]
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenwebp::zencodec::WebpEncoderConfig;

#[test]
fn declared_rows_preserve_strided_pixels_and_protocol() {
    for (lossy, cfg) in [
        WebpEncoderConfig::lossless(),
        WebpEncoderConfig::lossy().with_quality(80.0),
    ]
    .into_iter()
    .enumerate()
    {
        let desc = PixelDescriptor::RGB8_SRGB;
        for stride in [15, 18, 24] {
            let (w, h) = (5, 4);
            let mut bytes = vec![199; stride * h as usize];
            for y in 0..h as usize {
                for x in 0..w as usize * 3 {
                    bytes[y * stride + x] = (x * 13 + y * 41) as u8;
                }
            }
            let image = || PixelSlice::new(&bytes, w, h, stride, desc).unwrap();
            let packed: Vec<u8> = bytes
                .chunks(stride)
                .flat_map(|row| row[..w as usize * 3].iter().copied())
                .collect();
            let expected = cfg
                .clone()
                .job()
                .encoder()
                .unwrap()
                .encode(PixelSlice::new(&packed, w, h, w as usize * 3, desc).unwrap())
                .unwrap()
                .into_vec();
            let (decoded, ow, oh) = webpx::decode_rgba(&expected).unwrap();
            assert_eq!((ow, oh), (w, h));
            if lossy == 0 {
                let rgba: Vec<u8> = packed
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect();
                assert_eq!(decoded, rgba);
            }
            assert_eq!(
                cfg.clone()
                    .job()
                    .encoder()
                    .unwrap()
                    .encode(image())
                    .unwrap()
                    .into_vec(),
                expected
            );
            for declared in [false, true] {
                let mut enc = cfg.clone().job().encoder().unwrap();
                if declared {
                    assert!(enc.begin(0, h, desc).is_err());
                    assert_eq!(
                        enc.begin(w, h, desc).unwrap(),
                        zencodec::encode::OutputTiming::AT_FINISH
                    );
                    assert!(enc.begin(w, h, desc).is_err());
                }
                for y in [0, 2] {
                    enc.push_rows(
                        PixelSlice::new(&bytes[y * stride..], w, 2, stride, desc).unwrap(),
                    )
                    .unwrap();
                }
                assert_eq!(enc.finish().unwrap().into_vec(), expected);
            }
            let mut enc = cfg.clone().job().encoder().unwrap();
            enc.begin(w, h, desc).unwrap();
            assert!(enc.encode(image()).is_err());
            let mut enc = cfg.clone().job().encoder().unwrap();
            enc.begin(w, h, desc).unwrap();
            enc.push_rows(PixelSlice::new(&bytes, w, 2, stride, desc).unwrap())
                .unwrap();
            assert!(enc.finish().is_err());
            let mut enc = cfg.clone().job().encoder().unwrap();
            enc.begin(w, h, desc).unwrap();
            assert!(
                enc.push_rows(PixelSlice::new(&bytes, w - 1, 1, stride, desc).unwrap())
                    .is_err()
            );
            enc.push_rows(image()).unwrap();
            assert!(
                enc.push_rows(PixelSlice::new(&bytes, w, 1, stride, desc).unwrap())
                    .is_err()
            );
            assert_eq!(enc.finish().unwrap().into_vec(), expected);
        }
    }
}

#[test]
fn shared_streaming_conformance_exercises_declared_webp_rows() {
    for config in [WebpEncoderConfig::lossless(), WebpEncoderConfig::lossy()] {
        zencodec_testkit::check_streaming_output_honesty(
            config,
            zenwebp::zencodec::WebpDecoderConfig::new(),
            &zencodec_testkit::TestImage::rgb8_gradient(5, 4),
        )
        .unwrap();
    }
}

#[test]
fn begin_accepts_a_supported_webp_canvas() {
    let mut encoder = WebpEncoderConfig::lossless().job().encoder().unwrap();
    assert_eq!(
        encoder.begin(5, 4, PixelDescriptor::RGB8_SRGB).unwrap(),
        zencodec::encode::OutputTiming::AT_FINISH,
    );
}

#[test]
fn legacy_rows_do_not_encode_padding_as_samples() {
    let desc = PixelDescriptor::RGB8_SRGB;
    let packed: Vec<u8> = (0..60).map(|i| (i * 3) as u8).collect();
    let mut padded = vec![199; 96];
    for y in 0..4 {
        padded[y * 24..y * 24 + 15].copy_from_slice(&packed[y * 15..y * 15 + 15]);
    }
    let cfg = WebpEncoderConfig::lossless();
    let expected = cfg
        .clone()
        .job()
        .encoder()
        .unwrap()
        .encode(PixelSlice::new(&packed, 5, 4, 15, desc).unwrap())
        .unwrap()
        .into_vec();
    let mut encoder = cfg.job().encoder().unwrap();
    encoder
        .push_rows(PixelSlice::new(&padded, 5, 4, 24, desc).unwrap())
        .unwrap();
    assert_eq!(encoder.finish().unwrap().into_vec(), expected);
}
