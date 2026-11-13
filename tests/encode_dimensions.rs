//! VP8 has 14-bit dimensions; VP8L stores dimension minus one.
use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeError, EncodeRequest, LosslessConfig, LossyConfig,
    PixelLayout,
};

#[test]
fn lossy_rejects_unrepresentable_sides() {
    let config = LossyConfig::new().with_method(0);
    for side in [16384, 16385, 20000, 65535, 65536] {
        for (w, h) in [(side, 1), (1, side)] {
            let pixels = vec![128; w as usize * h as usize * 3];
            let error =
                match EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, w, h).encode() {
                    Err(error) => error,
                    Ok(_) => panic!("VP8 dimension must not wrap: {w}x{h}"),
                };
            assert!(
                matches!(error.error(), EncodeError::InvalidDimensions),
                "{w}x{h}: {error:?}"
            );
        }
    }
}

#[test]
fn representable_boundaries_decode_at_original_dimensions() {
    for lossless in [false, true] {
        let side = if lossless { 16384 } else { 16383 };
        for (w, h) in [(side, 1), (1, side)] {
            let pixels = vec![128; w as usize * h as usize * 3];
            let bytes = if lossless {
                let config = LosslessConfig::new().with_method(0);
                EncodeRequest::lossless(&config, &pixels, PixelLayout::Rgb8, w, h).encode()
            } else {
                let config = LossyConfig::new().with_method(0);
                EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, w, h).encode()
            }
            .unwrap();
            let (_, dw, dh) = DecodeRequest::new(&DecodeConfig::default(), &bytes)
                .decode_rgb()
                .unwrap();
            assert_eq!((dw, dh), (w, h));
            #[cfg(not(target_arch = "wasm32"))]
            {
                let limits = webpx::Limits::default()
                    .with_max_width(16384)
                    .with_max_height(16384);
                let config = webpx::DecoderConfig::default().limits(limits);
                let (_, dw, dh) = webpx::Decoder::new(&bytes)
                    .unwrap()
                    .config(config)
                    .decode_rgba_raw()
                    .unwrap();
                assert_eq!((dw, dh), (w, h), "independent libwebp dimensions");
            }
        }
    }
}
