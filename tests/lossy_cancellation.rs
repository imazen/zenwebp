//! Regression for 8d62efc on main: a pre-stopped token must fail a lossy
//! one-shot decode. 0.4.4 decoded lossy WebP successfully under a stopped
//! token — the native lossy path had no entry check, the zencodec adapter
//! never wired the token into the native request, and it swallowed lossy-path
//! errors by retrying the general path.

struct AlwaysStopped;

impl enough::Stop for AlwaysStopped {
    fn check(&self) -> Result<(), enough::StopReason> {
        Err(enough::StopReason::Cancelled)
    }
}

fn lossy_webp() -> Vec<u8> {
    let (w, h) = (32u32, 32u32);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7) as u8).collect();
    let cfg = zenwebp::LossyConfig::new();
    zenwebp::EncodeRequest::lossy(&cfg, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap()
}

#[test]
fn native_lossy_decode_honors_pre_stopped_token() {
    let data = lossy_webp();
    let cfg = zenwebp::DecodeConfig::default();
    assert!(
        zenwebp::DecodeRequest::new(&cfg, &data)
            .decode_rgba()
            .is_ok()
    );

    let stop = AlwaysStopped;
    for result in [
        zenwebp::DecodeRequest::new(&cfg, &data)
            .stop(&stop)
            .decode_rgba()
            .map(|_| ()),
        zenwebp::DecodeRequest::new(&cfg, &data)
            .stop(&stop)
            .decode_rgb()
            .map(|_| ()),
    ] {
        let err = result.expect_err("stopped token must cancel the lossy decode");
        assert!(
            matches!(err.error(), zenwebp::DecodeError::Cancelled(_)),
            "expected Cancelled, got {err:?}"
        );
    }
}

/// One-frame lossy animation (VP8X + ANIM + ANMF wrapping a bare VP8 chunk).
fn lossy_animation() -> Vec<u8> {
    let still = lossy_webp(); // simple format: RIFF WEBP "VP8 " ...
    let vp8 = &still[12..];
    let u24 = |v: u32| [v as u8, (v >> 8) as u8, (v >> 16) as u8];
    let mut anmf = Vec::new();
    anmf.extend_from_slice(&u24(0));
    anmf.extend_from_slice(&u24(0));
    anmf.extend_from_slice(&u24(31));
    anmf.extend_from_slice(&u24(31));
    anmf.extend_from_slice(&u24(100));
    anmf.push(0);
    anmf.extend_from_slice(vp8);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF\0\0\0\0WEBPVP8X");
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&[0x02, 0, 0, 0]);
    out.extend_from_slice(&u24(31));
    out.extend_from_slice(&u24(31));
    out.extend_from_slice(b"ANIM");
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&[0; 6]);
    out.extend_from_slice(b"ANMF");
    out.extend_from_slice(&(anmf.len() as u32).to_le_bytes());
    out.extend_from_slice(&anmf);
    if anmf.len() % 2 == 1 {
        out.push(0);
    }
    let riff = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff.to_le_bytes());
    out
}

#[test]
fn animated_lossy_frame_honors_pre_stopped_token() {
    let data = lossy_animation();
    let mut decoder = zenwebp::WebPDecoder::new(&data).unwrap();
    assert!(decoder.is_animated());
    let mut buf = vec![0u8; decoder.output_buffer_size().unwrap()];
    assert!(
        decoder.read_frame(&mut buf).is_ok(),
        "unstopped frame decodes"
    );

    let mut decoder = zenwebp::WebPDecoder::new(&data).unwrap();
    let stop = AlwaysStopped;
    decoder.set_stop(Some(&stop));
    let err = decoder
        .read_frame(&mut buf)
        .expect_err("stopped token must cancel the frame");
    assert!(
        matches!(err.error(), zenwebp::DecodeError::Cancelled(_)),
        "expected Cancelled, got {err:?}"
    );
}

#[cfg(feature = "zencodec")]
#[test]
fn zencodec_lossy_decode_honors_pre_stopped_token() {
    use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
    let data = lossy_webp();
    let job = || zenwebp::zencodec::WebpDecoderConfig::new().job();
    assert!(
        job()
            .decoder(std::borrow::Cow::Borrowed(&data[..]), &[])
            .and_then(|d| d.decode())
            .is_ok()
    );

    let err = job()
        .with_stop(zencodec::StopToken::new(AlwaysStopped))
        .decoder(std::borrow::Cow::Borrowed(&data[..]), &[])
        .and_then(|d| d.decode())
        .expect_err("stopped token must cancel the lossy decode");
    assert!(
        matches!(err.error(), zenwebp::DecodeError::Cancelled(_)),
        "expected Cancelled, got {err:?}"
    );
}
