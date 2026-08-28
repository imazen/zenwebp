//! Regression tests for the #78 review-backlog items landed in the
//! 2026-08-28 pass (encoder config knob-loss + animation ALPH gating).
//!
//! Every test here was watched to FAIL against the defect it pins (see the
//! commit body for the mutation each one was run against).

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zenwebp::mux::{AnimationConfig, AnimationEncoder};
use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout, ValidationError};

fn gradient_rgba(w: u32, h: u32, alpha: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let noise = if (x ^ y) & 1 == 0 { 0 } else { 7 };
            out[i] = ((x * 255 / w.max(1)) as u8).wrapping_add(noise);
            out[i + 1] = ((y * 255 / h.max(1)) as u8).wrapping_add(noise);
            out[i + 2] = (((x + y) * 255 / (w + h).max(1)) as u8) ^ noise;
            out[i + 3] = alpha(x, y);
        }
    }
    out
}

fn contains_chunk(webp: &[u8], fourcc: &[u8; 4]) -> bool {
    webp.windows(4).any(|w| w == fourcc)
}

/// A-R: `EncoderConfig::with_lossless` claimed to preserve common settings
/// but rebuilt the target config with `..new()`, dropping `alpha_quality`
/// and `target_size` in both directions.
#[test]
fn with_lossless_round_trips_alpha_quality_and_target_size() {
    let lossy = EncoderConfig::Lossy(
        LossyConfig::new()
            .with_quality(63.0)
            .with_method(2)
            .with_alpha_quality(37)
            .with_target_size(12_345),
    );
    let lossless = lossy.clone().with_lossless(true);
    match &lossless {
        EncoderConfig::Lossless(c) => {
            assert_eq!(c.quality, 63.0);
            assert_eq!(c.method, 2);
            assert_eq!(c.alpha_quality, 37, "alpha_quality dropped lossy→lossless");
            assert_eq!(c.target_size, 12_345, "target_size dropped lossy→lossless");
        }
        other => panic!("expected Lossless, got {other:?}"),
    }
    let back = lossless.with_lossless(false);
    match &back {
        EncoderConfig::Lossy(c) => {
            assert_eq!(c.alpha_quality, 37, "alpha_quality dropped lossless→lossy");
            assert_eq!(c.target_size, 12_345, "target_size dropped lossless→lossy");
        }
        other => panic!("expected Lossy, got {other:?}"),
    }
}

/// A-R: `alpha_effort` was the one lossy knob `validate()` never looked
/// at — `Some(200)` passed validation and was clamped later.
#[test]
fn validate_rejects_out_of_range_alpha_effort() {
    let mut cfg = LossyConfig::new();
    cfg.alpha_effort = Some(200);
    match cfg.validate() {
        Err(ValidationError::AlphaEffortOutOfRange { value: 200, valid }) => {
            assert_eq!(valid, 0..=6);
        }
        other => panic!("expected AlphaEffortOutOfRange, got {other:?}"),
    }
    cfg.alpha_effort = Some(6);
    cfg.validate().expect("6 is the top of the range");
    cfg.alpha_effort = None;
    cfg.validate().expect("unset is valid");
}

/// A-R: the animation path gated ALPH on `use_lossy && has_alpha()` while
/// the still path also requires the alpha plane to be non-opaque, so an
/// opaque RGBA animation frame carried a redundant all-255 ALPH chunk.
#[test]
fn opaque_rgba_animation_frame_has_no_alph_chunk() {
    let (w, h) = (48u32, 40u32);
    let cfg = EncoderConfig::new_lossy().with_quality(70.0);
    let opaque = gradient_rgba(w, h, |_, _| 255);
    let translucent = gradient_rgba(w, h, |x, y| if (x / 8 + y / 8) % 2 == 0 { 0 } else { 200 });

    let encode_anim = |frame: &[u8]| {
        let mut anim = AnimationEncoder::new(
            w,
            h,
            AnimationConfig {
                minimize_size: false,
                ..Default::default()
            },
        )
        .unwrap();
        anim.add_frame(frame, PixelLayout::Rgba8, 0, &cfg).unwrap();
        anim.add_frame(frame, PixelLayout::Rgba8, 100, &cfg)
            .unwrap();
        anim.finalize(100).unwrap()
    };

    let opaque_webp = encode_anim(&opaque);
    assert!(
        !contains_chunk(&opaque_webp, b"ALPH"),
        "opaque RGBA frames must not carry an ALPH chunk (still path parity)"
    );
    // Control: the same pipeline still emits ALPH when alpha is real.
    let translucent_webp = encode_anim(&translucent);
    assert!(
        contains_chunk(&translucent_webp, b"ALPH"),
        "translucent RGBA frames must keep their ALPH chunk"
    );
    // And the still path agrees on both.
    let still_opaque = EncodeRequest::new(&cfg, &opaque, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    assert!(!contains_chunk(&still_opaque, b"ALPH"));
}

/// The literal-only VP8L writer behind `encode_frame_lossless`'s
/// `implicit_dimensions` flag was unreachable and is gone; every layout —
/// including L8 / La8 — must still round-trip exactly through the full
/// pipeline.
#[test]
fn lossless_layouts_round_trip_exactly_after_fallback_removal() {
    let (w, h) = (37u32, 29u32);
    let cfg = EncoderConfig::new_lossless();
    // Alpha stays >= 1: the default `exact = false` (libwebp parity) scrubs
    // RGB under fully transparent pixels, which is not what this test pins.
    let rgba = gradient_rgba(w, h, |x, y| (((x * 7 + y * 3) & 255) | 1) as u8);
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let gray: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[1]).collect();
    let gray_alpha: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[1], p[3]])
        .collect();
    for (name, px, layout) in [
        ("rgba8", &rgba, PixelLayout::Rgba8),
        ("rgb8", &rgb, PixelLayout::Rgb8),
        ("l8", &gray, PixelLayout::L8),
        ("la8", &gray_alpha, PixelLayout::La8),
    ] {
        let webp = EncodeRequest::new(&cfg, px, layout, w, h)
            .encode()
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&webp).unwrap();
        assert_eq!((dw, dh), (w, h), "{name}");
        // Compare in RGBA space; widen the source the same way the encoder does.
        let expected: Vec<u8> = match layout {
            PixelLayout::Rgba8 => px.clone(),
            PixelLayout::Rgb8 => px
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            PixelLayout::L8 => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
            PixelLayout::La8 => px
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
            _ => unreachable!(),
        };
        assert_eq!(
            decoded, expected,
            "{name}: lossless round-trip must be exact"
        );
    }
}

#[cfg(feature = "target-zensim")]
mod target_zensim {
    use super::*;
    use zenwebp::ZensimTarget;

    struct AlwaysStopped;
    impl enough::Stop for AlwaysStopped {
        fn check(&self) -> Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }

    /// A-R: the closed-loop probes were built without the request's stop
    /// token, so cancellation was ignored for the whole iteration.
    #[test]
    fn target_zensim_iteration_honors_stop() {
        let (w, h) = (64u32, 48u32);
        let px = gradient_rgba(w, h, |_, _| 255);
        let cfg = LossyConfig::new().with_target_zensim(ZensimTarget::new(80.0).with_max_passes(3));
        let stopped = AlwaysStopped;
        let err = EncodeRequest::lossy(&cfg, &px, PixelLayout::Rgba8, w, h)
            .with_stop(&stopped)
            .encode()
            .expect_err("a pre-stopped token must cancel the target_zensim encode");
        assert!(
            matches!(err.error(), zenwebp::EncodeError::Cancelled(_)),
            "expected Cancelled, got {err:?}"
        );
    }

    /// A-R: `encode_with_stats` bypassed the target_zensim iteration and
    /// single-passed silently. Its bytes must now match `encode()` and its
    /// stats must describe those bytes.
    #[test]
    fn encode_with_stats_honors_target_zensim() {
        let (w, h) = (64u32, 48u32);
        let px = gradient_rgba(w, h, |_, _| 255);
        let cfg = LossyConfig::new().with_target_zensim(ZensimTarget::new(85.0).with_max_passes(4));
        let via_encode = EncodeRequest::lossy(&cfg, &px, PixelLayout::Rgba8, w, h)
            .encode()
            .unwrap();
        let (via_stats, stats) = EncodeRequest::lossy(&cfg, &px, PixelLayout::Rgba8, w, h)
            .encode_with_stats()
            .unwrap();
        assert_eq!(
            via_encode, via_stats,
            "encode_with_stats must run the same iteration as encode"
        );
        assert_eq!(stats.coded_size as usize, via_stats.len());
        // Control: with no target the two entry points also agree.
        let plain = LossyConfig::new().with_quality(75.0);
        let a = EncodeRequest::lossy(&plain, &px, PixelLayout::Rgba8, w, h)
            .encode()
            .unwrap();
        let (b, _) = EncodeRequest::lossy(&plain, &px, PixelLayout::Rgba8, w, h)
            .encode_with_stats()
            .unwrap();
        assert_eq!(a, b);
    }
}

/// A-R: the animation encoder silently single-passed frames whose config
/// set `target_zensim`; it now refuses with `UnsupportedOperation`.
#[test]
fn animation_encoder_rejects_target_zensim() {
    let (w, h) = (32u32, 32u32);
    let px = gradient_rgba(w, h, |_, _| 255);
    let mut cfg = LossyConfig::new();
    cfg.target_zensim = Some(zenwebp::ZensimTarget::new(80.0));
    let cfg = EncoderConfig::Lossy(cfg);
    let mut anim = AnimationEncoder::new(w, h, AnimationConfig::default()).unwrap();
    let err = anim
        .add_frame(&px, PixelLayout::Rgba8, 0, &cfg)
        .expect_err("target_zensim is stills-only; the animation path must not silently ignore it");
    assert!(
        matches!(
            err.error(),
            zenwebp::mux::MuxError::EncodeError(zenwebp::EncodeError::UnsupportedOperation(_))
        ),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Decoder / mux robustness subset (#78 B/C).
// ---------------------------------------------------------------------------

mod decode_mux {
    use zenwebp::decoder::LoopCount;
    use zenwebp::mux::{
        AnimationDecoder, BlendMethod, DisposeMethod, MuxError, MuxFrame, WebPDemuxer, WebPMux,
    };
    use zenwebp::{DecodeConfig, DecodeError, DecodeRequest, Limits, StreamingDecoder};

    fn animated_fixture() -> Vec<u8> {
        std::fs::read("tests/images/animated/random_lossy.webp").unwrap()
    }

    /// B-R: `decode_all` cloned one full canvas per frame with no aggregate
    /// bound — the per-frame `check_memory` inside `read_frame` never saw
    /// the accumulation. Streaming the same frames one at a time under the
    /// same limit must still work.
    #[test]
    fn decode_all_enforces_aggregate_memory_bound() {
        let data = animated_fixture();
        let probe = AnimationDecoder::new(&data).unwrap();
        let info = probe.info();
        let canvas = u64::from(info.canvas_width) * u64::from(info.canvas_height) * 4;
        assert!(info.frame_count >= 3, "fixture needs >= 3 frames");
        // Room for the working canvas + ~1.5 retained frames, not all of them.
        let limit = canvas * 5 / 2;
        let cfg = DecodeConfig::default().limits(Limits::none().max_memory(limit));

        let mut dec = AnimationDecoder::new_with_config(&data, &cfg).unwrap();
        let err = dec
            .decode_all()
            .expect_err("retaining every frame must exceed the aggregate bound");
        assert!(
            matches!(err.error(), DecodeError::MemoryLimitExceeded),
            "got {err:?}"
        );

        // Same budget, streamed: every frame decodes.
        let mut dec = AnimationDecoder::new_with_config(&data, &cfg).unwrap();
        let mut n = 0;
        while dec.next_frame().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, info.frame_count);

        // Generous budget: decode_all succeeds and returns every frame.
        let cfg = DecodeConfig::default()
            .limits(Limits::none().max_memory(canvas * (u64::from(info.frame_count) + 2)));
        let mut dec = AnimationDecoder::new_with_config(&data, &cfg).unwrap();
        assert_eq!(dec.decode_all().unwrap().len(), info.frame_count as usize);
    }

    /// B-R: `Limits::max_file_size` (`check_file_size`) had no caller on
    /// the native decode path, and `StreamingDecoder::append` grew its
    /// buffer without bound.
    #[test]
    fn max_file_size_is_enforced_natively_and_while_streaming() {
        let data = animated_fixture();
        let too_small = DecodeConfig::default().limits(Limits::none().max_file_size(64));
        let err = DecodeRequest::new(&too_small, &data)
            .decode_rgba()
            .expect_err("64-byte limit must reject a 22 KB file");
        assert!(
            matches!(err.error(), DecodeError::InvalidParameter(_)),
            "got {err:?}"
        );
        let exact = DecodeConfig::default().limits(Limits::none().max_file_size(data.len() as u64));
        DecodeRequest::new(&exact, &data)
            .decode_rgba()
            .expect("a limit equal to the file size must pass");

        // Streaming: the bound applies to the buffered total, so the
        // rejection fires on the append that would cross it.
        let mut dec = StreamingDecoder::with_config(too_small);
        let mut rejected = false;
        for chunk in data.chunks(40) {
            if dec.append(chunk).is_err() {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "streaming append must honor max_file_size");
        assert!(dec.bytes_buffered() <= 64);
    }

    /// C-R: a recorded ANMF that fails to parse ended `frames()` early
    /// while `num_frames()` / `len()` still counted it. The demuxer now
    /// rejects such a file up front; either way the iterator can never
    /// under-run its own length.
    #[test]
    fn demux_never_underreports_frames() {
        let mut data = animated_fixture();
        let clean = WebPDemuxer::new(&data).unwrap();
        let n = clean.num_frames();
        assert!(n >= 2);
        assert_eq!(clean.frames().count(), n as usize);

        // Corrupt the LAST frame's sub-chunk fourcc (payload + 16).
        let last_anmf = data
            .windows(4)
            .rposition(|w| w == b"ANMF")
            .expect("fixture has ANMF chunks");
        let sub_fourcc = last_anmf + 8 + 16;
        data[sub_fourcc..sub_fourcc + 4].copy_from_slice(b"JUNK");

        match WebPDemuxer::new(&data) {
            Err(e) => assert!(matches!(e.error(), MuxError::InvalidFormat(_)), "got {e:?}"),
            Ok(d) => {
                // If a future demuxer chooses to tolerate it, the contract
                // still holds: what it counts, it yields.
                assert_eq!(
                    d.frames().count(),
                    d.num_frames() as usize,
                    "frames() under-ran num_frames()"
                );
                assert_eq!(d.frames().len(), d.num_frames() as usize);
            }
        }
    }

    /// C-R: `duration_ms` was written with `write_u24_le`, silently
    /// truncating anything above 16 777 215.
    #[test]
    fn push_frame_rejects_duration_over_24_bits() {
        let frame = |duration_ms: u32| MuxFrame {
            x_offset: 0,
            y_offset: 0,
            width: 10,
            height: 10,
            duration_ms,
            dispose: DisposeMethod::None,
            blend: BlendMethod::Overwrite,
            bitstream: vec![0],
            alpha_data: None,
            is_lossless: true,
        };
        let mut mux = WebPMux::new(100, 100);
        mux.set_animation([0; 4], LoopCount::Forever);
        mux.push_frame(frame(0x00FF_FFFF))
            .expect("16777215 ms is the maximum representable duration");
        let err = mux
            .push_frame(frame(0x0100_0000))
            .expect_err("16777216 ms does not fit the 24-bit ANMF field");
        assert!(
            matches!(
                err.error(),
                MuxError::FrameDurationTooLarge {
                    duration_ms: 0x0100_0000
                }
            ),
            "got {err:?}"
        );
    }

    /// The streaming decoder's header-parse retry now backs off
    /// geometrically; the observable contract is unchanged — `info()` is
    /// available at `HeaderReady` and always by `Complete`, in one-shot and
    /// byte-at-a-time delivery alike.
    #[test]
    fn streaming_header_available_by_complete_under_any_chunking() {
        let data = animated_fixture();
        for chunk_len in [1usize, 7, 64, data.len()] {
            let mut dec = StreamingDecoder::new();
            let mut last = None;
            for chunk in data.chunks(chunk_len) {
                let status = dec.append(chunk).unwrap();
                if matches!(status, zenwebp::decoder::StreamStatus::HeaderReady) {
                    dec.info().expect("HeaderReady must make info() available");
                }
                last = Some(status);
            }
            assert!(
                matches!(last, Some(zenwebp::decoder::StreamStatus::Complete)),
                "chunk_len {chunk_len}: {last:?}"
            );
            let info = dec.info().expect("info() must be available once Complete");
            assert!(info.width > 0 && info.height > 0);
        }
    }
}
