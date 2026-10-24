//! Regression tests for the #78 review-backlog decoder/mux items (backport of
//! 5aabd47). Every test here was watched to FAIL against the defect it pins.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

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
            matches!(err, DecodeError::MemoryLimitExceeded),
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
