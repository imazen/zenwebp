//! how-far integration: every pulse entry point reports its documented units
//! to completion, and a pulse's cancellation stops the work part-way.
//!
//! Units: encodes report percent (total 100); still decodes report image rows
//! (total = height); animation decoders report frames (total = frame count);
//! the zencodec animation encoder reports one unit per accepted frame.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use how_far::{Child, Execution, PhaseSpec, PlanError, Pulse, Report, Stop, StopReason, Total};
use how_far_along::{Phase, PulseTree, Unstoppable};
use zencodec::decode::{
    AnimationFrameDecoder as _, Decode as _, DecodeJob as _, DecoderConfig as _,
};
use zencodec::encode::{
    AnimationFrameEncoder as _, EncodeJob as _, Encoder as _, EncoderConfig as _,
};
use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};
use zenwebp::{
    DecodeConfig, DecodeError, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout,
};

fn pattern(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let n = (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ seed) >> 7;
            v.extend_from_slice(&[
                (x * 255 / w) as u8 ^ (n as u8 & 15),
                (y * 255 / h) as u8,
                (n >> 4) as u8,
                ((x + y + seed) % 200 + 40) as u8,
            ]);
        }
    }
    v
}

fn tree() -> PulseTree {
    PulseTree::new(Phase::new("op", Total::Unknown), Unstoppable)
}

/// The tree's root as observed: (declared total, completed units).
fn observed(tree: &PulseTree) -> (Total, u64) {
    let s = tree.observer().snapshot();
    (s.total, s.completed)
}

/// A pulse that counts reports and cancels once `cancel_at` units arrived.
struct CancelAfter {
    advanced: AtomicU64,
    total: AtomicU64,
    cancel_at: u64,
}

impl CancelAfter {
    fn new(cancel_at: u64) -> Self {
        Self {
            advanced: AtomicU64::new(0),
            total: AtomicU64::new(0),
            cancel_at,
        }
    }
    fn advanced(&self) -> u64 {
        self.advanced.load(Ordering::Relaxed)
    }
}

impl Stop for CancelAfter {
    fn check(&self) -> Result<(), StopReason> {
        if self.advanced() >= self.cancel_at {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl Report for CancelAfter {
    fn advance(&self, completed: u64) {
        self.advanced.fetch_add(completed, Ordering::Relaxed);
    }
}

impl Pulse for CancelAfter {
    fn split(&self, _: Execution, _: &[PhaseSpec<'_>]) -> Result<Vec<Child<'_>>, PlanError> {
        Err(PlanError::Unsupported)
    }
    fn set_total(&self, total: Total) -> Result<(), PlanError> {
        if let Total::Exact(n) = total {
            self.total.store(n, Ordering::Relaxed);
        }
        Ok(())
    }
}

fn is_cancelled(e: &whereat::At<DecodeError>) -> bool {
    matches!(e.error(), DecodeError::Cancelled(_))
}

fn lossy(w: u32, h: u32) -> Vec<u8> {
    let cfg = EncoderConfig::new_lossy();
    EncodeRequest::new(&cfg, &pattern(w, h, 1), PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

fn lossless(w: u32, h: u32) -> Vec<u8> {
    let cfg = EncoderConfig::new_lossless();
    EncodeRequest::new(&cfg, &pattern(w, h, 2), PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

#[test]
fn encodes_report_percent_to_completion() {
    let (w, h) = (96, 80);
    let px = pattern(w, h, 3);
    for cfg in [EncoderConfig::new_lossy(), EncoderConfig::new_lossless()] {
        let t = tree();
        let with = EncodeRequest::new(&cfg, &px, PixelLayout::Rgba8, w, h)
            .with_pulse(&t)
            .encode()
            .unwrap();
        assert_eq!(observed(&t), (Total::Exact(100), 100));
        // Reporting changes nothing about the output.
        let without = EncodeRequest::new(&cfg, &px, PixelLayout::Rgba8, w, h)
            .encode()
            .unwrap();
        assert_eq!(with, without);
    }
}

#[test]
fn lossy_encode_cancels_part_way() {
    let (w, h) = (256, 256);
    let px = pattern(w, h, 4);
    let cfg = EncoderConfig::new_lossy();
    let pulse = CancelAfter::new(30);
    let err = EncodeRequest::new(&cfg, &px, PixelLayout::Rgba8, w, h)
        .with_pulse(&pulse)
        .encode()
        .unwrap_err();
    assert!(
        matches!(err.error(), zenwebp::EncodeError::Cancelled(_)),
        "{err:?}"
    );
    let done = pulse.advanced();
    assert!((30..100).contains(&done), "stopped at {done}%");
}

#[test]
fn still_decodes_report_rows_to_completion() {
    let config = DecodeConfig::default();
    for (label, data, h) in [
        ("lossy", lossy(96, 70), 70),
        ("lossless", lossless(96, 70), 70),
        ("lossy 1 row", lossy(40, 1), 1),
    ] {
        let t = tree();
        let (with, ..) = DecodeRequest::new(&config, &data)
            .pulse(&t)
            .decode_rgba()
            .unwrap();
        assert_eq!(observed(&t), (Total::Exact(h), h), "{label}");
        let (without, ..) = DecodeRequest::new(&config, &data).decode_rgba().unwrap();
        assert!(with == without, "{label}: pixels");
    }
}

#[test]
fn still_decodes_cancel_part_way() {
    let config = DecodeConfig::default();
    for (label, data) in [("lossy", lossy(256, 256)), ("lossless", lossless(256, 256))] {
        let pulse = CancelAfter::new(64);
        let err = DecodeRequest::new(&config, &data)
            .pulse(&pulse)
            .decode_rgba()
            .unwrap_err();
        assert!(is_cancelled(&err), "{label}: {err:?}");
        assert_eq!(pulse.total.load(Ordering::Relaxed), 256, "{label}");
        let rows = pulse.advanced();
        assert!((64..256).contains(&rows), "{label}: stopped at row {rows}");
    }
}

fn animation(frames: u32) -> Vec<u8> {
    let (w, h) = (32, 24);
    let mut enc = AnimationEncoder::new(
        w,
        h,
        AnimationConfig {
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    for i in 0..frames {
        let cfg = if i % 2 == 0 {
            EncoderConfig::new_lossless()
        } else {
            EncoderConfig::new_lossy()
        };
        enc.add_frame(&pattern(w, h, i + 10), PixelLayout::Rgba8, i * 50, &cfg)
            .unwrap();
    }
    enc.finalize(50).unwrap()
}

#[test]
fn animation_decoder_reports_frames() {
    let data = animation(4);
    let t = tree();
    let mut dec = AnimationDecoder::new(&data).unwrap();
    dec.set_pulse(&t);
    let frames = dec.decode_all().unwrap();
    assert_eq!(frames.len(), 4);
    assert_eq!(observed(&t), (Total::Exact(4), 4));

    // A per-call pulse gets one unit for its frame; a cancelled one poisons
    // the decoder until reset.
    let mut dec = AnimationDecoder::new(&data).unwrap();
    let call = CancelAfter::new(u64::MAX);
    dec.decode_next_with_pulse(&call).unwrap().unwrap();
    assert_eq!(call.advanced(), 1);
    let stopped = CancelAfter::new(0);
    assert!(is_cancelled(
        &dec.decode_next_with_pulse(&stopped).unwrap_err()
    ));
    assert!(dec.decode_next().is_err(), "poisoned until reset");
    dec.reset().unwrap();
    assert_eq!(dec.decode_all().unwrap().len(), 4);
}

#[test]
fn animation_encoder_per_frame_pulse_reports_percent() {
    let (w, h) = (32, 24);
    let mut enc = AnimationEncoder::new(w, h, AnimationConfig::default()).unwrap();
    let t = tree();
    enc.add_frame_with_pulse(
        &pattern(w, h, 1),
        PixelLayout::Rgba8,
        0,
        &EncoderConfig::new_lossy(),
        &t,
    )
    .unwrap();
    assert_eq!(observed(&t), (Total::Exact(100), 100));
    // Cancellation alone, through StopOnly.
    let stopped = CancelAfter::new(0);
    let err = enc
        .add_frame_with_pulse(
            &pattern(w, h, 2),
            PixelLayout::Rgba8,
            40,
            &EncoderConfig::new_lossless(),
            &how_far::StopOnly::borrowed(&stopped),
        )
        .unwrap_err();
    assert!(format!("{err:?}").contains("Cancel"), "{err:?}");
    // The previously accepted frame is still pending and finalizes.
    assert!(enc.finalize(40).is_ok());
}

#[test]
fn zencodec_jobs_report_through_with_pulse() {
    // Still decode: rows.
    let data = lossy(64, 48);
    let t = tree();
    let out = WebpDecoderConfig::new()
        .job()
        .with_pulse(t.share().unwrap())
        .decoder(Cow::Borrowed(&data), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(out.pixels().rows(), 48);
    assert_eq!(observed(&t), (Total::Exact(48), 48));

    // Animation decode: frames.
    let anim = animation(3);
    let t = tree();
    let mut dec = WebpDecoderConfig::new()
        .job()
        .with_pulse(t.share().unwrap())
        .animation_frame_decoder(Cow::Borrowed(&anim), &[])
        .unwrap();
    while dec.render_next_frame(None).unwrap().is_some() {}
    assert_eq!(observed(&t), (Total::Exact(3), 3));

    // Still encode: percent.
    let px = zenpixels::PixelBuffer::from_vec(
        pattern(40, 30, 5),
        40,
        30,
        zenpixels::PixelDescriptor::RGBA8_SRGB,
    )
    .unwrap();
    let t = tree();
    WebpEncoderConfig::lossy()
        .job()
        .with_pulse(t.share().unwrap())
        .encoder()
        .unwrap()
        .encode(px.as_slice())
        .unwrap();
    assert_eq!(observed(&t), (Total::Exact(100), 100));

    // Animation encode: one unit per accepted frame.
    let t = tree();
    let mut enc = WebpEncoderConfig::lossless()
        .job()
        .with_pulse(t.share().unwrap())
        .animation_frame_encoder()
        .unwrap();
    for _ in 0..3 {
        enc.push_frame_timed(
            px.as_slice(),
            zencodec::animation::FrameDuration::from_millis(20),
            None,
        )
        .unwrap();
    }
    enc.finish(None).unwrap();
    assert_eq!(observed(&t).1, 3);

    // A cancelled pulse stops a zencodec decode.
    let t = PulseTree::new(Phase::new("op", Total::Unknown), CancelAfter::new(0));
    let res = WebpDecoderConfig::new()
        .job()
        .with_pulse(t.share().unwrap())
        .decoder(Cow::Borrowed(&data), &[])
        .unwrap()
        .decode();
    assert!(res.is_err());
}
