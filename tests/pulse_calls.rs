//! A call's pulse and the job's pulse each cancel every animation call on their
//! own, each with its own reason, whether the job's pulse can report or not.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use how_far::{
    Child, Execution, PhaseSpec, PlanError, Pulse, Report, SharedPulse, Stop, StopReason,
};
use zencodec::ErrorCategory;
use zencodec::animation::FrameDuration;
use zencodec::decode::{
    AnimationFrameDecoder as _, DecodeJob as _, DecodeRowSink, DecoderConfig as _,
};
use zencodec::encode::{AnimationFrameEncoder as _, EncodeJob as _, EncoderConfig as _};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSliceMut};
use zenwebp::mux::{AnimationConfig, AnimationEncoder};
use zenwebp::zencodec::{WebpAnimationFrameDecoder, WebpAnimationFrameEncoder};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};
use zenwebp::{EncoderConfig, PixelLayout};

fn pattern(w: u32, h: u32, seed: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let n = i.wrapping_mul(2_654_435_761) ^ seed;
            [(n >> 3) as u8, (n >> 11) as u8, (i * 7) as u8, 255]
        })
        .collect()
}

fn animation() -> Vec<u8> {
    let config = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut enc = AnimationEncoder::new(32, 24, config).unwrap();
    for i in 0..3 {
        enc.add_frame(
            &pattern(32, 24, i),
            PixelLayout::Rgba8,
            i * 50,
            &EncoderConfig::new_lossless(),
        )
        .unwrap();
    }
    enc.finalize(50).unwrap()
}

/// Stopped with `reason` once switched on.
struct Switch {
    on: AtomicBool,
    reason: StopReason,
    reports: bool,
}

impl Switch {
    fn new(reason: StopReason, reports: bool) -> Arc<Self> {
        Arc::new(Self {
            on: AtomicBool::new(false),
            reason,
            reports,
        })
    }
    fn set(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }
}

impl Stop for Switch {
    fn check(&self) -> Result<(), StopReason> {
        if self.on.load(Ordering::Relaxed) {
            Err(self.reason)
        } else {
            Ok(())
        }
    }
}

impl Report for Switch {
    fn advance(&self, _: u64) {}
    fn may_report(&self) -> bool {
        self.reports
    }
}

impl Pulse for Switch {
    fn split(&self, _: Execution, _: &[PhaseSpec<'_>]) -> Result<Vec<Child<'_>>, PlanError> {
        Err(PlanError::Unsupported)
    }
}

/// Which pulse is switched on, and what the call then returns.
const CASES: [(bool, Option<bool>, Option<StopReason>); 5] = [
    // (job on, call: None = no call pulse / Some(on)), expected stop reason
    (true, None, Some(StopReason::TimedOut)),
    (true, Some(false), Some(StopReason::TimedOut)),
    (false, Some(true), Some(StopReason::Cancelled)),
    (false, Some(false), None),
    (false, None, None),
];

fn expected(reason: Option<StopReason>) -> Result<(), ErrorCategory> {
    reason.map_or(Ok(()), |r| Err(ErrorCategory::Stopped(r)))
}

fn category(error: whereat::At<zencodec::CodecError>) -> ErrorCategory {
    error.error().category()
}

type Call = fn(
    &mut WebpAnimationFrameEncoder,
    &PixelBuffer,
    Option<&dyn zencodec::Pulse>,
) -> Result<(), ErrorCategory>;

#[test]
fn call_and_job_pulses_each_stop_the_animation_encoder() {
    let px =
        PixelBuffer::from_vec(pattern(32, 24, 9), 32, 24, PixelDescriptor::RGBA8_SRGB).unwrap();
    let calls: [(&str, Call); 3] = [
        ("push_frame", |e, px, p| {
            e.push_frame(px.as_slice(), 20, p).map_err(category)
        }),
        ("push_frame_exact", |e, px, p| {
            e.push_frame_exact(px.as_slice(), FrameDuration::new(30, 1000).unwrap(), p)
                .map_err(category)
        }),
        ("push_frame_timed", |e, px, p| {
            e.push_frame_timed(px.as_slice(), FrameDuration::new(1, 30).unwrap(), p)
                .map(drop)
                .map_err(category)
        }),
    ];
    for reports in [false, true] {
        for (job_on, call_on, reason) in CASES {
            let job = Switch::new(StopReason::TimedOut, reports);
            let call = Switch::new(StopReason::Cancelled, false);
            let pulse = call_on.map(|_| &*call as &dyn zencodec::Pulse);
            let new = || {
                WebpEncoderConfig::lossless()
                    .job()
                    .with_pulse(SharedPulse::new(Arc::clone(&job)))
                    .animation_frame_encoder()
                    .unwrap()
            };
            let case = format!("job reports {reports}, job on {job_on}, call {call_on:?}");
            for (what, push) in calls {
                let mut encoder = new();
                job.set(job_on);
                call.set(call_on == Some(true));
                assert_eq!(
                    push(&mut encoder, &px, pulse),
                    expected(reason),
                    "{what}: {case}"
                );
                job.set(false);
                call.set(false);
            }
            let mut encoder = new();
            encoder.push_frame(px.as_slice(), 20, None).unwrap();
            job.set(job_on);
            call.set(call_on == Some(true));
            assert_eq!(
                encoder.finish(pulse).map(drop).map_err(category),
                expected(reason),
                "finish: {case}"
            );
        }
    }
}

#[derive(Default)]
struct Rows {
    buf: Vec<u8>,
    rows: u32,
}

impl DecodeRowSink for Rows {
    fn provide_next_buffer(
        &mut self,
        _y: u32,
        height: u32,
        width: u32,
        descriptor: PixelDescriptor,
    ) -> Result<PixelSliceMut<'_>, zencodec::decode::SinkError> {
        let stride = width as usize * descriptor.bytes_per_pixel();
        self.buf.resize(height as usize * stride, 0);
        self.rows += height;
        Ok(PixelSliceMut::new(&mut self.buf, width, height, stride, descriptor).unwrap())
    }
}

type Render =
    fn(&mut WebpAnimationFrameDecoder, Option<&dyn zencodec::Pulse>) -> Result<(), ErrorCategory>;

#[test]
fn call_and_job_pulses_each_stop_the_animation_decoder() {
    let data = animation();
    let renders: [(&str, Render); 3] = [
        ("render_next_frame", |d, p| {
            d.render_next_frame(p).map(drop).map_err(category)
        }),
        ("render_next_frame_owned", |d, p| {
            d.render_next_frame_owned(p).map(drop).map_err(category)
        }),
        ("render_next_frame_to_sink", |d, p| {
            let mut rows = Rows::default();
            let result = d
                .render_next_frame_to_sink(p, &mut rows)
                .map(drop)
                .map_err(category);
            if result.is_err() {
                assert_eq!(rows.rows, 0, "a stopped render wrote rows");
            }
            result
        }),
    ];
    for reports in [false, true] {
        for (job_on, call_on, reason) in CASES {
            let job = Switch::new(StopReason::TimedOut, reports);
            let call = Switch::new(StopReason::Cancelled, false);
            let pulse = call_on.map(|_| &*call as &dyn zencodec::Pulse);
            let case = format!("job reports {reports}, job on {job_on}, call {call_on:?}");
            for (what, render) in renders {
                let mut decoder = WebpDecoderConfig::new()
                    .job()
                    .with_pulse(SharedPulse::new(Arc::clone(&job)))
                    .animation_frame_decoder(Cow::Borrowed(&data), &[])
                    .unwrap();
                job.set(job_on);
                call.set(call_on == Some(true));
                assert_eq!(
                    render(&mut decoder, pulse),
                    expected(reason),
                    "{what}: {case}"
                );
                job.set(false);
                call.set(false);
            }
        }
    }
}
