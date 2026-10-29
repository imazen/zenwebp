//! A job's pulse reaches native work once per intended poll, in both of how-far's
//! reporting modes. A pulse that cannot report (`may_report() == false`, which
//! zencodec's deprecated `with_stop` builds) polls exactly where the legacy stop
//! token polls. A pulse that reports polls there too, plus the still encoders'
//! percent callbacks, which check the pulse they report to; decodes report rows
//! and frames without polling. Neither mode suppresses cancellation, and a
//! reporting pulse keeps its reports.
//!
//! Only setters both the stop and the pulse traits have are used (`with_stop`,
//! `with_pulse`, per-call `None`), so this builds against either zencodec.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]
#![allow(deprecated)] // The legacy `with_stop` arm, on purpose.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use how_far::{
    Child, Execution, PhaseSpec, PlanError, Pulse, Report, SharedPulse, Stop, StopReason, Total,
};
use zencodec::ErrorCategory;
use zencodec::animation::FrameDuration;
use zencodec::decode::{
    AnimationFrameDecoder as _, Decode as _, DecodeJob as _, DecodeRowSink, DecoderConfig as _,
};
use zencodec::encode::{
    AnimationFrameEncoder as _, EncodeJob as _, Encoder as _, EncoderConfig as _,
};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSliceMut};
use zenwebp::mux::{AnimationConfig, AnimationEncoder};
use zenwebp::zencodec::{WebpDecodeJob, WebpDecoderConfig, WebpEncodeJob, WebpEncoderConfig};
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

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

fn frame(w: u32, h: u32, seed: u32) -> PixelBuffer {
    PixelBuffer::from_vec(pattern(w, h, seed), w, h, PixelDescriptor::RGBA8_SRGB).unwrap()
}

fn still(cfg: &EncoderConfig, w: u32, h: u32) -> Vec<u8> {
    EncodeRequest::new(cfg, &pattern(w, h, 1), PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

/// Frames alternate lossless and lossy.
fn animation(frames: u32) -> Vec<u8> {
    let (w, h) = (32, 24);
    let config = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut enc = AnimationEncoder::new(w, h, config).unwrap();
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

/// Counts its polls and reports; times out from poll `stop_from` on.
struct Probe {
    polls: AtomicU64,
    advanced: AtomicU64,
    total: AtomicU64,
    stop_from: u64,
    reports: bool,
}

impl Probe {
    fn new(reports: bool, stop_from: u64) -> Arc<Self> {
        Arc::new(Self {
            polls: AtomicU64::new(0),
            advanced: AtomicU64::new(0),
            total: AtomicU64::new(0),
            stop_from,
            reports,
        })
    }
    fn polls(&self) -> u64 {
        self.polls.load(Ordering::Relaxed)
    }
    fn reports(&self) -> (u64, u64) {
        (
            self.total.load(Ordering::Relaxed),
            self.advanced.load(Ordering::Relaxed),
        )
    }
}

impl Stop for Probe {
    fn check(&self) -> Result<(), StopReason> {
        if self.polls.fetch_add(1, Ordering::Relaxed) >= self.stop_from {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}

impl Report for Probe {
    fn advance(&self, completed: u64) {
        self.advanced.fetch_add(completed, Ordering::Relaxed);
    }
    fn may_report(&self) -> bool {
        self.reports
    }
}

impl Pulse for Probe {
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

/// How a job receives the probe.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// The deprecated `with_stop` token.
    Legacy,
    /// `with_pulse`, with a pulse that cannot report.
    Quiet,
    /// `with_pulse`, with a pulse that reports.
    Reporting,
}

const ARMS: [Arm; 3] = [Arm::Legacy, Arm::Quiet, Arm::Reporting];

impl Arm {
    fn probe(self, stop_from: u64) -> Arc<Probe> {
        Probe::new(matches!(self, Arm::Reporting), stop_from)
    }
    fn decode_job(self, probe: &Arc<Probe>) -> WebpDecodeJob {
        let job = WebpDecoderConfig::new().job();
        match self {
            Arm::Legacy => job.with_stop(zencodec::StopToken::new(Arc::clone(probe))),
            Arm::Quiet | Arm::Reporting => job.with_pulse(SharedPulse::new(Arc::clone(probe))),
        }
    }
    fn encode_job(self, config: WebpEncoderConfig, probe: &Arc<Probe>) -> WebpEncodeJob {
        let job = config.job();
        match self {
            Arm::Legacy => job.with_stop(zencodec::StopToken::new(Arc::clone(probe))),
            Arm::Quiet | Arm::Reporting => job.with_pulse(SharedPulse::new(Arc::clone(probe))),
        }
    }
}

fn category(error: whereat::At<zencodec::CodecError>) -> ErrorCategory {
    error.error().category()
}

/// Collects decoded rows.
#[derive(Default)]
struct Rows {
    buf: Vec<u8>,
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
        Ok(PixelSliceMut::new(&mut self.buf, width, height, stride, descriptor).unwrap())
    }
}

fn decode(job: WebpDecodeJob, data: &[u8]) -> Result<(), ErrorCategory> {
    job.decoder(Cow::Borrowed(data), &[])
        .and_then(|decoder| decoder.decode())
        .map(drop)
        .map_err(category)
}

#[derive(Clone, Copy)]
enum Render {
    Borrowed,
    Owned,
    Sink,
}

fn render_all(job: WebpDecodeJob, data: &[u8], render: Render) -> Result<(), ErrorCategory> {
    let mut decoder = job
        .animation_frame_decoder(Cow::Borrowed(data), &[])
        .map_err(category)?;
    loop {
        let more = match render {
            Render::Borrowed => decoder.render_next_frame(None).map(|f| f.is_some()),
            Render::Owned => decoder.render_next_frame_owned(None).map(|f| f.is_some()),
            Render::Sink => decoder
                .render_next_frame_to_sink(None, &mut Rows::default())
                .map(|f| f.is_some()),
        }
        .map_err(category)?;
        if !more {
            return Ok(());
        }
    }
}

/// Three frames, through every push and finish.
fn animate(job: WebpEncodeJob, px: &PixelBuffer) -> Result<(), ErrorCategory> {
    let mut encoder = job.animation_frame_encoder().map_err(category)?;
    encoder
        .push_frame(px.as_slice(), 20, None)
        .map_err(category)?;
    encoder
        .push_frame_exact(px.as_slice(), FrameDuration::new(30, 1000).unwrap(), None)
        .map_err(category)?;
    encoder
        .push_frame_timed(px.as_slice(), FrameDuration::new(1, 30).unwrap(), None)
        .map_err(category)?;
    encoder.finish(None).map(drop).map_err(category)
}

type Run<'a> = &'a dyn Fn(Arm, &Arc<Probe>) -> Result<(), ErrorCategory>;

/// Polls of `run` under `arm`, with a probe that never stops.
fn polls(run: Run<'_>, arm: Arm) -> u64 {
    let probe = arm.probe(u64::MAX);
    run(arm, &probe).unwrap_or_else(|error| panic!("{arm:?}: {error:?}"));
    probe.polls()
}

/// Every arm polls the probe exactly as often as the legacy token.
fn same_polls(ops: &[(&str, Run<'_>)]) {
    for (what, run) in ops {
        let [legacy, quiet, reporting] = ARMS.map(|arm| polls(*run, arm));
        assert!(legacy > 0, "{what}: never polled");
        assert_eq!(
            (quiet, reporting),
            (legacy, legacy),
            "{what}: polls with the legacy token {legacy}, a quiet pulse {quiet}, a reporting pulse {reporting}"
        );
    }
}

#[test]
fn still_decodes_poll_the_job_pulse_once_per_checkpoint() {
    let lossy = still(&EncoderConfig::new_lossy(), 48, 40);
    let lossless = still(&EncoderConfig::new_lossless(), 48, 40);
    same_polls(&[
        ("lossy decode", &|arm, p| decode(arm.decode_job(p), &lossy)),
        ("lossless decode", &|arm, p| {
            decode(arm.decode_job(p), &lossless)
        }),
        ("push_decoder", &|arm, p| {
            arm.decode_job(p)
                .push_decoder(Cow::Borrowed(&lossy), &mut Rows::default(), &[])
                .map(drop)
                .map_err(category)
        }),
        ("streaming_decoder", &|arm, p| {
            arm.decode_job(p)
                .streaming_decoder(Cow::Borrowed(&lossy), &[])
                .map(drop)
                .map_err(category)
        }),
    ]);
}

#[test]
fn animation_decodes_poll_the_job_pulse_once_per_checkpoint() {
    let data = animation(3);
    same_polls(&[
        ("render_next_frame", &|arm, p| {
            render_all(arm.decode_job(p), &data, Render::Borrowed)
        }),
        ("render_next_frame_owned", &|arm, p| {
            render_all(arm.decode_job(p), &data, Render::Owned)
        }),
        ("render_next_frame_to_sink", &|arm, p| {
            render_all(arm.decode_job(p), &data, Render::Sink)
        }),
    ]);
}

#[test]
fn the_animation_encoder_and_inventory_poll_the_job_pulse_once_per_checkpoint() {
    let px = frame(32, 24, 7);
    #[cfg(feature = "inventory")]
    let data = still(&EncoderConfig::new_lossy(), 48, 40);
    same_polls(&[
        ("lossy animation", &|arm, p| {
            animate(arm.encode_job(WebpEncoderConfig::lossy(), p), &px)
        }),
        ("lossless animation", &|arm, p| {
            animate(arm.encode_job(WebpEncoderConfig::lossless(), p), &px)
        }),
        #[cfg(feature = "inventory")]
        ("inventory", &|arm, p| {
            arm.decode_job(p)
                .inventory(&data)
                .map(drop)
                .map_err(category)
        }),
    ]);
}

/// The still encoders' percent callbacks check the pulse they report to, so
/// a reporting job pulse adds exactly the polls a reporting pulse adds to the
/// native encode; a quiet one adds none.
#[test]
fn still_encodes_add_only_the_native_percent_checks() {
    let (w, h) = (48, 40);
    let px = frame(w, h, 3);
    for (what, config) in [
        ("lossy", WebpEncoderConfig::lossy()),
        ("lossless", WebpEncoderConfig::lossless()),
    ] {
        let native = |as_pulse: bool| {
            let probe = Probe::new(as_pulse, u64::MAX);
            let pixels = pattern(w, h, 3);
            let req = EncodeRequest::new(config.inner(), &pixels, PixelLayout::Rgba8, w, h);
            let req = if as_pulse {
                req.with_pulse(&*probe)
            } else {
                req.with_stop(&*probe)
            };
            req.encode().unwrap();
            probe.polls()
        };
        let (stop_polls, pulse_polls) = (native(false), native(true));
        let run = |arm: Arm, p: &Arc<Probe>| {
            arm.encode_job(config.clone(), p)
                .encoder()
                .and_then(|encoder| encoder.encode(px.as_slice()))
                .map(drop)
                .map_err(category)
        };
        let [legacy, quiet, reporting] = ARMS.map(|arm| polls(&run, arm));
        let counts = format!(
            "native stop {stop_polls}, native pulse {pulse_polls}; legacy {legacy}, quiet {quiet}, reporting {reporting}"
        );
        assert!(
            pulse_polls > stop_polls,
            "{what}: no percent checks ({counts})"
        );
        assert_eq!(quiet, legacy, "{what}: {counts}");
        assert_eq!(
            reporting + stop_polls,
            legacy + pulse_polls,
            "{what}: {counts}"
        );
    }
}

/// A stop at the first or the last poll ends every operation with its reason,
/// in every arm: a pulse that cannot report is still the job's cancellation.
#[test]
fn neither_reporting_mode_suppresses_cancellation() {
    let lossy = still(&EncoderConfig::new_lossy(), 48, 40);
    let lossless = still(&EncoderConfig::new_lossless(), 48, 40);
    let data = animation(3);
    let px = frame(32, 24, 7);
    let ops: &[(&str, Run<'_>)] = &[
        ("lossy decode", &|arm, p| decode(arm.decode_job(p), &lossy)),
        ("lossless decode", &|arm, p| {
            decode(arm.decode_job(p), &lossless)
        }),
        ("push_decoder", &|arm, p| {
            arm.decode_job(p)
                .push_decoder(Cow::Borrowed(&lossy), &mut Rows::default(), &[])
                .map(drop)
                .map_err(category)
        }),
        ("animation decode", &|arm, p| {
            render_all(arm.decode_job(p), &data, Render::Borrowed)
        }),
        ("lossy encode", &|arm, p| {
            arm.encode_job(WebpEncoderConfig::lossy(), p)
                .encoder()
                .and_then(|encoder| encoder.encode(px.as_slice()))
                .map(drop)
                .map_err(category)
        }),
        ("lossless encode", &|arm, p| {
            arm.encode_job(WebpEncoderConfig::lossless(), p)
                .encoder()
                .and_then(|encoder| encoder.encode(px.as_slice()))
                .map(drop)
                .map_err(category)
        }),
        ("animation encode", &|arm, p| {
            animate(arm.encode_job(WebpEncoderConfig::lossy(), p), &px)
        }),
    ];
    for (what, run) in ops {
        for arm in ARMS {
            let total = polls(*run, arm);
            for stop_from in [0, total - 1] {
                let probe = arm.probe(stop_from);
                assert_eq!(
                    run(arm, &probe),
                    Err(ErrorCategory::Stopped(StopReason::TimedOut)),
                    "{what}, {arm:?}: stop from poll {stop_from} of {total}"
                );
            }
        }
    }
}

/// A job pulse that can report receives the documented units: percent for
/// still encodes, rows for still decodes, frames for animation decodes, one
/// unit per accepted frame for the animation encoder.
#[test]
fn a_reporting_job_pulse_keeps_its_reports() {
    let lossy = still(&EncoderConfig::new_lossy(), 48, 40);
    let data = animation(3);
    let px = frame(32, 24, 7);
    let reports = |run: Run<'_>| {
        let probe = Arm::Reporting.probe(u64::MAX);
        run(Arm::Reporting, &probe).unwrap();
        probe.reports()
    };
    assert_eq!(
        reports(&|arm, p| decode(arm.decode_job(p), &lossy)),
        (40, 40)
    );
    assert_eq!(
        reports(&|arm, p| render_all(arm.decode_job(p), &data, Render::Borrowed)),
        (3, 3)
    );
    assert_eq!(
        reports(&|arm, p| {
            arm.encode_job(WebpEncoderConfig::lossy(), p)
                .encoder()
                .and_then(|encoder| encoder.encode(px.as_slice()))
                .map(drop)
                .map_err(category)
        }),
        (100, 100)
    );
    assert_eq!(
        reports(&|arm, p| animate(arm.encode_job(WebpEncoderConfig::lossy(), p), &px)).1,
        3
    );
}
