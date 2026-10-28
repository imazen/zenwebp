//! A stop's reason survives every native decode checkpoint. The VP8 row checks
//! and the VP8L block checks return the decoder's compact internal error, so a
//! reason it does not keep comes back as `Cancelled`.
#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};

use enough::{Stop, StopReason};
use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder};
use zenwebp::{
    DecodeConfig, DecodeError, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout,
};

fn pattern(w: u32, h: u32, seed: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let n = i.wrapping_mul(2_654_435_761) ^ seed;
            [
                (n >> 3) as u8,
                (n >> 11) as u8,
                (i * 7) as u8,
                ((n >> 19) as u8) | 128,
            ]
        })
        .collect()
}

fn still(cfg: &EncoderConfig) -> Vec<u8> {
    EncodeRequest::new(cfg, &pattern(48, 40, 1), PixelLayout::Rgba8, 48, 40)
        .encode()
        .unwrap()
}

fn animation() -> Vec<u8> {
    let config = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut enc = AnimationEncoder::new(32, 24, config).unwrap();
    for i in 0..3 {
        let cfg = if i % 2 == 0 {
            EncoderConfig::new_lossless()
        } else {
            EncoderConfig::new_lossy()
        };
        enc.add_frame(&pattern(32, 24, i + 10), PixelLayout::Rgba8, i * 50, &cfg)
            .unwrap();
    }
    enc.finalize(50).unwrap()
}

/// Counts polls; times out from poll `from` on.
struct TimeoutFrom {
    polls: AtomicU64,
    from: u64,
}

impl TimeoutFrom {
    fn new(from: u64) -> Self {
        Self {
            polls: AtomicU64::new(0),
            from,
        }
    }
}

impl Stop for TimeoutFrom {
    fn check(&self) -> Result<(), StopReason> {
        if self.polls.fetch_add(1, Ordering::Relaxed) >= self.from {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}

type Decode<'a> = &'a dyn Fn(&TimeoutFrom) -> Result<(), whereat::At<DecodeError>>;

/// Every poll, timing out, ends the decode with the timeout.
fn every_poll_keeps_the_reason(label: &str, decode: Decode<'_>) {
    let count = TimeoutFrom::new(u64::MAX);
    decode(&count).unwrap();
    let polls = count.polls.load(Ordering::Relaxed);
    assert!(polls > 1, "{label}: {polls} polls");
    for from in 0..polls {
        let error = decode(&TimeoutFrom::new(from)).unwrap_err();
        assert!(
            matches!(error.error(), DecodeError::Cancelled(StopReason::TimedOut)),
            "{label}: timeout from poll {from} of {polls} came back as {error:?}"
        );
    }
}

#[test]
fn still_decodes_keep_a_timeout_at_every_poll() {
    let config = DecodeConfig::default();
    for (label, data) in [
        ("lossy", still(&EncoderConfig::new_lossy())),
        ("lossless", still(&EncoderConfig::new_lossless())),
    ] {
        every_poll_keeps_the_reason(label, &|stop| {
            DecodeRequest::new(&config, &data)
                .stop(stop)
                .decode_rgba()
                .map(drop)
        });
    }
}

#[test]
fn animation_decodes_keep_a_timeout_at_every_poll() {
    let data = animation();
    every_poll_keeps_the_reason("animation", &|stop| {
        let mut decoder = AnimationDecoder::new(&data)?;
        decoder.set_stop(stop);
        decoder.decode_all().map(drop)
    });
}
