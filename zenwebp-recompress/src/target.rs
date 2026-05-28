//! Anchor tables mapping target zensim Profile A → encoder quality dial.
//!
//! These are placeholder linear anchors until `zwr-calibrate` runs the full
//! corpus sweep. They are intentionally conservative: the projected zensim-A
//! is biased ~1.5 points above the linear estimate to reduce undershoot in
//! the one-shot path.
//!
//! Re-fit recipe — see DESIGN.md "Calibration corpus":
//!
//! ```text
//! cargo run --release -p zwr-calibrate -- \
//!   --sources /mnt/v/input/codec-corpus-sc \
//!   --encoders libwebp,zenwebp \
//!   --quality-range 20:100:2 \
//!   --targets 0:100:5 \
//!   --strategies coeff,deblock,reencode,vp8l,remux \
//!   --output zenwebp-recompress/src/calibration/data.parquet
//! ```

/// Hand-fit anchor: target zensim-A → libwebp `-q` for photo content.
///
/// Derived from a 50-image cid22 sweep at libwebp 1.3.2, `-m 6`,
/// 4:2:0 subsampling, sRGB. Photo content. Re-fit with `zwr-calibrate`.
const LIBWEBP_PHOTO_ANCHORS: &[(f32, u8)] = &[
    (50.0, 30),
    (60.0, 45),
    (70.0, 60),
    (75.0, 68),
    (80.0, 75),
    (85.0, 82),
    (90.0, 88),
    (95.0, 95),
];

/// Hand-fit anchor for zenwebp's target-zensim path. The encoder already
/// closes the loop internally so this is identity-shaped, but kept here so
/// the router can use a single function regardless of encoder family.
#[allow(dead_code)] // Consumed when zenwebp encoder family path lands.
const ZENWEBP_PHOTO_ANCHORS: &[(f32, u8)] = LIBWEBP_PHOTO_ANCHORS;

/// Map target zensim-A → libwebp `-q` quality dial.
///
/// Interpolates linearly between anchor pairs; clamps to `[1, 100]`.
pub fn target_zensim_a_to_libwebp_q(target_zensim_a: f32) -> u8 {
    interpolate(LIBWEBP_PHOTO_ANCHORS, target_zensim_a)
}

/// Map target zensim-A → zenwebp `quality` for the target-zensim path.
#[allow(dead_code)] // Consumed when zenwebp encoder family path lands.
pub fn target_zensim_a_to_zenwebp_q(target_zensim_a: f32) -> u8 {
    interpolate(ZENWEBP_PHOTO_ANCHORS, target_zensim_a)
}

fn interpolate(anchors: &[(f32, u8)], target: f32) -> u8 {
    if anchors.is_empty() {
        return 75;
    }
    if target <= anchors[0].0 {
        return anchors[0].1;
    }
    if target >= anchors[anchors.len() - 1].0 {
        return anchors[anchors.len() - 1].1;
    }
    for w in anchors.windows(2) {
        let (lo_t, lo_q) = w[0];
        let (hi_t, hi_q) = w[1];
        if target >= lo_t && target <= hi_t {
            let frac = (target - lo_t) / (hi_t - lo_t).max(f32::EPSILON);
            let q = (lo_q as f32) + frac * (hi_q as f32 - lo_q as f32);
            return q.round().clamp(1.0, 100.0) as u8;
        }
    }
    anchors[anchors.len() - 1].1
}

/// Estimate the cumulative zensim-A vs the unknown reference from an encoder
/// family and source quality. This is the inverse of the table above plus a
/// small bias for the round-trip generation loss already baked into the
/// source.
pub(crate) fn source_q_to_zensim_a_estimate(_family: EncoderFamily, source_q: f32) -> f32 {
    // Linear bridge: source_q is roughly in 1:1 with the zensim-A score the
    // encoder would have hit if its target had been our score. Subtract a
    // small "generation loss" bias because the source's bits are already
    // quantized — recompressing to the same nominal Q gives slightly lower
    // zensim-A than encoding fresh from a clean reference.
    let base = 50.0 + (source_q.clamp(1.0, 100.0) - 30.0) * 0.7;
    (base - 1.5).clamp(0.0, 100.0)
}

// Re-export for source.rs convenience.
pub(crate) use crate::source::EncoderFamily;
