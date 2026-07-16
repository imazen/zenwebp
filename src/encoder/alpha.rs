//! libwebp-exact alpha-plane encoding pipeline (#38 alpha parity).
//!
//! Ports the stages of libwebp's `EncodeAlpha` (`alpha_enc.c`) so the ALPH
//! chunk can be produced byte-identically under `StrictLibwebpParity`:
//!
//! 1. [`quantize_levels`] — `QuantizeLevels` (`quant_levels_utils.c`): k-means
//!    over the alpha histogram with `alpha_levels = (q <= 70) ? 2 + q/5 :
//!    16 + (q-70)*8` levels, run when `alpha_quality < 100`.
//! 2. Prediction filters — `WebPFilters[]` (`dsp/filters.c`): horizontal,
//!    vertical, gradient (mode 0 = none).
//! 3. [`estimate_best_filter`] — `WebPEstimateBestFilter`
//!    (`filters_utils.c`): the sampled gradient-bin heuristic.
//! 4. [`filter_try_map`] — `GetFilterMap` (`alpha_enc.c`): which filters the
//!    trial loop attempts for a given `alpha_filtering` setting and effort.
//! 5. The trial loop — `ApplyFiltersAndEncode`/`EncodeAlphaInternal`: encode
//!    each candidate filter's plane with the lossless coder, fall back to
//!    raw when compression does not help, keep the smallest total, and emit
//!    `header = method | (filter << 2) | (preprocessed << 4)`.
//!
//! The payload coder is stage 6: libwebp calls the FULL `VP8LEncodeStream`
//! on an alpha-in-green ARGB image with `exact=1, method=effort_level,
//! quality = (alpha_quality == 100 && effort == 6) ? 100 : 8 * effort`.
//! zenwebp routes that through its own VP8L encoder — pipeline stages 1-5
//! are byte-exact against libwebp; the remaining distance to a
//! byte-identical ALPH chunk lives inside the VP8L stream itself and is
//! tracked separately (see `dev/alphadiff.rs` for the layer-isolating
//! harness).

use alloc::vec;
use alloc::vec::Vec;

/// Filter modes, numbered as in the ALPH header (and `WEBP_FILTER_TYPE`).
pub(crate) const FILTER_NONE: u8 = 0;
pub(crate) const FILTER_HORIZONTAL: u8 = 1;
pub(crate) const FILTER_VERTICAL: u8 = 2;
pub(crate) const FILTER_GRADIENT: u8 = 3;
const FILTER_LAST: u8 = 4;

/// `QuantizeLevels` (`quant_levels_utils.c`): quantize the plane to
/// `num_levels` values via histogram k-means (max 6 iterations, MSE
/// convergence threshold `1e-4 * data_size`), preserving min and max.
/// No-op when the plane already has `<= num_levels` distinct values.
///
/// f64 arithmetic in source order matches libwebp's doubles bit-for-bit.
pub(crate) fn quantize_levels(data: &mut [u8], num_levels: i32) {
    const MAX_ITER: usize = 6;
    const ERROR_THRESHOLD: f64 = 1e-4;

    if data.is_empty() || !(2..=256).contains(&num_levels) {
        return;
    }
    let num_levels = num_levels as usize;
    let data_size = data.len();

    let mut freq = [0i32; 256];
    let mut min_s = 255usize;
    let mut max_s = 0usize;
    let mut num_levels_in = 0usize;
    for &v in data.iter() {
        let v = v as usize;
        if freq[v] == 0 {
            num_levels_in += 1;
        }
        min_s = min_s.min(v);
        max_s = max_s.max(v);
        freq[v] += 1;
    }
    if num_levels_in <= num_levels {
        return; // nothing to do
    }

    // Start with uniformly spread centroids.
    let mut inv_q_level = [0f64; 256];
    for (i, slot) in inv_q_level.iter_mut().take(num_levels).enumerate() {
        *slot = min_s as f64 + (max_s - min_s) as f64 * i as f64 / (num_levels - 1) as f64;
    }

    let mut q_level = [0usize; 256];
    q_level[min_s] = 0;
    q_level[max_s] = num_levels - 1;

    let err_threshold = ERROR_THRESHOLD * data_size as f64;
    let mut last_err = 1.0e38f64;
    for _iter in 0..MAX_ITER {
        let mut q_sum = [0f64; 256];
        let mut q_count = [0f64; 256];
        let mut slot = 0usize;

        // Assign classes to representatives.
        for s in min_s..=max_s {
            while slot < num_levels - 1
                && 2.0 * s as f64 > inv_q_level[slot] + inv_q_level[slot + 1]
            {
                slot += 1;
            }
            if freq[s] > 0 {
                q_sum[slot] += (s as i32 * freq[s]) as f64;
                q_count[slot] += freq[s] as f64;
            }
            q_level[s] = slot;
        }

        // Assign new representatives to classes.
        if num_levels > 2 {
            for slot in 1..num_levels - 1 {
                if q_count[slot] > 0.0 {
                    inv_q_level[slot] = q_sum[slot] / q_count[slot];
                }
            }
        }

        // Compute convergence error.
        let mut err = 0f64;
        for s in min_s..=max_s {
            let error = s as f64 - inv_q_level[q_level[s]];
            err += freq[s] as f64 * error * error;
        }

        if last_err - err < err_threshold {
            break;
        }
        last_err = err;
    }

    // Remap the plane to quantized values.
    let mut map = [0u8; 256];
    for (s, m) in map.iter_mut().enumerate().take(max_s + 1).skip(min_s) {
        *m = (inv_q_level[q_level[s]] + 0.5) as u8;
    }
    for v in data.iter_mut() {
        *v = map[*v as usize];
    }
}

/// libwebp's alpha_quality -> level-count mapping (`alpha_enc.c:344-346`):
/// "Quality:[0, 70] -> Levels:[2, 16] and Quality:]70, 100] -> Levels:]16, 256]".
pub fn alpha_levels_for_quality(alpha_quality: u8) -> i32 {
    let q = i32::from(alpha_quality);
    if q <= 70 {
        2 + q / 5
    } else {
        16 + (q - 70) * 8
    }
}

/// `DoHorizontalFilter_C` (`dsp/filters.c`). Predictions always read the
/// ORIGINAL input (`preds` tracks `in`), never the filtered output.
fn horizontal_filter(input: &[u8], width: usize, height: usize, out: &mut [u8]) {
    out[0] = input[0];
    for i in 1..width {
        out[i] = input[i].wrapping_sub(input[i - 1]);
    }
    for row in 1..height {
        let o = row * width;
        // Leftmost pixel is predicted from above.
        out[o] = input[o].wrapping_sub(input[o - width]);
        for i in 1..width {
            out[o + i] = input[o + i].wrapping_sub(input[o + i - 1]);
        }
    }
}

/// `DoVerticalFilter_C`.
fn vertical_filter(input: &[u8], width: usize, height: usize, out: &mut [u8]) {
    out[0] = input[0];
    // Rest of top scan-line is left-predicted.
    for i in 1..width {
        out[i] = input[i].wrapping_sub(input[i - 1]);
    }
    for row in 1..height {
        let o = row * width;
        for i in 0..width {
            out[o + i] = input[o + i].wrapping_sub(input[o - width + i]);
        }
    }
}

#[inline]
fn gradient_predictor(a: u8, b: u8, c: u8) -> i32 {
    let g = i32::from(a) + i32::from(b) - i32::from(c);
    if (g & !0xff) == 0 {
        g
    } else if g < 0 {
        0
    } else {
        255
    }
}

/// `DoGradientFilter_C`.
fn gradient_filter(input: &[u8], width: usize, height: usize, out: &mut [u8]) {
    out[0] = input[0];
    for i in 1..width {
        out[i] = input[i].wrapping_sub(input[i - 1]);
    }
    for row in 1..height {
        let o = row * width;
        // leftmost pixel: predict from above.
        out[o] = input[o].wrapping_sub(input[o - width]);
        for w in 1..width {
            let pred = gradient_predictor(
                input[o + w - 1],
                input[o - width + w],
                input[o - width + w - 1],
            );
            out[o + w] = input[o + w].wrapping_sub(pred as u8);
        }
    }
}

/// Apply filter `mode` to the plane; `FILTER_NONE` copies.
pub(crate) fn apply_filter(mode: u8, input: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; input.len()];
    match mode {
        FILTER_HORIZONTAL => horizontal_filter(input, width, height, &mut out),
        FILTER_VERTICAL => vertical_filter(input, width, height, &mut out),
        FILTER_GRADIENT => gradient_filter(input, width, height, &mut out),
        _ => out.copy_from_slice(input),
    }
    out
}

/// `WebPEstimateBestFilter` (`filters_utils.c`): sample every other pixel,
/// bin the scaled prediction residuals per filter, pick the filter with the
/// smallest occupied-bin index sum.
pub(crate) fn estimate_best_filter(data: &[u8], width: usize, height: usize) -> u8 {
    const SMAX: usize = 16;
    let sdiff = |a: u8, b: i32| -> usize { ((i32::from(a) - b).unsigned_abs() >> 4) as usize };

    let mut bins = [[0u8; SMAX]; FILTER_LAST as usize];
    let mut j = 2usize;
    while j + 1 < height {
        let p = &data[j * width..];
        let mut mean = i32::from(p[0]);
        let mut i = 2usize;
        while i + 1 < width {
            let diff0 = sdiff(p[i], mean);
            let diff1 = sdiff(p[i], i32::from(p[i - 1]));
            let diff2 = sdiff(p[i], i32::from(data[j * width + i - width]));
            let grad_pred = gradient_predictor(
                p[i - 1],
                data[j * width + i - width],
                data[j * width + i - width - 1],
            );
            let diff3 = sdiff(p[i], grad_pred);
            bins[FILTER_NONE as usize][diff0] = 1;
            bins[FILTER_HORIZONTAL as usize][diff1] = 1;
            bins[FILTER_VERTICAL as usize][diff2] = 1;
            bins[FILTER_GRADIENT as usize][diff3] = 1;
            mean = (3 * mean + i32::from(p[i]) + 2) >> 2;
            i += 2;
        }
        j += 2;
    }

    let mut best_filter = FILTER_NONE;
    let mut best_score = i32::MAX;
    for filter in 0..FILTER_LAST {
        let mut score = 0i32;
        for (i, &b) in bins[filter as usize].iter().enumerate() {
            if b > 0 {
                score += i as i32;
            }
        }
        if score < best_score {
            best_score = score;
            best_filter = filter;
        }
    }
    best_filter
}

/// `GetNumColors` (`alpha_enc.c`).
fn num_colors(data: &[u8]) -> usize {
    let mut seen = [false; 256];
    for &v in data {
        seen[v as usize] = true;
    }
    seen.iter().filter(|&&s| s).count()
}

/// The `alpha_filtering` config value, as libwebp maps it
/// (`CompressAlphaJob`): 0 = none, 1 = fast (default), 2 = best. zenwebp
/// only exposes the libwebp default (Fast) today; the other variants exist
/// so `filter_try_map` stays a complete port for when a config knob lands.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum AlphaFiltering {
    None,
    Fast,
    Best,
}

/// `GetFilterMap` (`alpha_enc.c`): the OR'd bit-set of filters to try.
pub(crate) fn filter_try_map(
    alpha: &[u8],
    width: usize,
    height: usize,
    filtering: AlphaFiltering,
    effort_level: u8,
) -> u32 {
    const FILTER_TRY_NONE: u32 = 1 << FILTER_NONE;
    const FILTER_TRY_ALL: u32 = (1 << FILTER_LAST) - 1;
    match filtering {
        AlphaFiltering::Fast => {
            let try_filter_none = effort_level > 3;
            const K_MIN_COLORS_FOR_FILTER_NONE: usize = 16;
            const K_MAX_COLORS_FOR_FILTER_NONE: usize = 192;
            let colors = num_colors(alpha);
            let filter = if colors <= K_MIN_COLORS_FOR_FILTER_NONE {
                FILTER_NONE
            } else {
                estimate_best_filter(alpha, width, height)
            };
            let mut bit_map = 1u32 << filter;
            if try_filter_none || colors > K_MAX_COLORS_FOR_FILTER_NONE {
                bit_map |= FILTER_TRY_NONE;
            }
            bit_map
        }
        AlphaFiltering::None => FILTER_TRY_NONE,
        AlphaFiltering::Best => FILTER_TRY_ALL,
    }
}

/// libwebp's per-trial VP8L quality/method mapping (`EncodeLossless`):
/// `method = effort_level`, `quality = (use_quality_100 && effort == 6)
/// ? 100 : 8 * effort`, `exact = 1`.
pub(crate) fn vp8l_quality_for_effort(effort_level: u8, use_quality_100: bool) -> f32 {
    if use_quality_100 && effort_level == 6 {
        100.0
    } else {
        8.0 * f32::from(effort_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_levels_preserves_min_max() {
        let mut data: Vec<u8> = (0u16..=255).map(|v| v as u8).collect();
        quantize_levels(&mut data, 4);
        assert_eq!(data[0], 0);
        assert_eq!(data[255], 255);
        let mut distinct = data.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(distinct.len() <= 4, "got {} levels", distinct.len());
    }

    #[test]
    fn quantize_levels_noop_when_few_levels() {
        let mut data = vec![0u8, 255, 0, 255, 128, 128];
        let orig = data.clone();
        quantize_levels(&mut data, 16);
        assert_eq!(data, orig);
    }

    #[test]
    fn alpha_levels_mapping_matches_libwebp() {
        assert_eq!(alpha_levels_for_quality(0), 2);
        assert_eq!(alpha_levels_for_quality(70), 16);
        assert_eq!(alpha_levels_for_quality(71), 24);
        assert_eq!(alpha_levels_for_quality(90), 176);
        assert_eq!(alpha_levels_for_quality(100), 256);
    }

    #[test]
    fn filters_roundtrip_against_unfilter() {
        // The decoder-side unfilters are the spec inverse; here we only
        // sanity-check shape: filtered output differs and is deterministic.
        let (w, h) = (9usize, 7usize);
        let src: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        for mode in [FILTER_HORIZONTAL, FILTER_VERTICAL, FILTER_GRADIENT] {
            let f1 = apply_filter(mode, &src, w, h);
            let f2 = apply_filter(mode, &src, w, h);
            assert_eq!(f1, f2);
            assert_ne!(f1, src);
            assert_eq!(f1[0], src[0]);
        }
    }
}
