//! Closed-loop target-zensim adaptive encoder.
//!
//! Mirrors the design of zenjpeg's `target_zq` module (`src/encode/zq.rs`):
//! the encoder iteratively encodes the image, decodes it, measures the
//! resulting zensim score against the source, and either ships the result
//! (within the tolerance band) or adjusts global VP8 quality and tries again.
//!
//! # Convergence
//!
//! Three mechanisms keep most encodes to one or two passes:
//!
//! 1. **Per-bucket starting-q calibration** — the first-pass quality is
//!    chosen by interpolating in a per-content-type anchor table. Photo,
//!    Drawing, and Icon get their own tables (3 buckets vs zenjpeg's 5);
//!    these mirror zenjpeg's PHOTO_DETAILED / SCREEN_CONTENT shapes.
//! 2. **Asymmetric tolerance band** — `max_overshoot=Some(t)` means we ship
//!    if achieved is in `[target, target+t]` even when more passes are
//!    available. `max_undershoot=None` (default) is best-effort; setting
//!    `Some(t)` makes a final achieved < target-t a hard error.
//! 3. **Secant step** — when off-band, the next q is computed via a
//!    one-pair secant fitted to the most recent two (q, achieved) probes.
//!    Falls back to a fixed step on the first iteration.
//!
//! # Calibration recipe
//!
//! The per-bucket anchors below come from `dev/zensim_calibrate.rs`. To
//! re-fit (e.g. after touching the encoder's RD path or upgrading zensim):
//!
//! ```text
//! cargo run --release --features target-zensim --example zensim_calibrate -- <corpus>
//! ```
//!
//! The tool emits a TSV plus a Rust-formatted const block. Replace the
//! `PHOTO`, `DRAWING`, `ICON` arrays below with the harness output. Anchors
//! are `(target_zensim, starting_q)` pairs ordered by ascending target.

#![allow(dead_code)] // Phase 1: types are wired but not yet exposed via LossyConfig

use super::analysis::ImageContentType;

/// Explicit target-perceptual-quality specification.
///
/// Default: target=80, max_overshoot=Some(1.5), max_undershoot=None,
/// max_passes=2 — best-effort behavior tuned to land in band on pass 1
/// for typical photo content and never iterate more than once.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZensimTarget {
    /// Ideal zensim score. The encoder iterates to reach or exceed this
    /// within the [`Self::max_passes`] budget.
    pub target: f32,

    /// Distance ABOVE target the encoder will accept without further
    /// iteration. `None` = ship the first feasible result (single pass
    /// once target is met). `Some(t)` = if `achieved > target + t`,
    /// claw back bytes by loosening quality.
    ///
    /// Default: `Some(1.5)`.
    pub max_overshoot: Option<f32>,

    /// Distance BELOW target the encoder will accept as a SUCCESSFUL
    /// encode. `None` = best-effort, never error (default; encoder ships
    /// whatever it managed within `max_passes`). `Some(t)` = if final
    /// `achieved < target - t` after exhausting `max_passes`, the encoder
    /// returns `Err`.
    ///
    /// Set this when you NEED a strictness guarantee (archival, SLA-bound
    /// serving). Permissive callers leave it `None` and inspect
    /// [`ZensimEncodeMetrics::achieved_score`] themselves.
    ///
    /// Default: `None`.
    pub max_undershoot: Option<f32>,

    /// Iteration budget. `1` = single-pass (no correction; behaves like a
    /// regular encode at the calibrated starting q). `2` = default; one
    /// initial encode plus one correction pass.
    pub max_passes: u8,
}

impl Default for ZensimTarget {
    fn default() -> Self {
        Self {
            target: 80.0,
            max_overshoot: Some(1.5),
            max_undershoot: None,
            max_passes: 2,
        }
    }
}

impl ZensimTarget {
    /// Construct a `ZensimTarget` with the given target and default
    /// tolerances / passes. Equivalent to `ZensimTarget { target,
    /// ..Default::default() }`.
    #[must_use]
    pub fn new(target: f32) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    /// Builder-style override of [`Self::max_overshoot`].
    #[must_use]
    pub fn with_max_overshoot(mut self, v: Option<f32>) -> Self {
        self.max_overshoot = v;
        self
    }

    /// Builder-style override of [`Self::max_undershoot`].
    #[must_use]
    pub fn with_max_undershoot(mut self, v: Option<f32>) -> Self {
        self.max_undershoot = v;
        self
    }

    /// Builder-style override of [`Self::max_passes`].
    #[must_use]
    pub fn with_max_passes(mut self, n: u8) -> Self {
        self.max_passes = n;
        self
    }
}

/// Outcome of a target-zensim encode, returned alongside the WebP bytes
/// from `LossyConfig::encode_rgb_with_metrics` and friends.
///
/// `targets_met` is `false` when:
/// - the iteration ran (target_zensim was set with the feature enabled), AND
/// - `achieved_score < target.target`, AND
/// - `max_undershoot.is_some()` AND `target - achieved > max_undershoot.unwrap()`
///
/// In all other cases — including configs without `target_zensim`, configs
/// where the feature is compiled out, and best-effort runs that landed
/// below target — `targets_met` is `true`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZensimEncodeMetrics {
    /// Final achieved zensim score. `f32::NAN` for non-target-zensim
    /// configs or when the feature is disabled (no measurement done).
    pub achieved_score: f32,

    /// Number of encode passes performed (including the initial pass).
    /// `1` for non-target-zensim configs.
    pub passes_used: u8,

    /// Encoded WebP byte count.
    pub bytes: usize,

    /// Whether the strictness contract was honored. See struct docs.
    pub targets_met: bool,
}

impl ZensimEncodeMetrics {
    /// Construct metrics for a non-target-zensim encode. `bytes` is the
    /// final output size.
    pub(crate) fn no_target(bytes: usize) -> Self {
        Self {
            achieved_score: f32::NAN,
            passes_used: 1,
            bytes,
            targets_met: true,
        }
    }
}

/// Per-bucket starting-q calibration. Maps a target zensim score to the
/// VP8 quality value that landed at-or-just-above the target on the
/// calibration corpus for `bucket`.
///
/// Anchors are `(target_zensim, starting_q)`. Linear interpolation between
/// adjacent anchors; clamped to the lowest/highest quality at the
/// endpoints.
///
/// # Anchor table source
///
/// PHOTO and DRAWING are fitted by `dev/zensim_calibrate.rs` against
/// 20 images of CID22 validation (median smallest-q-meeting-target per
/// content bucket). ICON is hand-distilled and conservative — re-fit
/// when an icon corpus is added (the CID22 subset has no ≤128px
/// images). To re-fit, see this module's top-level comment.
///
/// Fit data (CID22 validation, 20 images, 2026-04-26):
///   Photo   n=12, p25/p75 spread ≤ 7q across all 7 anchors.
///   Drawing n=8,  p25/p75 spread ≤ 5q across all 7 anchors.
#[must_use]
pub(crate) fn zensim_to_starting_q_for_bucket(target: f32, bucket: ImageContentType) -> f32 {
    // Photo bucket: natural photographs from CID22.
    const PHOTO: &[(f32, f32)] = &[
        (60.0, 30.0),
        (70.0, 60.0),
        (75.0, 75.0),
        (80.0, 85.0),
        (85.0, 90.0),
        (90.0, 98.0),
        (95.0, 100.0),
    ];
    // Drawing bucket: low-uniformity / complex texture content from
    // CID22 (the classifier puts mixed photo+UI here).
    const DRAWING: &[(f32, f32)] = &[
        (60.0, 30.0),
        (70.0, 60.0),
        (75.0, 72.5),
        (80.0, 85.0),
        (85.0, 90.0),
        (90.0, 100.0),
        (95.0, 100.0),
    ];
    // Icon bucket: ≤128px. Tiny images — every coefficient counts. Push
    // q higher to preserve detail. Hand-distilled (no fit corpus yet).
    const ICON: &[(f32, f32)] = &[
        (60.0, 65.0),
        (70.0, 78.0),
        (75.0, 85.0),
        (80.0, 90.0),
        (85.0, 95.0),
        (90.0, 98.0),
        (95.0, 100.0),
    ];
    let anchors = match bucket {
        ImageContentType::Photo => PHOTO,
        ImageContentType::Drawing | ImageContentType::Text => DRAWING,
        ImageContentType::Icon => ICON,
    };
    interpolate_anchors(target, anchors)
}

/// Linear interpolation over a sorted (target, q) anchor table. Clamps
/// to the endpoints' q values outside the bracketed range. Returns
/// `target` itself if the table is empty.
fn interpolate_anchors(target: f32, anchors: &[(f32, f32)]) -> f32 {
    if anchors.is_empty() {
        return target;
    }
    if target <= anchors[0].0 {
        return anchors[0].1;
    }
    let last = anchors[anchors.len() - 1];
    if target >= last.0 {
        return last.1.min(100.0);
    }
    for w in anchors.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if target >= lo.0 && target <= hi.0 {
            let t = (target - lo.0) / (hi.0 - lo.0);
            return lo.1 + t * (hi.1 - lo.1);
        }
    }
    target
}

// ============================================================================
// Iteration loop (gated on the `target-zensim` feature)
// ============================================================================

#[cfg(feature = "target-zensim")]
pub(crate) mod iteration {
    use super::*;
    use crate::PixelLayout;
    use crate::encoder::api::EncodeError;
    use crate::encoder::config::LossyConfig;
    use alloc::format;
    use alloc::vec::Vec;

    /// Result of running the closed-loop iteration: bytes + metrics, or
    /// an error if a hard constraint was violated.
    pub(crate) type IterationResult = Result<(Vec<u8>, ZensimEncodeMetrics), EncodeError>;

    /// Maximum |Δq| applied per secant step. Prevents oscillation when
    /// the metric is locally non-linear.
    const MAX_DELTA_Q: f32 = 10.0;
    /// Default sensitivity (Δq per zensim-unit gap) for the FIRST step,
    /// before we have a (q, score) pair to fit a secant against.
    const DEFAULT_SENSITIVITY: f32 = 1.5;
    /// Minimum |Δq| applied per step (avoids no-op when sensitivity ×
    /// gap rounds to ~0).
    const MIN_DELTA_Q: f32 = 0.5;

    /// Run the closed loop. Pass 0 is global-q at the calibrated start;
    /// pass 1+ uses per-segment diffmap-driven correction when segments
    /// are active (Phase 3), falling back to a global-q secant step when
    /// segments are disabled (`num_segments == 1`) or unavailable.
    pub(crate) fn run(
        cfg: &LossyConfig,
        target: ZensimTarget,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> IterationResult {
        // 1. Detect bucket from a quick classifier pass on the source.
        let bucket = detect_bucket(rgb, width, height);

        // 2. Resolve the starting q.
        let mut q = match bucket {
            Some(b) => zensim_to_starting_q_for_bucket(target.target, b),
            None => zensim_to_starting_q_for_bucket(target.target, ImageContentType::Photo),
        };
        q = q.clamp(0.0, 100.0);

        // 3. Encode pass 0 (global-q, no per-segment overrides).
        let (bytes0, stats0) = encode_at(cfg, q, false, None, rgb, width, height)?;
        let max_passes = target.max_passes.max(1);

        if max_passes <= 1 {
            return Ok((
                bytes0.clone(),
                ZensimEncodeMetrics {
                    achieved_score: f32::NAN,
                    passes_used: 1,
                    bytes: bytes0.len(),
                    targets_met: true,
                },
            ));
        }

        // 4. Measure pass 0 with per-pixel diffmap (used by Phase 3 if
        // segments are active).
        let z = zensim::Zensim::new(zensim::ZensimProfile::latest());
        let pre = build_source_reference(&z, rgb, width, height).ok_or_else(|| {
            EncodeError::InvalidBufferSize(
                "zensim precompute_reference failed (image too small?)".into(),
            )
        })?;
        let (score0, dm0) = measure_score_and_diffmap(&z, &pre, &bytes0, width, height)?;

        let mut best = Candidate {
            bytes: bytes0,
            score: score0,
            q,
            seg_overrides: None,
        };

        // Already in band? Ship pass 0.
        if in_band(score0, &target) {
            return finalize(best, 1, &target);
        }

        // Per-segment correction setup. Available iff segments are
        // active AND the encoder produced a segment_map.
        let seg_map = stats0.segment_map.clone();
        let mb_w = stats0.mb_width as usize;
        let mb_h = stats0.mb_height as usize;
        // Number of distinct segment IDs in the map. After
        // simplify_segments() this may be fewer than 4.
        let num_segments = seg_map
            .iter()
            .copied()
            .max()
            .map(|m| (m as usize) + 1)
            .unwrap_or(1);
        let per_segment_enabled =
            num_segments > 1 && seg_map.len() == mb_w * mb_h && !seg_map.is_empty();

        // prev_probe holds the SECOND-most-recent (q, score) so the
        // secant fits a slope between it and the current (q, last_score).
        let mut prev_probe: Option<(f32, f32)> = None;
        let mut last_q = q;
        let mut last_score = score0;
        let mut last_dm = dm0;
        // Cumulative per-segment overrides (applied across passes).
        let mut cum_overrides: [i8; 4] = [0; 4];

        // 5. Iterate.
        for pass in 1..max_passes {
            // Decide whether this pass uses per-segment correction or
            // a global-q step. Per-segment is preferred when active.
            let use_per_segment = per_segment_enabled;

            let (next_q, next_overrides) = if use_per_segment {
                // Phase 3: aggregate diffmap → per-segment mean → tighten
                // the worst segment / loosen the best when in claw-back.
                let new_overrides = next_segment_overrides(
                    cum_overrides,
                    &seg_map,
                    mb_w,
                    mb_h,
                    &last_dm,
                    width,
                    height,
                    last_score,
                    &target,
                    num_segments,
                );
                // Keep q the same when doing per-segment correction.
                (last_q, Some(new_overrides))
            } else {
                let nq = compute_next_q(last_q, last_score, prev_probe, &target);
                (nq.clamp(0.0, 100.0), None)
            };

            // If neither q changed nor overrides moved, bail.
            let q_moved = (next_q - last_q).abs() >= 0.05;
            let overrides_moved = match next_overrides {
                Some(o) => o != cum_overrides,
                None => false,
            };
            if !q_moved && !overrides_moved {
                break;
            }

            // multi_pass_stats=true on probe encodes — small size win
            // amortizes across passes.
            let (bytes_n, stats_n) =
                encode_at(cfg, next_q, true, next_overrides, rgb, width, height)?;
            let (score_n, dm_n) = measure_score_and_diffmap(&z, &pre, &bytes_n, width, height)?;
            let passes_used = pass + 1;

            best = pick_best(
                best,
                Candidate {
                    bytes: bytes_n,
                    score: score_n,
                    q: next_q,
                    seg_overrides: next_overrides,
                },
                &target,
            );

            if in_band(score_n, &target) {
                return finalize(best, passes_used, &target);
            }

            // Update state for next iteration.
            if let Some(ov) = next_overrides {
                cum_overrides = ov;
            }
            prev_probe = Some((last_q, last_score));
            last_q = next_q;
            last_score = score_n;
            last_dm = dm_n;
            let _ = stats_n; // segment_map is stable across passes by design.
        }

        finalize(best, max_passes, &target)
    }

    struct Candidate {
        bytes: Vec<u8>,
        score: f32,
        q: f32,
        seg_overrides: Option<[i8; 4]>,
    }

    fn pick_best(prev: Candidate, cand: Candidate, target: &ZensimTarget) -> Candidate {
        let prev_feas = prev.score >= target.target;
        let cand_feas = cand.score >= target.target;
        match (prev_feas, cand_feas) {
            (false, true) => cand,
            (true, false) => prev,
            (true, true) => {
                // Both meet target → pick the one with fewer bytes (best
                // bytes-recovery outcome).
                if cand.bytes.len() < prev.bytes.len() {
                    cand
                } else {
                    prev
                }
            }
            (false, false) => {
                // Neither meets target → pick the higher score.
                if cand.score > prev.score { cand } else { prev }
            }
        }
    }

    /// Returns true if `score` is in the comfort band — i.e. >= target
    /// AND (no overshoot configured OR overshoot within budget).
    fn in_band(score: f32, target: &ZensimTarget) -> bool {
        if score < target.target {
            return false;
        }
        match target.max_overshoot {
            Some(t) => (score - target.target) <= t,
            None => true, // No claw-back wanted; first feasible ships.
        }
    }

    /// Compute next q via secant when we have a previous probe, fixed-
    /// step otherwise.
    fn compute_next_q(
        q: f32,
        last_score: f32,
        prev: Option<(f32, f32)>,
        target: &ZensimTarget,
    ) -> f32 {
        let gap = target.target - last_score;
        // Secant: estimate dscore/dq from the two most recent probes.
        // (Note: `q` and `last_score` are the SAME probe as the latest
        // one in `prev_probe` until we update. So the secant fits
        // against the second-most-recent point.)
        let delta = if let Some((q_prev, s_prev)) = prev {
            let dq = q - q_prev;
            let ds = last_score - s_prev;
            if dq.abs() > 0.1 && ds.abs() > 0.05 {
                let slope = ds / dq; // zensim units per q point
                let mut step = gap / slope.max(0.05);
                step = step.clamp(-MAX_DELTA_Q, MAX_DELTA_Q);
                if step.abs() < MIN_DELTA_Q {
                    step = MIN_DELTA_Q.copysign(step);
                }
                step
            } else {
                fallback_step(gap)
            }
        } else {
            fallback_step(gap)
        };
        q + delta
    }

    fn fallback_step(gap: f32) -> f32 {
        let mut step = gap * DEFAULT_SENSITIVITY;
        step = step.clamp(-MAX_DELTA_Q, MAX_DELTA_Q);
        if step.abs() < MIN_DELTA_Q {
            step = MIN_DELTA_Q.copysign(if gap == 0.0 { 1.0 } else { gap });
        }
        step
    }

    fn finalize(best: Candidate, passes_used: u8, target: &ZensimTarget) -> IterationResult {
        // Strict-mode failure check: if max_undershoot is set and we
        // missed by more than that, return an error.
        if let Some(slack) = target.max_undershoot
            && best.score < target.target - slack
        {
            return Err(EncodeError::InvalidBufferSize(format!(
                "target_zensim: achieved {:.3} below floor {:.3} (max_undershoot {:.3}) after {} passes",
                best.score, target.target, slack, passes_used,
            )));
        }
        let targets_met = best.score >= target.target
            || target
                .max_undershoot
                .is_none_or(|t| (target.target - best.score) <= t);
        let bytes_len = best.bytes.len();
        Ok((
            best.bytes,
            ZensimEncodeMetrics {
                achieved_score: best.score,
                passes_used,
                bytes: bytes_len,
                targets_met,
            },
        ))
    }

    /// Encode RGB pixels at the given quality with optional per-segment
    /// quant-index overrides. `enable_multi_pass` toggles
    /// `multi_pass_stats` for inside-the-loop probes (small size win).
    /// Returns bytes plus the encoder's `EncodeStats` (carrying the
    /// segment_map needed for Phase 3 aggregation).
    fn encode_at(
        cfg: &LossyConfig,
        q: f32,
        enable_multi_pass: bool,
        seg_overrides: Option<[i8; 4]>,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<u8>, crate::encoder::api::EncodeStats), EncodeError> {
        let mut probe_cfg = cfg.clone();
        probe_cfg.quality = q.clamp(0.0, 100.0);
        // Force multi_pass_stats on/off as requested (always enabled for
        // pass 1+ to amortize the marginal cost across the search).
        probe_cfg.multi_pass_stats = enable_multi_pass;
        // Iteration must NOT recurse — disable target_zensim/target_psnr/
        // target_size on the probe config. Only target_zensim is plumbed
        // here; existing target_size/target_psnr semantics stay independent.
        probe_cfg.target_size = 0;
        probe_cfg.target_psnr = 0.0;
        probe_cfg.target_zensim = None;
        probe_cfg.segment_quant_overrides = seg_overrides;

        let req = crate::encoder::api::EncodeRequest::lossy(
            &probe_cfg,
            rgb,
            PixelLayout::Rgb8,
            width,
            height,
        );
        match req.encode_with_stats() {
            Ok((bytes, stats)) => Ok((bytes, stats)),
            Err(at_err) => Err(at_err.decompose().0),
        }
    }

    /// Build a precomputed zensim reference from RGB bytes.
    fn build_source_reference(
        z: &zensim::Zensim,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Option<zensim::PrecomputedReference> {
        let w = width as usize;
        let h = height as usize;
        if rgb.len() < w * h * 3 {
            return None;
        }
        let pixels: &[[u8; 3]] = bytemuck::cast_slice(&rgb[..w * h * 3]);
        let slice = zensim::RgbSlice::new(pixels, w, h);
        z.precompute_reference(&slice).ok()
    }

    /// Decode `webp` and compute zensim score + per-pixel diffmap. The
    /// diffmap is `Vec<f32>` of length `width * height` in row-major
    /// order — used by Phase 3 per-segment aggregation.
    fn measure_score_and_diffmap(
        z: &zensim::Zensim,
        pre: &zensim::PrecomputedReference,
        webp: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(f32, Vec<f32>), EncodeError> {
        let (rgb, w, h) = crate::oneshot::decode_rgb(webp).map_err(|e| {
            EncodeError::InvalidBufferSize(format!(
                "target_zensim: decode for measurement failed: {:?}",
                e.decompose().0,
            ))
        })?;
        if w != width || h != height {
            return Err(EncodeError::InvalidBufferSize(format!(
                "target_zensim: decoded dims {}x{} != source {}x{}",
                w, h, width, height,
            )));
        }
        let n = (w as usize) * (h as usize) * 3;
        if rgb.len() < n {
            return Err(EncodeError::InvalidBufferSize(
                "target_zensim: short decoded buffer".into(),
            ));
        }
        let chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb[..n]);
        let slice = zensim::RgbSlice::new(chunks, w as usize, h as usize);
        let dm = z
            .compute_with_ref_and_diffmap(pre, &slice, zensim::DiffmapWeighting::Trained)
            .map_err(|e| {
                EncodeError::InvalidBufferSize(format!(
                    "zensim compute_with_ref_and_diffmap failed: {:?}",
                    e
                ))
            })?;
        let score = dm.score() as f32;
        Ok((score, dm.diffmap().to_vec()))
    }

    /// Aggregate the per-pixel diffmap into per-segment means, then
    /// compute the next per-segment quant_index override. The policy:
    ///
    /// - If overall score < target: tighten the segment with the highest
    ///   mean diffmap (reduce its quant by 1–3 depending on gap).
    /// - If achieved > target + max_overshoot: loosen the segment with
    ///   the lowest mean diffmap.
    ///
    /// Returns the NEW cumulative overrides (not deltas). Bounded so
    /// `seg_quant + override` stays in `[-32, 32]` cumulatively (the
    /// `compute_segment_quant` clamp handles the absolute [0, 127]
    /// range; the soft cap here prevents oscillation).
    #[allow(clippy::too_many_arguments)]
    fn next_segment_overrides(
        cum: [i8; 4],
        seg_map: &[u8],
        mb_w: usize,
        mb_h: usize,
        diffmap: &[f32],
        width: u32,
        height: u32,
        score: f32,
        target: &ZensimTarget,
        num_segments: usize,
    ) -> [i8; 4] {
        let mut sum = [0.0f64; 4];
        let mut count = [0u32; 4];
        let w = width as usize;
        let h = height as usize;
        // For each MB (16x16 region), accumulate the mean diffmap for
        // that MB into its segment's running totals.
        for my in 0..mb_h {
            for mx in 0..mb_w {
                let seg = seg_map.get(my * mb_w + mx).copied().unwrap_or(0) as usize;
                if seg >= 4 {
                    continue;
                }
                let y0 = my * 16;
                let x0 = mx * 16;
                let y1 = (y0 + 16).min(h);
                let x1 = (x0 + 16).min(w);
                if y1 == y0 || x1 == x0 {
                    continue;
                }
                let mut s = 0.0f64;
                let mut n = 0u32;
                for y in y0..y1 {
                    let row = &diffmap[y * w..y * w + w];
                    for &v in &row[x0..x1] {
                        s += v as f64;
                        n += 1;
                    }
                }
                if n > 0 {
                    sum[seg] += s / n as f64;
                    count[seg] += 1;
                }
            }
        }
        let mut means = [0.0f32; 4];
        for s in 0..num_segments.min(4) {
            if count[s] > 0 {
                means[s] = (sum[s] / count[s] as f64) as f32;
            }
        }
        // Find worst (highest mean) and best (lowest mean) active
        // segments.
        let mut worst = 0usize;
        let mut best = 0usize;
        for s in 0..num_segments.min(4) {
            if count[s] == 0 {
                continue;
            }
            if means[s] > means[worst] || count[worst] == 0 {
                worst = s;
            }
            if means[s] < means[best] || count[best] == 0 {
                best = s;
            }
        }

        let mut out = cum;
        let gap = target.target - score;
        if gap > 0.0 {
            // Score below target — tighten worst segment.
            // Step magnitude scaled to gap: small gap → -1, larger → up to -3.
            let step = if gap > 4.0 {
                -3
            } else if gap > 2.0 {
                -2
            } else {
                -1
            };
            // Soft cap accumulated tightening at -16 to leave room for
            // future passes and stay within sensible quant range.
            out[worst] = (i32::from(out[worst]) + step).clamp(-16, 16) as i8;
        } else if let Some(t) = target.max_overshoot
            && (score - target.target) > t
        {
            // Above the comfort band — loosen the best segment.
            let overshoot = score - target.target - t;
            let step = if overshoot > 4.0 {
                3
            } else if overshoot > 2.0 {
                2
            } else {
                1
            };
            out[best] = (i32::from(out[best]) + step).clamp(-16, 16) as i8;
        }
        out
    }

    /// Run the bucket classifier on the RGB source. We need a Y plane —
    /// derive a quick luma approximation from RGB rather than running the
    /// full encoder analyzer (which would re-encode an extra time).
    fn detect_bucket(rgb: &[u8], width: u32, height: u32) -> Option<ImageContentType> {
        let w = width as usize;
        let h = height as usize;
        if w < 8 || h < 8 || rgb.len() < w * h * 3 {
            return None;
        }
        // BT.601 Y = 0.299R + 0.587G + 0.114B.
        let mut y_plane: Vec<u8> = Vec::with_capacity(w * h);
        let mut alpha_hist = [0u32; 256];
        // Use the high-byte of (Y-difference between neighbors) as a stand-in
        // for the encoder's alpha histogram. We don't have a real alpha plane
        // here so populate it with Y values themselves — the classifier
        // primarily uses bimodality + edge density + uniformity, all of
        // which are tolerant of histogram shape.
        for px in rgb.chunks_exact(3) {
            let y = ((u32::from(px[0]) * 76 + u32::from(px[1]) * 150 + u32::from(px[2]) * 30) >> 8)
                as u8;
            y_plane.push(y);
            alpha_hist[y as usize] += 1;
        }
        let bucket = crate::encoder::analysis::classify_image_type(&y_plane, w, h, w, &alpha_hist);
        Some(bucket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_default() {
        let t = ZensimTarget::default();
        assert_eq!(t.target, 80.0);
        assert_eq!(t.max_overshoot, Some(1.5));
        assert_eq!(t.max_undershoot, None);
        assert_eq!(t.max_passes, 2);
    }

    #[test]
    fn target_builder() {
        let t = ZensimTarget::new(85.0)
            .with_max_overshoot(Some(0.5))
            .with_max_undershoot(Some(2.0))
            .with_max_passes(3);
        assert_eq!(t.target, 85.0);
        assert_eq!(t.max_overshoot, Some(0.5));
        assert_eq!(t.max_undershoot, Some(2.0));
        assert_eq!(t.max_passes, 3);
    }

    #[test]
    fn metrics_no_target() {
        let m = ZensimEncodeMetrics::no_target(1234);
        assert!(m.achieved_score.is_nan());
        assert_eq!(m.passes_used, 1);
        assert_eq!(m.bytes, 1234);
        assert!(m.targets_met);
    }

    #[test]
    fn calibration_monotonic_per_bucket() {
        for &b in &[
            ImageContentType::Photo,
            ImageContentType::Drawing,
            ImageContentType::Icon,
        ] {
            let mut prev = 0.0f32;
            for t in (60..=95).step_by(5) {
                let q = zensim_to_starting_q_for_bucket(t as f32, b);
                assert!(
                    q >= prev,
                    "non-monotonic at {b:?} target {t}: {prev} -> {q}"
                );
                assert!((1.0..=100.0).contains(&q), "{b:?} target {t} q={q}");
                prev = q;
            }
        }
    }

    #[test]
    fn calibration_clamps_at_endpoints() {
        // Below the lowest anchor: clamp to lowest q.
        let q_low = zensim_to_starting_q_for_bucket(40.0, ImageContentType::Photo);
        assert_eq!(q_low, 30.0);
        // Above the highest anchor: clamp to highest q.
        let q_high = zensim_to_starting_q_for_bucket(99.0, ImageContentType::Photo);
        assert_eq!(q_high, 100.0);
    }
}
