//! VP8 lossy encoder implementation.
//!
//! This module provides DCT-based lossy compression for WebP images,
//! compatible with the VP8 intra-frame format.
//!
//! ## Module organization
//!
//! - `header`: VP8 bitstream header encoding
//! - `mode_selection`: Intra mode selection (I4, I16, UV) with RD optimization
//! - `prediction`: Block prediction generation
//! - residuals: Token buffer and coefficient encoding
//!
//! ## Encoding pipeline
//!
//! 1. **Color conversion**: RGB → YUV420 (with optional sharp YUV)
//! 2. **Analysis pass**: Compute per-MB complexity (alpha) for segmentation
//! 3. **Segmentation**: K-means clustering assigns MBs to 1-4 quantization segments
//! 4. **Mode selection**: For each MB, choose best prediction mode via RD cost
//! 5. **Transform & quantize**: DCT + quantization + optional trellis optimization
//! 6. **Entropy coding**: Arithmetic coding of residual coefficients
//! 7. **Loop filter**: Deblocking filter parameters computed for decoder
//!
//! ## Quality settings
//!
//! The encoder supports quality 0-100 (like JPEG), with method levels 0-6
//! controlling the speed/quality trade-off:
//! - Methods 0-2: Fast, basic mode selection
//! - Methods 3-4: Better mode decisions, perceptual optimizations
//! - Methods 5-6: Exhaustive search, trellis quantization

use alloc::vec;
use alloc::vec::Vec;
use archmage::prelude::*;
use core::mem;

#[allow(unused_imports)]
use whereat::at;

use super::analysis::analyze_image_with_hint_gate;
use super::api::EncodeError;
use super::api::PixelLayout;
use super::arithmetic::ArithmeticEncoder;
use super::cost::{
    LevelCosts, ProbaStats, assign_segments_kmeans, classify_image_type, compute_segment_quant,
    content_type_to_tuning,
};
use super::vec_writer::VecWriter;
use crate::common::prediction::*;
use crate::common::types::Frame;
use crate::common::types::*;
// VP8_AC_TABLE2 is pulled in via the `common::types::*` glob above so the
// decoder can share the same definition without depending on the encoder.
// convert_image_sharp_yuv_with_config is called via full path below
use crate::decoder::yuv::convert_image_y;

mod header;
pub(crate) mod mode_selection;
mod prediction;
mod residuals;

//------------------------------------------------------------------------------
// Quality to quantization index mapping
//
// Use centralized functions from fast_math module

use super::fast_math::quality_to_quant_index;

//------------------------------------------------------------------------------
// Quality search state for target_size convergence
//
// Ported from libwebp src/enc/frame_enc.c PassStats

/// Convergence threshold for quality search (ported from libwebp DQ_LIMIT).
/// Quality search is considered converged when |dq| < DQ_LIMIT.
const DQ_LIMIT: f32 = 0.4;

/// Running totals threaded through a single encoding-pass row loop.
///
/// Kept on the stack of [`Vp8Encoder::encode_image`] so the per-row helper
/// ([`Vp8Encoder::encode_mb_row`]) and per-MB helper
/// ([`Vp8Encoder::encode_macroblock`]) can mutate the same accumulators
/// without threading 8 distinct `&mut` references.
///
/// The fields mirror the per-pass locals that used to live inline in
/// `encode_image` (`total_mb`, `skip_mb`, `block_count_i4`, `block_count_i16`,
/// `sse_y/u/v`, `refresh_countdown`); after each pass the orchestrator copies
/// them back out so the existing post-pass code path keeps the same shape.
struct MbRowState {
    total_mb: u32,
    skip_mb: u32,
    block_count_i4: u32,
    block_count_i16: u32,
    sse_y: u64,
    sse_u: u64,
    sse_v: u64,
    refresh_countdown: i32,
}

/// State for quality search convergence (target size or PSNR).
/// Uses secant method to interpolate toward target value.
/// Ported from libwebp's PassStats struct.
struct PassStats {
    is_first: bool,
    dq: f32,
    q: f32,
    last_q: f32,
    qmin: f32,
    qmax: f32,
    value: f64,      // current encoded size
    last_value: f64, // previous encoded size
    target: f64,     // target size
}

impl PassStats {
    /// Initialize pass stats for target size search.
    fn new_for_size(target_size: u32, quality: u8, qmin: u8, qmax: u8) -> Self {
        let qmin_f = f32::from(qmin);
        let qmax_f = f32::from(qmax);
        let q = f32::from(quality).clamp(qmin_f, qmax_f);
        Self {
            is_first: true,
            dq: 10.0,
            q,
            last_q: q,
            qmin: qmin_f,
            qmax: qmax_f,
            value: 0.0,
            last_value: 0.0,
            target: f64::from(target_size),
        }
    }

    /// Initialize pass stats for target PSNR search.
    /// PSNR increases with quality, so the secant direction is reversed
    /// compared to size search (higher quality = higher PSNR).
    fn new_for_psnr(target_psnr: f32, quality: u8, qmin: u8, qmax: u8) -> Self {
        let qmin_f = f32::from(qmin);
        let qmax_f = f32::from(qmax);
        let q = f32::from(quality).clamp(qmin_f, qmax_f);
        Self {
            is_first: true,
            dq: 10.0,
            q,
            last_q: q,
            qmin: qmin_f,
            qmax: qmax_f,
            value: 0.0,
            last_value: 0.0,
            target: f64::from(target_psnr),
        }
    }

    /// Compute next quality value using secant method.
    /// Returns the new quality to try.
    fn compute_next_q(&mut self) -> f32 {
        let dq = if self.is_first {
            // First iteration: move in direction of target
            self.is_first = false;
            if self.value > self.target {
                -self.dq
            } else {
                self.dq
            }
        } else if (self.value - self.last_value).abs() > f64::EPSILON {
            // Secant method: linear interpolation to find next q
            let slope = (self.target - self.value) / (self.last_value - self.value);
            (slope * f64::from(self.last_q - self.q)) as f32
        } else {
            0.0 // converged
        };

        // Limit dq to avoid large swings
        self.dq = dq.clamp(-30.0, 30.0);
        self.last_q = self.q;
        self.last_value = self.value;
        self.q = (self.q + self.dq).clamp(self.qmin, self.qmax);
        self.q
    }

    /// Check if convergence is reached.
    fn is_converged(&self) -> bool {
        self.dq.abs() <= DQ_LIMIT
    }
}

//------------------------------------------------------------------------------
// SSE (Sum of Squared Errors) distortion functions
//
// These measure the distortion between source and predicted blocks.
// Lower SSE = better prediction = less data to encode.

/// Compute SSE for a 16x16 luma block within bordered prediction buffer
/// Compares source YUV data against predicted block with border
#[inline]
pub(crate) fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    incant!(
        sse_16x16_luma_dispatch(src_y, src_width, mbx, mby, pred),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse_16x16_luma_dispatch_v3(
    _token: X64V3Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_sse::sse_16x16_luma_sse2(_token, src_y, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse_16x16_luma_dispatch_neon(
    token: NeonToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_neon::sse_16x16_luma_neon(token, src_y, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse_16x16_luma_dispatch_wasm128(
    token: Wasm128Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_wasm::sse_16x16_luma_wasm(token, src_y, src_width, mbx, mby, pred)
}

#[inline(always)]
fn sse_16x16_luma_dispatch_scalar(
    _token: ScalarToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 16 * src_width + mbx * 16;
    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;
        for x in 0..16 {
            let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// Compute SSE for an 8x8 chroma block within bordered prediction buffer
#[inline]
pub(crate) fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    incant!(
        sse_8x8_chroma_dispatch(src_uv, src_width, mbx, mby, pred),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse_8x8_chroma_dispatch_v3(
    _token: X64V3Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_sse::sse_8x8_chroma_sse2(_token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse_8x8_chroma_dispatch_neon(
    token: NeonToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_neon::sse_8x8_chroma_neon(token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse_8x8_chroma_dispatch_wasm128(
    token: Wasm128Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_wasm::sse_8x8_chroma_wasm(token, src_uv, src_width, mbx, mby, pred)
}

#[inline(always)]
fn sse_8x8_chroma_dispatch_scalar(
    _token: ScalarToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 8 * src_width + mbx * 8;
    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;
        for x in 0..8 {
            let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

// currently in decoder it actually stores this information on the macroblock but that's confusing
// because it doesn't update the macroblock, just the complexity values as we decode
// this is used as the complexity per 13.3 in the decoder
#[derive(Clone, Copy, Default)]
struct Complexity {
    y2: u8,
    y: [u8; 4],
    u: [u8; 2],
    v: [u8; 2],
}

#[derive(Default)]
struct QuantizationIndices {
    yac_abs: u8,
    ydc_delta: Option<i8>,
    y2dc_delta: Option<i8>,
    y2ac_delta: Option<i8>,
    uvdc_delta: Option<i8>,
    uvac_delta: Option<i8>,
}

/// TODO: Consider merging this with the MacroBlock from the decoder
#[derive(Clone, Copy, Default)]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,

    coeffs_skipped: bool,

    /// Winning-mode distortion `D` (source-vs-reconstruction SSE) for I16
    /// macroblocks. `None` for I4 mode (LumaMode::B). Threaded from
    /// `pick_best_intra16` / `pick_intra16_fast_dc` via
    /// `choose_macroblock_info`. Consumed by the `store_max_delta` call
    /// site to gate `max_edge_per_segment` updates on `D > segment.min_disto`,
    /// matching libwebp's `quant_enc.c:1111` per-MB filter-strength gate
    /// (issue #44).
    intra16_d: Option<u32>,
    /// The I16 CANDIDATE's Y2 (zigzag) and blocky flag, retained even when I4
    /// wins the macroblock.
    ///
    /// libwebp runs `StoreMaxDelta` at the end of `PickBestIntra16`
    /// (`quant_enc.c:1035`), i.e. BEFORE `PickBestIntra4` can override the mode:
    ///
    /// ```c
    /// if ((rd->nz & 0x100ffff) == 0x1000000 && rd->D > dqm->min_disto) {
    ///   StoreMaxDelta(dqm, rd->y_dc_levels);
    /// }
    /// ```
    ///
    /// So a macroblock whose I16 candidate is blocky-and-high-distortion feeds
    /// `max_edge` even if it is finally emitted as I4. zenwebp used to gate that
    /// on the FINAL mode (`!is_i4`) and drop `intra16_d` for I4 macroblocks,
    /// which under-counted `max_edge` and left the segment's loop-filter level
    /// un-bumped. Measured on CID22 382297 q5/m4/sns50/flt60/segs4: one missed
    /// macroblock put `max_edge[1]` at 1 instead of 2, so segment 1's filter
    /// level came out 29 where libwebp writes 55 — a header-field divergence
    /// with every mode decision otherwise identical. (#38)
    ///
    /// `intra16_cand_d` is the I16 CANDIDATE's raw distortion (unlike
    /// `intra16_d`, which is the WINNING mode's and is None for I4). It is what
    /// libwebp's `rd->D > dqm->min_disto` half of the gate compares.
    intra16_cand_d: u32,
    intra16_y2_zz: [i32; 16],
    intra16_blocky: bool,
}

pub(super) type ChromaCoeffs = [i32; 16 * 4];

/// Space-joined level list for the LEVFINAL debug dump (#38).
#[cfg(feature = "mode_debug")]
fn fmt_levels(levels: &[i32; 16]) -> alloc::string::String {
    use core::fmt::Write as _;
    let mut s = alloc::string::String::new();
    for v in levels {
        let _ = write!(s, " {v}");
    }
    s
}

/// Quantized zigzag coefficients for a macroblock, stored for multi-pass encoding.
/// libwebp stores quantized coefficients from pass 1 and reuses them in pass 2+.
/// These are the final quantized values (post-trellis if applicable), ready for
/// direct token recording without re-quantization.
#[derive(Clone)]
struct QuantizedMbCoeffs {
    /// Y2 DC transform coefficients (16 values), only used for I16 mode
    y2_zigzag: [i32; 16],
    /// Y1 block coefficients (16 blocks × 16 values), zigzag order
    y1_zigzag: [[i32; 16]; 16],
    /// U block coefficients (4 blocks × 16 values), zigzag order
    u_zigzag: [[i32; 16]; 4],
    /// V block coefficients (4 blocks × 16 values), zigzag order
    v_zigzag: [[i32; 16]; 4],
}

impl QuantizedMbCoeffs {
    /// Pre-allocated zero coefficients for skipped macroblocks.
    const ZERO: Self = Self {
        y2_zigzag: [0; 16],
        y1_zigzag: [[0; 16]; 16],
        u_zigzag: [[0; 16]; 4],
        v_zigzag: [[0; 16]; 4],
    };

    /// Check if all coefficients are zero (for skip detection).
    /// Uses bitwise OR accumulator — faster than iterating with early-exit
    /// because it avoids branch mispredictions on the common non-zero case.
    #[inline]
    fn is_all_zero(&self, is_i4: bool, first_coeff_y1: usize) -> bool {
        let mut acc: u32 = 0;
        // Check Y2 (only for I16 mode)
        if !is_i4 {
            for &c in &self.y2_zigzag {
                acc |= c as u32;
            }
        }
        // Check Y1 blocks
        for block in &self.y1_zigzag {
            for &c in &block[first_coeff_y1..] {
                acc |= c as u32;
            }
        }
        // Check U blocks
        for block in &self.u_zigzag {
            for &c in block {
                acc |= c as u32;
            }
        }
        // Check V blocks
        for block in &self.v_zigzag {
            for &c in block {
                acc |= c as u32;
            }
        }
        acc == 0
    }

    /// Returns true if this MB matches libwebp's "blocky I16" gating
    /// condition for `StoreMaxDelta` (`quant_enc.c:1111`):
    /// Y2 has any nonzero AND all 16 Y1 AC coefficients are zero.
    /// In libwebp's `nz` mask this is `(nz & 0x100ffff) == 0x1000000`.
    /// Caller must already know the MB is I16 (Y2 only exists then).
    #[inline]
    fn is_blocky_i16(&self) -> bool {
        // Y1 AC = positions 1..16 of every Y1 block (DC is position 0).
        let mut y1_ac_acc: u32 = 0;
        for block in &self.y1_zigzag {
            for &c in &block[1..16] {
                y1_ac_acc |= c as u32;
            }
        }
        if y1_ac_acc != 0 {
            return false;
        }
        let mut y2_acc: u32 = 0;
        for &c in &self.y2_zigzag {
            y2_acc |= c as u32;
        }
        y2_acc != 0
    }
}

struct Vp8Encoder<'a> {
    writer: &'a mut Vec<u8>,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,
    segments_update_map: bool,
    segment_tree_probs: [Prob; 3],
    /// Segment ID for each macroblock (mb_width * mb_height)
    segment_map: Vec<u8>,
    /// Per-MB FastMBAnalyze hints (only populated when `method <= 1`).
    /// When `Some`, `choose_macroblock_info` consumes hints directly to skip
    /// full RD mode evaluation, mirroring libwebp's `RefineUsingDistortion`
    /// flow at m0/m1. Empty for higher methods.
    fast_mb_hints: Vec<crate::encoder::analysis::MbModeHint>,
    /// Per-MB analysis UV mode (0 = DC, 1 = TM), populated alongside
    /// `fast_mb_hints`. Consumed at m0 under `StrictLibwebpParity` where
    /// libwebp leaves the chroma mode at the analysis pick (`refine_uv_mode`
    /// is 0). Empty otherwise.
    fast_mb_uv_hints: Vec<u8>,

    macroblock_no_skip_coeff: Option<u8>,
    quantization_indices: QuantizationIndices,

    token_probs: TokenProbTables,
    /// Token statistics for adaptive probability updates
    proba_stats: ProbaStats,
    /// libwebp `fast_probe` (StatLoop, `frame_enc.c`): at method 0 the
    /// stats-collection pass records residuals over only the first
    /// `nb_mbs>>2` (or 50 if the frame has ≤200 MBs) macroblocks, and the
    /// emitted coefficient probabilities are finalized from that subset. When
    /// `Some(limit)` — set under `StrictLibwebpParity` at method 0 — snapshot
    /// `proba_stats` after `limit` MBs and derive the emitted probabilities
    /// from the snapshot, matching libwebp. `None` records the whole frame
    /// (the tuned default, and every method ≥ 1).
    fast_probe_stat_limit: Option<usize>,
    /// Frozen copy of `proba_stats` taken at the `fast_probe_stat_limit`
    /// boundary; `compute_updated_probabilities` reads this instead of the
    /// full-frame `proba_stats` when present.
    fast_probe_snapshot: Option<ProbaStats>,
    /// `skip_mb` frozen at the same `fast_probe_stat_limit` boundary. libwebp
    /// counts `nb_skip` in `OneStatPass` over only the stats subset (m0
    /// `fast_probe`), while `FinalizeSkipProba` divides by the FULL frame —
    /// so at m0 the skip probability must come from the subset count.
    /// `None` when the subset never closed (m1/m2, frames smaller than the
    /// limit, tuned default): use the full-frame `skip_mb`. (#38)
    fast_probe_skip_count: Option<u32>,
    /// Updated probabilities computed from statistics
    updated_probs: Option<TokenProbTables>,
    /// Precomputed level costs for coefficient cost estimation
    level_costs: LevelCosts,
    /// Whether to use trellis quantization for better RD optimization
    do_trellis: bool,
    /// Whether to use trellis during mode selection (RD_OPT_TRELLIS_ALL, method >= 6)
    do_trellis_i4_mode: bool,
    /// Whether to use chroma error diffusion to reduce banding
    do_error_diffusion: bool,
    /// Encoding method (0-6): 0=fastest, 6=best quality
    method: u8,
    /// Spatial noise shaping strength (0-100)
    sns_strength: u8,
    /// Preprocessing options. Matches libwebp's `WebPConfig::preprocessing`.
    /// Default `Preprocessing::none()` (off), matching libwebp's default.
    smooth_segment_map: bool,
    /// Cost model selection (mode selection + trellis). Default
    /// `ZenwebpDefault` enables perceptual extensions per method level;
    /// `StrictLibwebpParity` disables them.
    cost_model: super::api::CostModel,
    /// Run a stat-collection pre-pass (m4 only — m5/m6 already saturate).
    /// Default `false`. Set via `LossyConfig::with_multi_pass_stats(true)`.
    multi_pass_stats: bool,
    /// libwebp's `enc->do_search` — `(target_size > 0 || target_PSNR > 0)`
    /// (`webp_enc.c:118`). It disables `StatLoop`'s `fast_probe`, so the m0/m3
    /// stats pass covers the whole frame instead of a subset. zen drives target
    /// search from an outer loop (`encode_with_quality_search`), which re-runs
    /// this encoder per q, so the inner encode has no notion of the search on
    /// its own — this field carries it in. Only read under
    /// `StrictLibwebpParity`. (#38)
    do_search: bool,
    /// Per-segment additive quant_index offsets applied AFTER
    /// `compute_segment_quant`'s SNS modulation. `None` (default) leaves
    /// segments untouched. Crate-internal hook for `target_zensim`'s
    /// per-segment correction pass.
    segment_quant_overrides: Option<[i8; 4]>,
    /// Loop filter strength (0-100)
    filter_strength: u8,
    /// Loop filter sharpness (0-7)
    filter_sharpness: u8,
    /// Number of segments (1-4)
    num_segments: u8,
    /// Selected preset (used for Auto detection)
    preset: super::api::Preset,
    /// Partition limit (0-100): extra I4 penalty to prevent partition 0 overflow
    partition_limit: u8,

    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    /// Coefficient-data partitions. The VP8 bitstream supports 1, 2, 4, or 8
    /// partitions (interleaved by MB row mod num_parts), but zenwebp currently
    /// always emits exactly one. Implementing multi-partition output would
    /// require routing token emission through `partitions[r % num_parts]` and
    /// growing this Vec — there is no public knob to request that today.
    /// Kept as a Vec to avoid churning the header-emission code (which already
    /// computes `partitions.len().ilog2()`) when multi-partition is added. (#35-#3)
    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,

    // Error diffusion state for chroma DC coefficients
    // This implements Floyd-Steinberg-like error spreading to reduce banding
    // top_derr[mbx][channel][0..2] = errors from block above
    // left_derr[channel][0..2] = errors from block to the left
    top_derr: Vec<[[i8; 2]; 2]>,
    left_derr: [[i8; 2]; 2],

    /// Token buffer for deferred coefficient encoding (method >= 2).
    /// Stores bit-level tokens during the recording pass for later emission
    /// with optimized probability tables.
    token_buffer: Option<residuals::TokenBuffer>,
    /// Stored macroblock info from token recording pass, used to write
    /// headers without redoing mode selection.
    stored_mb_info: Vec<MacroblockInfo>,
    /// Stored quantized coefficients for token buffer approach.
    /// Mode decisions + quantized zigzag coefficients stored during encoding.
    stored_mb_coeffs: Vec<QuantizedMbCoeffs>,
    /// Maximum observed edge magnitude per segment (port of libwebp's
    /// `VP8SegmentInfo::max_edge`, `vp8i_enc.h:199`). Updated per-MB by
    /// `store_max_delta` when a "blocky" I16 macroblock is detected (Y2
    /// nonzero, all Y1 AC zero), then consumed by `adjust_filter_strength`
    /// after the encode loop to bump per-segment loop-filter levels for
    /// edge-rich segments. See libwebp `quant_enc.c:1031-1040, 1108-1113`
    /// and `filter_enc.c:198-237` (`VP8AdjustFilterStrength`). #34
    max_edge_per_segment: [i32; MAX_SEGMENTS],
    /// Set to `true` after the trellis-reuse cache verification has run
    /// once during the current encode. Used by the `cfg(debug_assertions)`
    /// guard in `record_residual_tokens_storing` to bound the cost of
    /// re-running `trellis_quantize_block` for verification — once per
    /// encode is enough to detect a regression at the moment it's
    /// introduced, and avoids 16× per-MB debug-build overhead at m5/m6.
    /// Reset to `false` per encode in `init_for_encode`. Used only under
    /// `cfg(debug_assertions)` but kept on the struct unconditionally so
    /// the field offsets match between debug and release builds.
    verified_trellis_reuse: bool,
}

impl<'a> Vp8Encoder<'a> {
    fn new(writer: &'a mut Vec<u8>) -> Self {
        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: core::array::from_fn(|_| Segment::default()),
            segments_enabled: false,
            segments_update_map: false,
            segment_tree_probs: [255, 255, 255], // Default probs
            segment_map: Vec::new(),
            fast_mb_hints: Vec::new(),
            fast_mb_uv_hints: Vec::new(),

            macroblock_no_skip_coeff: None,
            quantization_indices: QuantizationIndices::default(),

            token_probs: Default::default(),
            proba_stats: ProbaStats::new(),
            fast_probe_stat_limit: None,
            fast_probe_snapshot: None,
            fast_probe_skip_count: None,
            updated_probs: None,
            level_costs: LevelCosts::new(),
            // Trellis quantization for RD-optimized coefficient selection.
            // Uses proper probability-dependent init/EOB/skip costs from LevelCosts.
            do_trellis: true,
            // Trellis during I4 mode selection (RD_OPT_TRELLIS_ALL) - method 6
            do_trellis_i4_mode: false,
            // Error diffusion improves quality in smooth gradients
            do_error_diffusion: true,
            // Default to balanced method
            method: 4,
            sns_strength: 50,
            smooth_segment_map: false,
            cost_model: super::api::CostModel::ZenwebpDefault,
            multi_pass_stats: false,
            do_search: false,
            segment_quant_overrides: None,
            filter_strength: 60,
            filter_sharpness: 0,
            num_segments: 4,
            preset: super::api::Preset::Default,
            partition_limit: 0,

            top_complexity: Vec::new(),
            left_complexity: Complexity::default(),

            top_b_pred: Vec::new(),
            left_b_pred: [IntraMode::default(); 4],

            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],

            left_border_y: [0u8; 16 + 1],
            left_border_u: [0u8; 8 + 1],
            left_border_v: [0u8; 8 + 1],
            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),

            // Error diffusion starts with zero
            top_derr: Vec::new(),
            left_derr: [[0; 2]; 2],

            token_buffer: None,
            stored_mb_info: Vec::new(),
            stored_mb_coeffs: Vec::new(),
            max_edge_per_segment: [0; MAX_SEGMENTS],
            verified_trellis_reuse: false,
        }
    }

    /// Take the encoder's segment_map and grid dimensions for diagnostics.
    /// Used by the `target-zensim` closed-loop iteration to drive
    /// per-segment correction from the encoder's actual k-means assignment
    /// rather than a spatial proxy. Empty `segment_map` means segments
    /// were disabled (`num_segments == 1`) or the analysis pass didn't run.
    #[cfg(feature = "target-zensim")]
    fn into_diagnostics(self) -> super::api::EncodeDiagnostics {
        super::api::EncodeDiagnostics {
            segment_map: self.segment_map,
            mb_width: self.macroblock_width,
            mb_height: self.macroblock_height,
            num_segments: self.num_segments,
        }
    }

    /// Get the segment for a macroblock at (mbx, mby).
    ///
    /// When segments are enabled, looks up the segment ID from the segment map.
    /// Otherwise, returns segment 0.
    #[inline]
    fn get_segment_for_mb(&self, mbx: usize, mby: usize) -> &Segment {
        let segment_id = if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            self.segment_map[mb_idx] as usize
        } else {
            0
        };
        &self.segments[segment_id]
    }

    /// Get segment ID for a macroblock at (mbx, mby).
    #[inline]
    fn get_segment_id_for_mb(&self, mbx: usize, mby: usize) -> Option<usize> {
        if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            Some(self.segment_map[mb_idx] as usize)
        } else {
            None
        }
    }

    /// Compute updated probabilities from recorded statistics.
    ///
    /// IMPORTANT: For multi-pass encoding, this computes the optimal probabilities
    /// to use for emission. The header encoder compares against COEFF_PROBS (decoder
    /// defaults) to decide which updates to signal.
    ///
    /// Returns true if any probabilities were updated (changed from COEFF_PROBS defaults).
    /// This matches libwebp's FinalizeTokenProbas which sets dirty = has_changed where
    /// has_changed is true only if any coefficient was set to a value different from default.
    fn compute_updated_probabilities(&mut self) -> bool {
        // libwebp ships DEFAULT coefficient probabilities at RD_OPT_NONE
        // (method <= 2) with a single segment. `OneStatPass` returns
        // `size_p0 = ΣH + segment_hdr.size`, which is 0 there — H is 0 without
        // rate optimization, and the segment header is empty with one segment —
        // and `StatLoop` treats `size_p0 == 0` as a failure, returning before
        // `FinalizeTokenProbas` ever runs (frame_enc.c:648). The result is
        // default-proba coding (larger output, same pixels). Reproduce that
        // under `StrictLibwebpParity`; the tuned default keeps the
        // image-adapted proba update (which is strictly smaller).
        if self.cost_model == super::api::CostModel::StrictLibwebpParity
            && self.method <= 2
            && !self.segments_enabled
        {
            self.updated_probs = Some(COEFF_PROBS);
            return false;
        }

        // Always start from COEFF_PROBS (decoder defaults) for computing what to update.
        // This ensures the header signaling matches what the decoder expects.
        let mut updated = COEFF_PROBS;
        let mut has_changed = false;

        // Under libwebp fast_probe (method-0 parity), finalize from the
        // subset snapshot taken at the boundary rather than the full frame.
        let stats_src = self
            .fast_probe_snapshot
            .as_ref()
            .unwrap_or(&self.proba_stats);

        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for p in 0..11 {
                        // Compare against COEFF_PROBS (decoder defaults), not token_probs
                        let default_prob = COEFF_PROBS[t][b][c][p];
                        let update_prob = COEFF_UPDATE_PROBS[t][b][c][p];

                        let (should_update, new_p, _savings) =
                            stats_src.should_update(t, b, c, p, default_prob, update_prob);

                        // Update if savings are positive, matching libwebp's approach.
                        // The signaling cost (8 bits for value + 1 bit flag) is already
                        // included in the should_update calculation.
                        if should_update {
                            updated[t][b][c][p] = new_p;
                            // has_changed is true if new_p differs from default (matching libwebp)
                            has_changed |= new_p != default_prob;
                        }
                    }
                }
            }
        }

        // Full stats+proba dump per finalize, format-matched to the
        // instrumented libwebp's REFRESHDBG2 block in `FinalizeTokenProbas`
        // (libwebp--zen38trace) so the two can be diffed line-by-line. (#38)
        #[cfg(feature = "mode_debug")]
        if std::env::var("REFRESHDBG2").is_ok() {
            use core::fmt::Write as _;
            for t in 0..4 {
                for b in 0..8 {
                    for c in 0..3 {
                        let mut line = alloc::string::String::new();
                        let _ = write!(line, "FDUMP {t} {b} {c} S");
                        for p in 0..11 {
                            let v = stats_src.stats[t][b][c][p];
                            let _ = write!(line, " {}/{}", v & 0xffff, (v >> 16) & 0xffff);
                        }
                        let _ = write!(line, " P");
                        for p in 0..11 {
                            let _ = write!(line, " {}", updated[t][b][c][p]);
                        }
                        eprintln!("{line}");
                    }
                }
            }
            eprintln!("FDUMP-END");
        }

        // Always set updated_probs with the computed values.
        // For multi-pass, this ensures the header has the final probabilities to signal.
        // If no updates are beneficial, updated will equal COEFF_PROBS.
        self.updated_probs = Some(updated);

        has_changed
    }

    /// Reset encoder state for a new encoding pass.
    /// Used by multi-pass encoding (method >= 5) to reset all state before
    /// re-encoding with updated probability/cost tables.
    fn reset_for_new_pass(&mut self) {
        // Reset complexity tracking
        for complexity in self.top_complexity.iter_mut() {
            *complexity = Complexity::default();
        }
        self.left_complexity = Complexity::default();

        // Reset B-pred tracking
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        // Reset border pixels to initial state
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        for val in self.top_border_y.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_u.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_v.iter_mut() {
            *val = 127;
        }

        // Reset chroma error diffusion state
        self.left_derr = [[0; 2]; 2];
        for derr in self.top_derr.iter_mut() {
            *derr = [[0; 2]; 2];
        }

        // Estimate output size: ~0.3 bytes per pixel is conservative for most quality levels
        let num_pixels =
            usize::from(self.macroblock_width) * usize::from(self.macroblock_height) * 256; // 16x16 per macroblock
        let estimated_partition_size = num_pixels / 4; // ~0.25 bytes per pixel for coefficients

        // Reset partitions with pre-allocated capacity
        self.partitions = vec![ArithmeticEncoder::with_capacity(estimated_partition_size)];

        // Reset encoder (header is small, ~1KB is plenty)
        self.encoder = ArithmeticEncoder::with_capacity(1024);

        // Reset per-segment max edge tracking for this encode (#34).
        self.max_edge_per_segment = [0; MAX_SEGMENTS];

        // Reset the trellis-reuse verification flag so the
        // `cfg(debug_assertions)` guard in `record_residual_tokens_storing`
        // re-verifies the cache once per new encode (catches drift on the
        // first block, then stays out of the per-MB hot path).
        self.verified_trellis_reuse = false;
    }

    /// Update `max_edge_per_segment` for a "blocky" I16 macroblock.
    ///
    /// Port of libwebp's `StoreMaxDelta` (`quant_enc.c:1031-1040`). Looks at
    /// the first three AC coefficients of the Y2 (WHT) block — these encode
    /// the average DC differences between adjacent Y1 sub-blocks, so a large
    /// magnitude here corresponds to visible inter-block edges that the loop
    /// filter should attenuate.
    ///
    /// `y2_zigzag` is in zigzag order, matching libwebp's `y_dc_levels`
    /// (which is the output of `VP8EncQuantizeBlockWHT`, also zigzag-ordered).
    /// Indices 1, 2, 4 correspond to natural-order positions 1, 4, 2.
    #[inline]
    /// Whether the edge-based loop-filter bump (`store_max_delta` →
    /// `VP8AdjustFilterStrength`) should accumulate this MB. libwebp only calls
    /// `StoreMaxDelta` from `PickBestIntra16` — the RD path (method >= 3); at
    /// m0-2 (`RefineUsingDistortion`/`SimpleQuantize`) it never accumulates, so
    /// `max_edge` stays 0 and the filter is left at its analysis-time level.
    /// zenwebp historically accumulated at all methods (helps smooth-gradient
    /// filtering at low q — #44); under StrictLibwebpParity we restrict it to
    /// the RD path to match libwebp's per-segment filter levels at m0-2.
    fn store_max_edge_active(&self) -> bool {
        self.method >= 3 || self.cost_model != super::api::CostModel::StrictLibwebpParity
    }

    fn store_max_delta(&mut self, segment_id: usize, y2_zigzag: &[i32; 16]) {
        let v0 = y2_zigzag[1].unsigned_abs();
        let v1 = y2_zigzag[2].unsigned_abs();
        let v2 = y2_zigzag[4].unsigned_abs();
        let max_v = v0.max(v1).max(v2) as i32;
        if max_v > self.max_edge_per_segment[segment_id] {
            self.max_edge_per_segment[segment_id] = max_v;
        }
    }

    /// Post-encode adjustment of per-segment loop-filter strengths based on
    /// observed edge magnitudes (port of libwebp's `VP8AdjustFilterStrength`,
    /// `filter_enc.c:198-237`, called from `PostLoopFinalize` in
    /// `frame_enc.c:730`).
    ///
    /// For each segment, computes a target filter strength from the maximum
    /// observed Y2 edge magnitude scaled by the Y2 AC quantizer:
    /// `delta = (max_edge * y2.q[1]) >> 3` (the `>> 3` accounts for inverse
    /// WHT scaling). This delta is mapped to a filter level via
    /// `filter_strength_from_delta`, and the per-segment level is bumped
    /// upward only — never lowered. Finally the frame-level
    /// `filter_level` is raised to the maximum per-segment fstrength.
    ///
    /// Effect: edge-rich segments (text/charts) get stronger filtering than
    /// the analysis-time estimate, reducing blocking artifacts.
    ///
    /// Note: zenwebp stores `loopfilter_level` as a signed delta from
    /// `frame.filter_level`, while libwebp stores it as an absolute
    /// `dqm.fstrength`. We convert to absolute, bump, then recompute the
    /// delta against the (possibly raised) frame-level filter.
    ///
    /// Per-MB `D > min_disto` gate: applied at the `store_max_delta` call
    /// site (issue #44). `MacroblockInfo::intra16_d` carries the winning
    /// I16 mode's raw source-vs-reconstruction SSE through from
    /// `pick_best_intra16` / `pick_intra16_fast_dc`; only MBs whose `D`
    /// exceeds `Segment::min_disto` (= `20 * y1.q[0]` per
    /// `quant_enc.c:264`) contribute to `max_edge_per_segment`. This
    /// matches libwebp's gate at `quant_enc.c:1111` and prevents
    /// flat-region MBs (whose Y2 AC may encode faint cross-MB DC drift)
    /// from over-bumping the filter on `color_blocks`-style content.
    ///
    /// When `max_edge_per_segment` is all-zero (no blocky-I16 MBs were
    /// observed), the body still runs and may raise `frame.filter_level`
    /// to the maximum existing per-segment absolute fstrength
    /// (`frame_level + loopfilter_level`). This matches libwebp: the
    /// function unconditionally assigns `enc->filter_hdr.level = max_level`
    /// at `filter_enc.c:236`.
    fn adjust_filter_strength(&mut self) {
        if self.filter_strength == 0 {
            return;
        }

        // Bounds reference for the asserts and clamps below:
        //   frame_level       ∈ [0, 63]   — u8 set by `compute_filter_level`
        //                                   (≤ 63 by construction) or by an
        //                                   earlier call to this function.
        //   loopfilter_level  ∈ [-63, 63] — i8, signed-6-bit bitstream field;
        //                                   clamped at write time.
        //   absolute[s]       ∈ [-63, 126] in worst case (frame=63 + loop=63
        //                                   or frame=0 + loop=-63).
        //   edge_strength     ∈ [0, 63]   — `LEVELS_FROM_DELTA` saturates
        //                                   at 63 (loop filter is 6-bit).
        let mut absolute = [0i32; MAX_SEGMENTS];
        let frame_level = i32::from(self.frame.filter_level);
        debug_assert!(
            (0..=63).contains(&frame_level),
            "frame.filter_level={frame_level} outside [0, 63]"
        );
        for s in 0..MAX_SEGMENTS {
            absolute[s] = frame_level + i32::from(self.segments[s].loopfilter_level);
        }

        let mut max_level = 0i32;
        for s in 0..MAX_SEGMENTS {
            let max_edge = self.max_edge_per_segment[s];
            #[cfg(feature = "mode_debug")]
            if std::env::var("FSDBG").is_ok() {
                eprintln!(
                    "ZENADJ seg{s} max_edge={max_edge} y2q1={} fstrength_before={}",
                    self.segments[s].y2ac, self.segments[s].loopfilter_level
                );
            }
            // y2.q[1] is the Y2 AC quantizer in libwebp's expanded matrix.
            // zenwebp stores it directly on Segment as `y2ac` (i16).
            let y2_q1 = i32::from(self.segments[s].y2ac);
            let delta_raw = max_edge.saturating_mul(y2_q1) >> 3;
            // Clamp to u8 for the table lookup (table saturates at 63 well
            // before MAX_DELTA_SIZE=64; libwebp clamps inside the function).
            let delta_u8 = delta_raw.clamp(0, 255) as u8;
            let edge_strength = i32::from(super::cost::filter_strength_from_delta(
                self.filter_sharpness,
                delta_u8,
            ));
            if edge_strength > absolute[s] {
                absolute[s] = edge_strength;
            }
            if absolute[s] > max_level {
                max_level = absolute[s];
            }
        }

        // Raise frame filter_level to the max per-segment fstrength
        // (matches libwebp `enc->filter_hdr.level = max_level`). `max_level`
        // can theoretically reach 126 (= max possible absolute[s]) so the
        // clamp can fire; the bitstream field is 6-bit unsigned (0–63).
        let new_frame_level = max_level.clamp(0, 63) as u8;
        self.frame.filter_level = new_frame_level;

        // Recompute per-segment deltas against the (possibly raised) frame
        // level. With absolute[s] ∈ [-63, 126] and new_frame ∈ [0, 63], the
        // delta is in [-126, 126]; the bitstream signed-6-bit field is
        // [-63, 63], so the clamp can fire at extremes. In well-formed
        // pre-state where frame_level ≈ analysis-time `compute_filter_level`
        // and loopfilter_level deltas are small, the clamp is a no-op.
        let new_frame_i32 = i32::from(new_frame_level);
        for s in 0..MAX_SEGMENTS {
            let raw_delta = absolute[s] - new_frame_i32;
            let clamped = raw_delta.clamp(-63, 63);
            debug_assert!(
                raw_delta == clamped,
                "loopfilter_level delta clamp fired: seg={s} absolute={} \
                 new_frame={new_frame_level} raw_delta={raw_delta} → clamped={clamped} — \
                 either the per-segment fstrength range exceeded expectation or \
                 frame_level was set outside [0, 63] before this call",
                absolute[s],
            );
            self.segments[s].loopfilter_level = clamped as i8;
        }
    }

    /// Port of libwebp's `WebPCleanupTransparentArea` (YUV flavor,
    /// `picture_tools_enc.c`), run AFTER RGB→YUV conversion exactly like
    /// libwebp's lossy flow (`webp_enc.c`: convert, then cleanup): per 8×8
    /// luma block, mixed-alpha blocks get their invisible lumas replaced by
    /// the block's visible-luma average (`SmoothenBlock`), fully-transparent
    /// blocks are flattened (Y 8×8, U/V 4×4) to the first pixel of the run;
    /// right/bottom partial blocks are smoothened only. `exact(true)` opts
    /// out, same as libwebp. Alpha itself is untouched. (#38)
    #[allow(clippy::too_many_arguments)]
    fn cleanup_transparent_area_yuv(
        alpha: &[u8],
        a_stride: usize,
        y_plane: &mut [u8],
        y_stride: usize,
        u_plane: &mut [u8],
        v_plane: &mut [u8],
        uv_stride: usize,
        width: usize,
        height: usize,
    ) {
        const SIZE: usize = 8;
        const SIZE2: usize = SIZE / 2;

        // `SmoothenBlock`: average visible lumas over invisible ones;
        // returns true when the whole block is transparent.
        fn smoothen_block(
            alpha: &[u8],
            a_stride: usize,
            a_off: usize,
            luma: &mut [u8],
            y_stride: usize,
            y_off: usize,
            w: usize,
            h: usize,
        ) -> bool {
            let mut sum = 0u32;
            let mut count = 0u32;
            for y in 0..h {
                for x in 0..w {
                    if alpha[a_off + y * a_stride + x] != 0 {
                        count += 1;
                        sum += u32::from(luma[y_off + y * y_stride + x]);
                    }
                }
            }
            if count > 0 && count < (w * h) as u32 {
                let avg = (sum / count) as u8;
                for y in 0..h {
                    for x in 0..w {
                        if alpha[a_off + y * a_stride + x] == 0 {
                            luma[y_off + y * y_stride + x] = avg;
                        }
                    }
                }
            }
            count == 0
        }

        fn flatten(plane: &mut [u8], off: usize, v: u8, stride: usize, size: usize) {
            for y in 0..size {
                plane[off + y * stride..off + y * stride + size].fill(v);
            }
        }

        let mut y = 0usize;
        while y + SIZE <= height {
            let a_row = y * a_stride;
            let y_row = y * y_stride;
            let uv_row = (y / 2) * uv_stride;
            let mut need_reset = true;
            let mut values = [0u8; 3];
            let mut x = 0usize;
            while x + SIZE <= width {
                if smoothen_block(
                    alpha,
                    a_stride,
                    a_row + x,
                    y_plane,
                    y_stride,
                    y_row + x,
                    SIZE,
                    SIZE,
                ) {
                    if need_reset {
                        values[0] = y_plane[y_row + x];
                        values[1] = u_plane[uv_row + (x >> 1)];
                        values[2] = v_plane[uv_row + (x >> 1)];
                        need_reset = false;
                    }
                    flatten(y_plane, y_row + x, values[0], y_stride, SIZE);
                    flatten(u_plane, uv_row + (x >> 1), values[1], uv_stride, SIZE2);
                    flatten(v_plane, uv_row + (x >> 1), values[2], uv_stride, SIZE2);
                } else {
                    need_reset = true;
                }
                x += SIZE;
            }
            if x < width {
                smoothen_block(
                    alpha,
                    a_stride,
                    a_row + x,
                    y_plane,
                    y_stride,
                    y_row + x,
                    width - x,
                    SIZE,
                );
            }
            y += SIZE;
        }
        if y < height {
            let sub_height = height - y;
            let a_row = y * a_stride;
            let y_row = y * y_stride;
            let mut x = 0usize;
            while x + SIZE <= width {
                smoothen_block(
                    alpha,
                    a_stride,
                    a_row + x,
                    y_plane,
                    y_stride,
                    y_row + x,
                    SIZE,
                    sub_height,
                );
                x += SIZE;
            }
            if x < width {
                smoothen_block(
                    alpha,
                    a_stride,
                    a_row + x,
                    y_plane,
                    y_stride,
                    y_row + x,
                    width - x,
                    sub_height,
                );
            }
        }
    }

    /// Convert input pixels to YUV420 planes and prime the encoder frame buffers.
    ///
    /// Handles ARGB→RGBA fixup, transparent-area cleanup, planar YUV import,
    /// sharp-YUV / fast-YUV / luma-only dispatch, and a strided-buffer-size
    /// sanity check, then forwards into [`Self::setup_encoding`] which fills
    /// `self.frame.{ybuf,ubuf,vbuf}` and initialises per-segment quant /
    /// filter state.
    fn prepare_input_for_encoding(
        &mut self,
        data: &[u8],
        color: PixelLayout,
        width: u16,
        height: u16,
        stride: usize,
        params: &super::api::EncoderParams,
    ) {
        // For ARGB input, convert to RGBA so the standard RGBA code path handles it.
        let argb_converted;
        let (data, color) = if color == PixelLayout::Argb8 {
            let w = usize::from(width);
            let h = usize::from(height);
            let bpp = 4usize;
            let stride_bytes = stride * bpp;
            let row_bytes = w * bpp;
            let mut out = alloc::vec![0u8; w * h * 4];
            for y in 0..h {
                garb::bytes::argb_to_rgba(
                    &data[y * stride_bytes..y * stride_bytes + row_bytes],
                    &mut out[y * w * 4..(y + 1) * w * 4],
                )
                .expect("validated buffer sizes");
            }
            argb_converted = out;
            (argb_converted.as_slice(), PixelLayout::Rgba8)
        } else {
            (data, color)
        };

        let (mut y_bytes, mut u_bytes, mut v_bytes) = if color == PixelLayout::Yuv420 {
            // YUV420 planar data: [Y, U, V] packed into a single buffer
            let w = usize::from(width);
            let h = usize::from(height);
            let y_size = w * h;
            let uv_w = w.div_ceil(2);
            let uv_h = h.div_ceil(2);
            let uv_size = uv_w * uv_h;

            let y_plane = &data[..y_size];
            let u_plane = &data[y_size..y_size + uv_size];
            let v_plane = &data[y_size + uv_size..y_size + uv_size * 2];

            crate::decoder::yuv::import_yuv420_planes(y_plane, u_plane, v_plane, width, height)
        } else if params.sharp_yuv.is_some()
            && !matches!(color, PixelLayout::L8 | PixelLayout::La8)
            && (params.cost_model == super::api::CostModel::StrictLibwebpParity
                || params.sharp_yuv == Some(zenyuv::SharpYuvConfig::default()))
        {
            // libwebp's SharpYUV algorithm, byte-exact port. Used under
            // parity AND as the tuned `.sharp_yuv(true)` default — measured
            // +1.0..+1.8 zsim over standard conversion (4-6x zenyuv's
            // Newton refinement) at +2-5% bytes, and 1.5x faster than
            // libwebp's own SSE2 build; see
            // `benchmarks/sharpyuv_port_2026-07-16.md`. A custom
            // `sharp_yuv_config(..)` still selects zenyuv's converter below.
            if width >= 4 && height >= 4 {
                crate::encoder::sharpyuv::convert_image_sharp_libwebp(
                    data, color, width, height, stride,
                )
            } else {
                // libwebp disables iterative conversion below
                // kMinDimensionIterativeConversion (4) — standard path.
                let prec = if params.cost_model == super::api::CostModel::StrictLibwebpParity {
                    crate::decoder::yuv::ChromaPrec::LibwebpExact
                } else {
                    crate::decoder::yuv::ChromaPrec::TunedByteRound
                };
                crate::decoder::yuv::convert_image_yuv_fast(
                    data, color, width, height, stride, prec,
                )
            }
        } else if let Some(sharp_cfg) = &params.sharp_yuv {
            crate::decoder::yuv::convert_image_sharp_yuv_with_config(
                data, color, width, height, stride, *sharp_cfg,
            )
        } else {
            match color {
                // zenyuv (SIMD Y) + gamma-corrected scalar chroma.
                // Under StrictLibwebpParity the chroma uses libwebp's exact
                // YUV_FIX+2 precision; otherwise the tuned byte-rounded path
                // (measurably better on synthetic low-q — see the zensim gate).
                PixelLayout::Rgb8 | PixelLayout::Rgba8 | PixelLayout::Bgr8 | PixelLayout::Bgra8 => {
                    let prec = if params.cost_model == super::api::CostModel::StrictLibwebpParity {
                        crate::decoder::yuv::ChromaPrec::LibwebpExact
                    } else {
                        crate::decoder::yuv::ChromaPrec::TunedByteRound
                    };
                    crate::decoder::yuv::convert_image_yuv_fast(
                        data, color, width, height, stride, prec,
                    )
                }
                PixelLayout::L8 => convert_image_y::<1>(data, width, height, stride),
                PixelLayout::La8 => convert_image_y::<2>(data, width, height, stride),
                PixelLayout::Yuv420 | PixelLayout::Argb8 => unreachable!(),
            }
        };

        if color != PixelLayout::Yuv420 {
            let bpp = match color {
                PixelLayout::L8 => 1usize,
                PixelLayout::La8 => 2,
                PixelLayout::Rgb8 | PixelLayout::Bgr8 => 3,
                PixelLayout::Rgba8 | PixelLayout::Bgra8 | PixelLayout::Argb8 => 4,
                PixelLayout::Yuv420 => unreachable!(),
            };
            let w = usize::from(width);
            let h = usize::from(height);
            let min_size = if h > 0 {
                stride * bpp * (h - 1) + w * bpp
            } else {
                0
            };
            assert!(
                data.len() >= min_size,
                "buffer too small: got {}, need at least {} for {}x{} stride={} {:?}",
                data.len(),
                min_size,
                w,
                h,
                stride,
                color
            );
        }

        // libwebp default (`exact=0`): after conversion, smoothen/flatten the
        // Y/U/V under transparent alpha so invisible pixels compress to
        // nothing (`WebPCleanupTransparentArea`, YUV flavor — run at the
        // same point in the flow as libwebp's lossy encode). `exact(true)`
        // opts out. The ALPH plane reads the ORIGINAL data; alpha is
        // untouched here.
        if !params.exact
            && matches!(
                color,
                PixelLayout::Rgba8 | PixelLayout::Bgra8 | PixelLayout::La8
            )
        {
            let (bpp, a_off) = if color == PixelLayout::La8 {
                (2usize, 1usize)
            } else {
                (4, 3)
            };
            let w = usize::from(width);
            let h = usize::from(height);
            let alpha_plane: alloc::vec::Vec<u8> = (0..h)
                .flat_map(|y| {
                    let row = y * stride * bpp;
                    data[row..row + w * bpp].chunks_exact(bpp).map(|p| p[a_off])
                })
                .collect();
            if alpha_plane.contains(&0) {
                let y_stride = usize::from(width.div_ceil(16)) * 16;
                let uv_stride = usize::from(width.div_ceil(16)) * 8;
                Self::cleanup_transparent_area_yuv(
                    &alpha_plane,
                    w,
                    &mut y_bytes,
                    y_stride,
                    &mut u_bytes,
                    &mut v_bytes,
                    uv_stride,
                    w,
                    h,
                );
                // The MB-alignment padding was replicated from the
                // PRE-cleanup edges; refresh it so padded columns/rows
                // extend the cleaned pixels (libwebp replicates from its
                // tight planes at import time, i.e. post-cleanup).
                let repad = |plane: &mut [u8], tw: usize, th: usize, pw: usize| {
                    for y in 0..th {
                        let last = plane[y * pw + tw - 1];
                        plane[y * pw + tw..(y + 1) * pw].fill(last);
                    }
                    let (filled, rest) = plane.split_at_mut(th * pw);
                    let last_row = &filled[(th - 1) * pw..];
                    for chunk in rest.chunks_exact_mut(pw) {
                        chunk.copy_from_slice(last_row);
                    }
                };
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                repad(&mut y_bytes, w, h, y_stride);
                repad(&mut u_bytes, cw, ch, uv_stride);
                repad(&mut v_bytes, cw, ch, uv_stride);
            }
        }

        // ZYUVDUMP=<path>: dump the tight Y/U/V region of the encoder input
        // planes (mb-aligned strides trimmed) for input-conversion parity
        // diffing against libwebp's `dump_encoder_yuv` (#38 SharpYUV).
        #[cfg(feature = "mode_debug")]
        if let Ok(path) = std::env::var("ZYUVDUMP") {
            let w = usize::from(width);
            let h = usize::from(height);
            let (lw, cw) = (w.div_ceil(16) * 16, w.div_ceil(16) * 8);
            let (tcw, tch) = (w.div_ceil(2), h.div_ceil(2));
            let mut out = alloc::vec::Vec::new();
            for j in 0..h {
                out.extend_from_slice(&y_bytes[j * lw..j * lw + w]);
            }
            for j in 0..tch {
                out.extend_from_slice(&u_bytes[j * cw..j * cw + tcw]);
            }
            for j in 0..tch {
                out.extend_from_slice(&v_bytes[j * cw..j * cw + tcw]);
            }
            std::fs::write(path, out).unwrap();
        }

        self.setup_encoding(
            params.lossy_quality,
            width,
            height,
            y_bytes,
            u_bytes,
            v_bytes,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_image(
        &mut self,
        data: &[u8],
        color: PixelLayout,
        width: u16,
        height: u16,
        stride: usize,
        params: &super::api::EncoderParams,
        stop: &dyn enough::Stop,
        progress: &dyn super::api::EncodeProgress,
    ) -> super::api::EncodeResult<super::api::EncodeStats> {
        // Store method and configure features based on it
        self.method = params.method.min(6); // Clamp to 0-6
        // Method feature mapping (aligned with libwebp):
        //   m0-2: RD_OPT_NONE - fast mode, no RD optimization
        //   m3-4: RD_OPT_BASIC - RD scoring for mode selection, no trellis
        //   m5:   RD_OPT_TRELLIS - trellis quantization during encoding
        //   m6:   RD_OPT_TRELLIS_ALL - trellis during I4 mode selection
        self.do_trellis = self.method >= 5;
        self.do_trellis_i4_mode = self.method >= 6;
        // Store tuning parameters
        self.sns_strength = params.sns_strength.min(100);
        self.smooth_segment_map = params.smooth_segment_map;
        self.cost_model = params.cost_model;
        // libwebp allocates the chroma error-diffusion buffers only when
        // `quality <= ERROR_DIFFUSION_QUALITY (98)` (`webp_enc.c:167`) — at
        // q99/q100 `top_derr == NULL` and `CorrectDCValues` never runs, so
        // parity must disable diffusion there (traced at q99 m4: zen's H/TM
        // UV candidates carried diffused DCs, libwebp's didn't, flipping ~14%
        // of UV picks). The tuned default keeps diffusion at every quality:
        // at q99+ the DC errors it shapes are ±1-2 steps and the effect is
        // noise-level, so switching tuned off up there is not worth a bytes
        // change without a measured win. (#38)
        self.do_error_diffusion = params.lossy_quality <= 98
            || self.cost_model != super::api::CostModel::StrictLibwebpParity;
        self.multi_pass_stats = params.multi_pass_stats;
        // Mirrors libwebp's `enc->do_search = (config->target_size > 0 ||
        // config->target_PSNR > 0)` (`webp_enc.c:118`).
        self.do_search = params.target_size > 0 || params.target_psnr > 0.0;
        self.segment_quant_overrides = params.segment_quant_overrides;
        self.filter_strength = params.filter_strength.min(100);
        self.filter_sharpness = params.filter_sharpness.min(7);
        self.num_segments = params.num_segments.clamp(1, 4);
        self.preset = params.preset;
        self.partition_limit = params.partition_limit.unwrap_or(0).min(100);
        self.prepare_input_for_encoding(data, color, width, height, stride, params);

        // Calculate initial level costs for mode selection and trellis
        if self.level_costs.is_dirty() {
            self.level_costs.calculate(&self.token_probs);
        }

        // Token buffer encoding (matching libwebp's VP8EncTokenLoop).
        //
        // Records coefficient decisions as compact tokens while collecting
        // probability statistics with mid-stream refresh. For method >= 5,
        // performs multiple passes: each pass does FULL re-encoding with
        // level_costs derived from the previous pass's observed probabilities.
        //
        // Benefits of multi-pass (matches libwebp's approach):
        // - Image-specific probability tables (not generic defaults)
        // - Better mode selection with empirical cost tables
        // - More efficient arithmetic coding with observed probabilities
        //
        // Key insight: Each pass does full re-encoding (predict → residual → DCT →
        // quantize → record). Mode decisions and quantization may differ between
        // passes because level_costs change. Borders come from reconstructed pixels,
        // so each pass has consistent predictions within itself.

        let num_mb = usize::from(self.macroblock_width) * usize::from(self.macroblock_height);

        // Number of passes based on method (mirrors libwebp's `StatLoop` behavior).
        // libwebp runs `config->pass` stat-collection passes before the final emit
        // pass, refreshing both `proba` and `level_costs` between iterations
        // (`frame_enc.c:626-684, 795-906, 840-844`). The first pass uses default
        // costs; the second pass benefits from image-tuned probabilities, which
        // makes mode selection and trellis pick options that are actually cheap
        // under the real distribution.
        //
        // Earlier in zenwebp's history multi-pass was tested without refreshing
        // `level_costs` mid-pass and reportedly hurt compression — but the
        // existing infrastructure here (lines 884-895) DOES rebuild level_costs
        // between passes, so the prior negative result no longer applies. #27.
        //
        // 2 passes at m4 only. m5/m6 already use trellis quantization, which is
        // image-adapted via the per-pass `proba_stats` accumulation; adding a
        // second pass at m5/m6 measurably regresses size (+0.6 to +0.9% on
        // CID22 — see `differences/baselines/post-batch2-27.tsv`). At m4 the
        // second pass net-helps (-0.1 to -0.4%) because the simple-quant
        // m4 path doesn't have a per-MB feedback mechanism otherwise.
        // Multi-pass stat collection is opt-in via `LossyConfig::with_multi_pass_stats(true)`
        // (default OFF, gated only at m4). m5/m6 already image-adapt via per-pass
        // `proba_stats` in trellis; adding a second pass at those tiers regresses size
        // (see #27 investigation). Multi-pass at m4 doubles encode time for ~0.1% size
        // win on photos — useful inside `target_size`/`target_zensim` search loops.
        // libwebp calls `StatLoop(enc)` UNCONDITIONALLY as the first statement of
        // `VP8EncLoop` (frame_enc.c), for every method: a stats pass that
        // finalizes the coefficient probabilities (`FinalizeTokenProbas`) and
        // rebuilds the level-cost tables (`VP8CalculateLevelCosts`) BEFORE the
        // encode loop. So every RD decision libwebp makes — from the very first
        // MB — is scored against image-adapted probabilities, whereas zen's
        // single pass scores against the shipped defaults and only drifts toward
        // adapted values via the mid-row refresh.
        //
        // That is the root of the m3-m6 byte-parity gap: those are exactly the
        // cost-driven methods (332 of 426 remaining failures), and it shows up
        // most at high q where there are more and larger coefficients for a wrong
        // rate to mis-score. Traced to a single table entry at q90/m3 mb(28,7):
        // identical levels, identical everything on the rate path, but
        // `prob[band1][ctx2]` was still `[39,77,162,232,…]` (defaults) in zen vs
        // `[17,51,85,127,…]` (adapted) in libwebp. See
        // `benchmarks/byteparity_scope_2026-07-14.md`.
        //
        // zen's existing two-pass machinery IS this: pass 0 accumulates stats,
        // then pass 1 applies `updated_probs` and rebuilds `level_costs` (below).
        // It was just gated to m4. Under parity, run it for every method.
        // Parity-gated: the tuned default keeps its prior pass count, since a
        // stats pass changes its bytes and would need a fresh A/B first.
        // MEASURED AND REJECTED (2026-07-15): naively running zen's existing
        // two-pass here under parity (`= 2` for every method) REGRESSED the
        // byteparity grid 3578/4004 -> 2299/4004, with m3-m6 going from ~70-99
        // failures each to ~400 (nearly every cell). zen's pass-0/pass-1 is NOT
        // semantically libwebp's StatLoop: libwebp's stats pass uses
        // `rd_opt = (method >= 3 || do_search) ? RD_OPT_BASIC : RD_OPT_NONE`
        // (so m5/m6 stat with BASIC, not trellis), never emits, and its token
        // loop then calls `ResetTokenStats` on the last pass and re-finalizes
        // probabilities from THAT pass's own stats every `max_count` MBs.
        // A faithful port has to reproduce those semantics; do not re-attempt
        // by flipping this count.
        let num_passes: usize = if self.multi_pass_stats && self.method == 4 {
            2
        } else {
            1
        };

        // Stats accumulators (populated during last pass)
        let mut final_sse_y: u64 = 0;
        let mut final_sse_u: u64 = 0;
        let mut final_sse_v: u64 = 0;
        let mut final_block_count_i4: u32 = 0;
        let mut final_block_count_i16: u32 = 0;
        let mut final_skip_mb: u32 = 0;

        for pass in 0..num_passes {
            let is_last_pass = pass == num_passes - 1;

            // Clear token buffer for this pass
            let mut tb = residuals::TokenBuffer::with_estimated_capacity(num_mb);
            // libwebp picks its coefficient recorder by which encode loop runs:
            // `use_tokens = (rd_opt >= RD_OPT_BASIC)` sends m3-m6 through
            // `VP8EncTokenLoop` -> `VP8RecordCoeffTokens` (Cat5/6 statistic goes
            // to node 9), while m0-m2 (RD_OPT_NONE) run plain `VP8EncLoop` ->
            // `RecordResiduals` -> `VP8RecordCoeffs`, whose level-code table
            // records node 10. The two upstream paths genuinely disagree, so
            // parity has to follow the method. The tuned default keeps zenwebp's
            // prior node-10 accounting for every method, leaving its bytes
            // unchanged. See `TokenBuffer::cat56_stat_node9`. (#38)
            tb.set_cat56_stat_node9(
                self.cost_model == super::api::CostModel::StrictLibwebpParity && self.method >= 3,
            );
            self.token_buffer = Some(tb);

            // Reset statistics only on first pass or last pass (matches libwebp).
            // For intermediate passes, statistics ACCUMULATE across passes, giving
            // a more robust probability estimate based on multiple encodings.
            if pass == 0 || is_last_pass {
                self.proba_stats.reset();
            }

            // libwebp fast_probe subset (method-0 StatLoop): the emitted
            // coefficient probas are finalized from only the first
            // (num_mb>>2 | 50) MBs. Arm the snapshot boundary under parity so
            // the final `compute_updated_probabilities` uses that subset. If
            // the frame has fewer MBs than the limit, the snapshot never fires
            // and the whole frame is used — matching libwebp, which also
            // records every MB in that case.
            // libwebp's `StatLoop` shortens the stats pass for m0 and m3:
            //   fast_probe = ((method == 0 || method == 3) && !do_search)
            //   m3: nb_mbs = (nb_mbs > 200) ? nb_mbs >> 1 : 100  // needs more
            //                                                    // stats to be
            //                                                    // reliable
            //   m0: nb_mbs = (nb_mbs > 200) ? nb_mbs >> 2 : 50
            // so the emitted probas are finalized from only that many MBs. m4-m6
            // use the whole frame. `do_search` is libwebp's target-size search,
            // which disables fast_probe entirely.
            // NOTE: libwebp's `fast_probe` also shortens the m3 stats pass
            // (`nb_mbs = (nb_mbs > 200) ? nb_mbs >> 1 : 100`), but that belongs
            // to `StatLoop`, which zen does not have. Applying the m3 limit to
            // this mid-row refresh instead is NOT equivalent and was measured as
            // part of the rejected two-pass attempt above. Keep it m0-only until
            // StatLoop itself is ported. (`self.do_search` carries libwebp's
            // `enc->do_search`, which gates `fast_probe`, ready for that port.)
            self.fast_probe_snapshot = None;
            self.fast_probe_skip_count = None;
            self.fast_probe_stat_limit = if self.cost_model
                == super::api::CostModel::StrictLibwebpParity
                && self.method == 0
                && !self.do_search
            {
                Some(if num_mb > 200 { num_mb >> 2 } else { 50 })
            } else {
                None
            };

            // Clear stored info - we only keep the last pass's results
            self.stored_mb_info.clear();
            self.stored_mb_info.reserve(num_mb);
            // stored_mb_coeffs is only needed for multi-pass re-encoding.
            // Since num_passes == 1, skip the 1.6MB allocation entirely.
            if num_passes > 1 {
                self.stored_mb_coeffs.clear();
                self.stored_mb_coeffs.reserve(num_mb);
            }

            if pass > 0 {
                // Pass 1+: Apply updated probabilities from previous pass
                // This gives us image-specific cost tables for mode selection and trellis
                if let Some(ref updated) = self.updated_probs {
                    self.token_probs = *updated;
                }

                // Recalculate level_costs from the new probabilities
                // This is the key to multi-pass: trellis and mode selection use
                // empirical costs, potentially making different (better) decisions
                self.level_costs.mark_dirty();
                self.level_costs.calculate(&self.token_probs);
            }

            // Reset all encoder state for this pass
            self.reset_for_new_pass();

            // Mid-stream refresh interval: roughly every total_mb/8 macroblocks
            let max_count = (num_mb / 8).max(96) as i32; // MIN_COUNT = 96 (matches libwebp)
            let mut refresh_countdown = max_count;

            let mut total_mb: u32 = 0;
            let mut skip_mb: u32 = 0;
            let mut block_count_i4: u32 = 0;
            let mut block_count_i16: u32 = 0;
            // SSE accumulators for PSNR computation
            let mut sse_y: u64 = 0;
            let mut sse_u: u64 = 0;
            let mut sse_v: u64 = 0;
            let y_stride = usize::from(self.macroblock_width) * 16;
            let uv_stride = usize::from(self.macroblock_width) * 8;
            let mut last_progress_pct: u8 = 0;

            // ===== ENCODING PASS =====
            // Each pass does full encoding: mode selection + transform + quantize + record
            let mut row_state = MbRowState {
                total_mb,
                skip_mb,
                block_count_i4,
                block_count_i16,
                sse_y,
                sse_u,
                sse_v,
                refresh_countdown,
            };
            for mby in 0..self.macroblock_height {
                self.encode_mb_row(
                    mby,
                    y_stride,
                    uv_stride,
                    num_passes,
                    max_count,
                    &mut row_state,
                    stop,
                )?;

                // Report progress after each row
                let pct = ((u32::from(mby) + 1) * 100 / u32::from(self.macroblock_height)) as u8;
                if pct > last_progress_pct {
                    last_progress_pct = pct;
                    progress
                        .on_progress(pct.min(99))
                        .map_err(|e| at!(EncodeError::from(e)))?; // cap at 99, report 100 after finalize
                }
            }
            total_mb = row_state.total_mb;
            skip_mb = row_state.skip_mb;
            block_count_i4 = row_state.block_count_i4;
            block_count_i16 = row_state.block_count_i16;
            sse_y = row_state.sse_y;
            sse_u = row_state.sse_u;
            sse_v = row_state.sse_v;
            refresh_countdown = row_state.refresh_countdown;
            let _ = refresh_countdown; // consumed when the pass ends

            // Compute skip probability from actual data.
            // libwebp gates per-MB skip-bit emission on `skip_proba < SKIP_PROBA_THRESHOLD (250)`
            // (libwebp `src/enc/frame_enc.c:118-132` `FinalizeSkipProba` / `use_skip_proba`).
            // When fewer than ~2% of MBs are skip-eligible, the per-MB skip bit costs more than
            // it saves, so libwebp signals `use_skip_proba=0` in the frame header and omits the
            // per-MB skip bits entirely — the decoder then assumes every MB has residual data.
            //
            // We now record EOB-at-0 tokens for every skipped MB during the encode loop (matches
            // libwebp's `RecordTokens` behavior — `frame_enc.c:415`), and the token buffer tracks
            // per-MB span offsets via `begin_mb()`. At emit time:
            //   - `use_skip_proba=0` (`macroblock_no_skip_coeff = None`): emit ALL tokens — the
            //     decoder reads residuals for every MB, so the EOB tokens for skipped MBs are
            //     exactly what it expects.
            //   - `use_skip_proba=1` (`Some(prob)`): the per-MB skip flag tells the decoder which
            //     MBs to skip; we filter those MBs' tokens out of the emission stream so the
            //     decoder sees the residuals exactly where it expects them.
            // This is the full #25 fix — the gate now fires whenever `prob >= 250`, not only
            // when `skip_mb == 0`.
            if self.cost_model == super::api::CostModel::StrictLibwebpParity {
                if self.method >= 3 {
                    // m3-m6 run libwebp's `VP8EncTokenLoop`, which asserts
                    // `proba->use_skip_proba == 0` at entry (`frame_enc.c:816`)
                    // — the flag is never enabled on the token path, so parity
                    // emits all coefficient tokens and no skip flag. (An
                    // earlier revision forced this for EVERY method, claiming
                    // libwebp never enables the flag; that was only ever
                    // verified at sns0/flt0/segs1, where libwebp's skip count
                    // happens to land at use_skip=0 too. Disproven at m0-m2
                    // with SNS/multi-segment configs.) (#38)
                    self.macroblock_no_skip_coeff = None;
                } else if !self.segments_enabled {
                    // m0-m2 with a single effective segment: `StatLoop` bails
                    // out early — `OneStatPass` returns `size_p0 = ΣH +
                    // segment_hdr.size`, which is 0 at RD_OPT_NONE without a
                    // segment header, and `StatLoop` treats that as failure,
                    // returning BEFORE `FinalizeSkipProba` ever runs. The flag
                    // keeps its `VP8DefaultProbas` value of 0. Same upstream
                    // quirk `compute_updated_probabilities` reproduces for the
                    // coefficient probas (dumped via SKIPDBG: at q5/m1/sns0/
                    // segs1 libwebp counts nb_skip=53 in the stats pass but
                    // never finalizes, shipping use_skip=0). (#38)
                    self.macroblock_no_skip_coeff = None;
                } else {
                    // m0-m2 run plain `VP8EncLoop`, whose StatLoop DOES
                    // finalize the skip probability (`frame_enc.c:679`):
                    //   - `nb_skip` is counted in `OneStatPass` over the stats
                    //     subset only — m0 `fast_probe` shortens the pass to
                    //     `total>200 ? total>>2 : 50` MBs; m1/m2 cover the
                    //     whole frame (`frame_enc.c:629,644-650`).
                    //   - `FinalizeSkipProba` then divides by the FULL frame:
                    //     `skip_proba = (total - nb) * 255 / total` — truncated
                    //     integer division, NO clamp (`CalcSkipProba`,
                    //     `frame_enc.c:113`), and
                    //     `use_skip_proba = (skip_proba < 250)`.
                    // The stats-pass skip decisions equal the emission-pass
                    // ones at RD_OPT_NONE (`RefineUsingDistortion` uses fixed
                    // mode costs, never the adapted level costs), so zen's
                    // per-MB skip count over the same subset reproduces
                    // `nb_skip` exactly. (#38)
                    let total = u64::from(total_mb);
                    let nb_skip = u64::from(self.fast_probe_skip_count.unwrap_or(skip_mb));
                    let skip_proba =
                        ((total - nb_skip) * 255).checked_div(total).unwrap_or(255) as u8;
                    const SKIP_PROBA_THRESHOLD: u8 = 250; // frame_enc.c:111
                    self.macroblock_no_skip_coeff = if skip_proba < SKIP_PROBA_THRESHOLD {
                        Some(skip_proba)
                    } else {
                        None
                    };
                }
            } else if total_mb > 0 {
                let non_skip_mb = total_mb - skip_mb;
                let prob = ((255 * non_skip_mb + total_mb / 2) / total_mb).min(255) as u8;
                const SKIP_PROBA_THRESHOLD: u8 = 250;
                if prob >= SKIP_PROBA_THRESHOLD {
                    self.macroblock_no_skip_coeff = None;
                } else {
                    self.macroblock_no_skip_coeff = Some(prob.clamp(1, 254));
                }
            }

            // Finalize probabilities from this pass (used by next pass or final emission)
            self.compute_updated_probabilities();

            // Save stats from final pass
            final_sse_y = sse_y;
            final_sse_u = sse_u;
            final_sse_v = sse_v;
            final_block_count_i4 = block_count_i4;
            final_block_count_i16 = block_count_i16;
            final_skip_mb = skip_mb;
        }

        // ===== FINALIZE: write bitstream =====

        // Bump per-segment loop-filter strength based on observed edge
        // magnitudes (port of libwebp's `VP8AdjustFilterStrength`, called
        // from `PostLoopFinalize` in `frame_enc.c:730`). Must run before
        // the header writer reads `frame.filter_level` and per-segment
        // `loopfilter_level` deltas. #34
        self.adjust_filter_strength();

        // Write compressed frame header (includes probability updates)
        self.encode_compressed_frame_header();

        // Write macroblock headers from stored info.
        // Take the vec out to avoid borrow conflict with self.write_macroblock_header.
        let stored_mb_info = mem::take(&mut self.stored_mb_info);

        // Reset b-pred tracking state for header writing
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        let mb_w = usize::from(self.macroblock_width);
        for (idx, mb_info) in stored_mb_info.iter().enumerate() {
            let mbx = idx % mb_w;
            if mbx == 0 {
                self.left_b_pred = [IntraMode::default(); 4];
            }
            self.write_macroblock_header(mb_info, mbx);
        }

        // Emit tokens to partition using final probabilities.
        // When the per-MB skip flag is in use (`macroblock_no_skip_coeff = Some`),
        // skip the recorded EOB tokens for MBs whose `coeffs_skipped=true` flag was
        // already written into the header — the decoder reads no residuals for them.
        // When the gate fires (`= None`), every MB's tokens are emitted unconditionally,
        // matching libwebp's `use_skip_proba=0` path. (#25)
        let final_probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
        let token_buf = self.token_buffer.take().unwrap();
        if self.macroblock_no_skip_coeff.is_some() {
            // Build skip mask from stored MB info (already moved out into local `stored_mb_info`).
            let skip_mask: Vec<bool> = stored_mb_info.iter().map(|m| m.coeffs_skipped).collect();
            token_buf.emit_tokens_filtered(&mut self.partitions[0], final_probs, &skip_mask);
        } else {
            token_buf.emit_tokens(&mut self.partitions[0], final_probs);
        }

        // Assemble output
        let compressed_header_encoder = mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        // VP8 frame tag encodes partition 0 size in 19 bits (max 524,287 bytes).
        // Exceeding this produces a corrupt bitstream that decoders reject.
        const VP8_MAX_PARTITION0_SIZE: u32 = (1 << 19) - 1;
        let partition0_size = compressed_header_bytes.len() as u32;
        if partition0_size > VP8_MAX_PARTITION0_SIZE {
            return Err(at!(EncodeError::Partition0Overflow {
                size: partition0_size,
                max: VP8_MAX_PARTITION0_SIZE,
            }));
        }

        self.write_uncompressed_frame_header(partition0_size);

        self.writer.write_all(&compressed_header_bytes);

        self.write_partitions();

        // VP8 bitstream even-length padding.
        //
        // libwebp pads the whole VP8 bitstream to an even length and counts
        // the pad byte INSIDE the VP8 chunk size (`syntax_enc.c`
        // `VP8EncWrite`: `pad = vp8_size & 1; vp8_size += pad;`, then
        // `PutVP8Header(pic, vp8_size)` writes the padded, even size). The
        // pad byte therefore lives inside the `VP8 ` chunk and the chunk needs
        // no separate RIFF alignment byte.
        //
        // zenwebp's default emits the true (possibly odd) bitstream length as
        // the chunk size and lets `write_chunk` add the spec-standard RIFF
        // padding byte AFTER the chunk (not counted in the chunk size). Both
        // are valid WebP and decode identically, but they differ in the
        // `VP8 ` chunk-size field (and pad-byte position) whenever the stream
        // is odd-length — which blocks byte-identity with libwebp on ~half of
        // all inputs. Under `StrictLibwebpParity`, append the pad here so it
        // lands inside the chunk exactly as libwebp does; `write_chunk` then
        // sees an even payload and adds no further padding.
        if self.cost_model == super::api::CostModel::StrictLibwebpParity
            && self.writer.len() % 2 == 1
        {
            self.writer.push(0);
        }

        // Clean up
        self.stored_mb_info.clear();

        // Build encoding statistics
        let num_pixels_y =
            u64::from(self.macroblock_width) * 16 * u64::from(self.macroblock_height) * 16;
        let num_pixels_uv =
            u64::from(self.macroblock_width) * 8 * u64::from(self.macroblock_height) * 8;

        let psnr_y = sse_to_psnr(final_sse_y, num_pixels_y);
        let psnr_u = sse_to_psnr(final_sse_u, num_pixels_uv);
        let psnr_v = sse_to_psnr(final_sse_v, num_pixels_uv);
        let total_sse = final_sse_y + final_sse_u + final_sse_v;
        let total_pixels = num_pixels_y + 2 * num_pixels_uv;
        let psnr_all = sse_to_psnr(total_sse, total_pixels);

        let mut stats = super::api::EncodeStats {
            psnr: [psnr_y, psnr_u, psnr_v, psnr_all, 0.0],
            block_count_i4: final_block_count_i4,
            block_count_i16: final_block_count_i16,
            block_count_skip: final_skip_mb,
            ..Default::default()
        };

        // Fill segment info
        for (i, segment) in self.segments.iter().enumerate().take(4) {
            stats.segment_quant[i] = segment.quant_index;
            stats.segment_level[i] = self.frame.filter_level;
        }

        progress
            .on_progress(100)
            .map_err(|e| at!(EncodeError::from(e)))?;

        Ok(stats)
    }

    /// Encode a single macroblock row.
    ///
    /// Resets per-row left state, walks every `mbx` in the row, and forwards
    /// per-MB work into [`Self::encode_macroblock`]. Cancellation checks fire
    /// every 16 MBs via `stop.check()` (driven from `row_state.total_mb`).
    /// Mutates `row_state` and all per-row encoder state in `self`.
    #[allow(clippy::too_many_arguments)]
    fn encode_mb_row(
        &mut self,
        mby: u16,
        y_stride: usize,
        uv_stride: usize,
        num_passes: usize,
        max_count: i32,
        row_state: &mut MbRowState,
        stop: &dyn enough::Stop,
    ) -> super::api::EncodeResult<()> {
        // Reset left state for start of row
        self.left_complexity = Complexity::default();
        self.left_b_pred = [IntraMode::default(); 4];
        self.left_derr = [[0; 2]; 2]; // reset chroma error diffusion for row start
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        for mbx in 0..self.macroblock_width {
            // Check for cancellation every 16 macroblocks
            if row_state.total_mb & 15 == 0 {
                stop.check().map_err(|e| at!(EncodeError::from(e)))?;
            }

            // Mid-stream probability refresh (like libwebp's VP8EncTokenLoop).
            //
            // We refresh `updated_probs` periodically so the eventual emission
            // pass picks up evolving statistics, but we deliberately do NOT
            // rebuild `self.level_costs` at the same time. Empirically:
            // refreshing probs alone helps slightly (1.0111x → 1.0101x);
            // also rebuilding level_costs mid-row hurts (1.0101x → 1.0114x).
            //
            // Effect on this pass is therefore weak: cost-driven mode
            // selection inside the row continues to use the level_costs
            // computed at pass start. The real per-pass cost refresh happens
            // through the multi-pass loop (see #27 / two-pass at m4 path).
            // This call is mostly bookkeeping so the final probability
            // emission reflects accumulated stats. (#35-#6)
            row_state.refresh_countdown -= 1;
            if row_state.refresh_countdown < 0 {
                let _changed = self.compute_updated_probabilities();
                #[cfg(feature = "mode_debug")]
                if std::env::var("REFRESHDBG").is_ok() {
                    let s = self.proba_stats.stats[3][1][2];
                    let split = |v: u32| (v & 0xffff, (v >> 16) & 0xffff); // (nb, total)
                    eprintln!(
                        "REFRESH at mb({mbx},{mby}) changed={_changed} prob[3][1][2]={:?}",
                        self.updated_probs.as_ref().map(|p| p[3][1][2])
                    );
                    eprintln!(
                        "  stats[3][1][2] (nb,total) per proba idx: {:?}",
                        (0..11).map(|i| split(s[i])).collect::<alloc::vec::Vec<_>>()
                    );
                }
                // libwebp's token loop refreshes the level-cost tables
                // alongside the probabilities every `max_count` MBs
                // (`VP8CalculateLevelCosts` right after `FinalizeTokenProbas`,
                // frame_enc.c:834-836), so mode-selection coefficient costs
                // track the evolving distribution for the rest of the pass.
                // zenwebp's tuned default deliberately skips this (rebuilding
                // `level_costs` mid-row measurably regressed default
                // compression, 1.0101x→1.0114x), leaving cost-based mode
                // selection on the pass-start (default-proba) tables. Under
                // StrictLibwebpParity we match libwebp: rebuild `level_costs`
                // from the just-refreshed image-adapted probabilities so the
                // I16/I4/UV coefficient costs (and thus the mode picks) line up
                // — without this, chroma-UV costs run ~1.5× high and flip the
                // DC/TM choice on later rows (e.g. 382297 m3 mb(4,4)).
                if self.cost_model == super::api::CostModel::StrictLibwebpParity
                    && let Some(updated) = self.updated_probs
                {
                    self.level_costs.mark_dirty();
                    self.level_costs.calculate(&updated);
                }
                row_state.refresh_countdown = max_count;
            }

            self.encode_macroblock(mbx, mby, y_stride, uv_stride, num_passes, row_state);
        }

        Ok(())
    }

    /// Encode a single macroblock: mode select, transform, quantize, record tokens.
    ///
    /// The orchestrator for one (mbx, mby) pair. Selects intra mode via
    /// [`Self::choose_macroblock_info`], runs DCT/transform on luma + chroma,
    /// accumulates per-MB SSE for PSNR, routes through the trellis / non-trellis
    /// quantize+record path, then appends the per-MB info to `self.stored_mb_info`.
    ///
    /// `row_state` carries per-row sums that the caller integrates into per-pass
    /// totals. Side effects on `self`: token buffer, complexity context, segment
    /// edge tracking (`max_edge_per_segment`), and `stored_mb_info` /
    /// `stored_mb_coeffs` vectors.
    fn encode_macroblock(
        &mut self,
        mbx: u16,
        mby: u16,
        y_stride: usize,
        uv_stride: usize,
        num_passes: usize,
        row_state: &mut MbRowState,
    ) {
        let macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

        // Update b_pred context for next macroblock's mode selection
        // This must happen during encoding pass, not just header writing
        let mbx_usize = usize::from(mbx);
        if let Some(bpred) = macroblock_info.luma_bpred {
            // I4 mode: update with per-block modes
            // top_b_pred gets the bottom row (row 3)
            for x in 0..4 {
                self.top_b_pred[mbx_usize * 4 + x] = bpred[3 * 4 + x];
            }
            // left_b_pred gets the rightmost column (column 3)
            for y in 0..4 {
                self.left_b_pred[y] = bpred[y * 4 + 3];
            }
        } else {
            // I16 mode: all context slots get the derived intra mode
            let intra_mode = macroblock_info
                .luma_mode
                .into_intra()
                .unwrap_or(IntraMode::DC);
            for x in 0..4 {
                self.top_b_pred[mbx_usize * 4 + x] = intra_mode;
            }
            for y in 0..4 {
                self.left_b_pred[y] = intra_mode;
            }
        }

        // Transform blocks (updates border state for next macroblock)
        let y_block_data = self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

        let (u_block_data, v_block_data) =
            self.transform_chroma_blocks(mbx.into(), mby.into(), macroblock_info.chroma_mode);

        // Accumulate SSE for PSNR computation (source vs reconstructed)
        row_state.sse_y += u64::from(sse_16x16_luma(
            &self.frame.ybuf,
            y_stride,
            usize::from(mbx),
            usize::from(mby),
            &y_block_data.pred_block,
        ));
        row_state.sse_u += u64::from(sse_8x8_chroma(
            &self.frame.ubuf,
            uv_stride,
            usize::from(mbx),
            usize::from(mby),
            &u_block_data.pred_block,
        ));
        row_state.sse_v += u64::from(sse_8x8_chroma(
            &self.frame.vbuf,
            uv_stride,
            usize::from(mbx),
            usize::from(mby),
            &v_block_data.pred_block,
        ));

        // Count block types
        if macroblock_info.luma_mode == LumaMode::B {
            row_state.block_count_i4 += 1;
        } else {
            row_state.block_count_i16 += 1;
        }

        // Quantize and record tokens.
        //
        // For non-trellis methods (0-4): quantize once into stored
        // coefficients, derive the skip from them, record from stored.
        //
        // For trellis methods (5-6): use integrated path since trellis
        // quantization depends on complexity context updated per-block.
        row_state.total_mb += 1;
        let is_i4 = macroblock_info.luma_mode == LumaMode::B;
        let first_coeff_y1 = if is_i4 { 0usize } else { 1 };

        let mut mb_info = macroblock_info;
        let store_coeffs = num_passes > 1;
        // Segment id resolved here so both trellis/non-trellis paths
        // can update `max_edge_per_segment` (#34).
        let mb_segment_id = macroblock_info.segment_id.unwrap_or(0);

        // Mark the start of this MB's tokens so we can selectively
        // suppress them at emit time when the per-MB skip flag is in
        // play (`macroblock_no_skip_coeff = Some`). When the flag is
        // omitted (`= None`), every MB's tokens are emitted, so the
        // EOB tokens we record below for skipped MBs are exactly what
        // the decoder expects (matches libwebp #25).
        self.token_buffer
            .as_mut()
            .expect("token buffer not initialized")
            .begin_mb();

        if self.do_trellis && self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity
        {
            // libwebp's m5/m6 skip decision is `is_skipped = (rd->nz == 0)`
            // (`VP8Decimate`, quant_enc.c) — the nz of the FINAL trellis
            // quantization, not a separate simple-quant test.
            // The old separate skip test re-quantized with the simple bias,
            // which drops borderline coefficients the trellis keeps (the
            // trellis derives level0 with a NEUTRAL bias plus the sharpen
            // term, then RD-scores keeping vs dropping) — so the two disagree
            // on rare MBs. Traced at 1025469 q20 m5 sns0 mb(14,4): simple
            // quant says all-zero, the trellis keeps a single AC level=1
            // (identical trellis output on both sides), libwebp codes it and
            // zen skipped it. Record the actual levels and derive the skip
            // from them; for an all-zero MB the recorded tokens are the same
            // EOB-at-0 stream the ZERO path records, so agreeing MBs are
            // byte-unchanged. (#38)
            //
            // StoreMaxDelta runs BEFORE the skip test: libwebp stores it at
            // the end of PickBestIntra16, before the skip decision, so a
            // finally-skipped MB still feeds max_edge (see the non-trellis
            // branch below for the same ordering).
            #[cfg(feature = "mode_debug")]
            if std::env::var("SMDBG").is_ok() {
                eprintln!(
                    "ZENSM mb({mbx},{mby}) seg={mb_segment_id} blocky={} cand_d={} min_disto={} y2_124={},{},{}",
                    macroblock_info.intra16_blocky,
                    macroblock_info.intra16_cand_d,
                    self.segments[mb_segment_id].min_disto,
                    macroblock_info.intra16_y2_zz[1],
                    macroblock_info.intra16_y2_zz[2],
                    macroblock_info.intra16_y2_zz[4],
                );
            }
            if self.store_max_edge_active()
                && macroblock_info.intra16_blocky
                && macroblock_info.intra16_cand_d > self.segments[mb_segment_id].min_disto
            {
                let y2 = macroblock_info.intra16_y2_zz;
                self.store_max_delta(mb_segment_id, &y2);
            }
            let stored_coeffs = self.record_residual_tokens_storing(
                &macroblock_info,
                mbx as usize,
                &y_block_data.coeffs,
                &u_block_data.coeffs,
                &v_block_data.coeffs,
                y_block_data.trellis_y1_zigzag.as_ref(),
            );
            // Final recorded levels for one MB, format-matched to the
            // LEVFINAL dump in the instrumented libwebp's VP8Decimate
            // (libwebp--zen38trace). TARGX/TARGY select the MB. (#38)
            #[cfg(feature = "mode_debug")]
            if std::env::var("LEVFINAL").is_ok()
                && std::env::var("TARGX").is_ok_and(|v| v == mbx.to_string())
                && std::env::var("TARGY").is_ok_and(|v| v == mby.to_string())
            {
                eprintln!("LEVFINAL mb({mbx},{mby}) type={}", i32::from(!is_i4));
                eprintln!("  y_dc: {}", fmt_levels(&stored_coeffs.y2_zigzag));
                for (b, blk) in stored_coeffs.y1_zigzag.iter().enumerate() {
                    eprintln!("  y_ac[{b:2}]: {}", fmt_levels(blk));
                }
                for (b, blk) in stored_coeffs.u_zigzag.iter().enumerate() {
                    eprintln!("  uv[{b}]: {}", fmt_levels(blk));
                }
                for (b, blk) in stored_coeffs.v_zigzag.iter().enumerate() {
                    eprintln!("  uv[{}]: {}", b + 4, fmt_levels(blk));
                }
            }
            if stored_coeffs.is_all_zero(is_i4, first_coeff_y1) {
                row_state.skip_mb += 1;
                mb_info.coeffs_skipped = true;
            }
            if store_coeffs {
                self.stored_mb_coeffs.push(stored_coeffs);
            }
        } else if self.do_trellis {
            // Tuned trellis path (m5/m6): record the FINAL (trellis) levels
            // and derive the skip from them, like the parity arm above.
            // Historically the skip was decided by a separate simple-quant
            // test (`check_all_coeffs_zero`, since removed), which disagreed
            // with the trellis on borderline coefficients — the encoder then
            // reconstructed WITH a kept coefficient (and predicted subsequent
            // MBs from it) while signaling a skip, so its reference drifted
            // from the decoder's pixels. Deriving the skip from the recorded
            // levels fixes that mismatch; measured on the 15-image A/B corpus
            // (m4 control ±0): m5 +0.011% size / +0.104 zsim, m6 +0.015% /
            // +0.097 — adopted per
            // `benchmarks/tuned_candidates_2026-07-16.md`.
            let stored_coeffs = self.record_residual_tokens_storing(
                &macroblock_info,
                mbx as usize,
                &y_block_data.coeffs,
                &u_block_data.coeffs,
                &v_block_data.coeffs,
                y_block_data.trellis_y1_zigzag.as_ref(),
            );
            if stored_coeffs.is_all_zero(is_i4, first_coeff_y1) {
                row_state.skip_mb += 1;
                mb_info.coeffs_skipped = true;
            } else {
                // Track edge magnitude for I16 "blocky" MBs (#34). Gate on
                // `D > min_disto` per libwebp `quant_enc.c:1111` (issue #44);
                // flat-region MBs produce small D and are filtered out,
                // preventing the loop filter from over-bumping on synthetic
                // color-block content. (The parity variant — I16 CANDIDATE
                // gating, fired before the skip decision — lives in the
                // StrictLibwebpParity trellis branch above.)
                let blocky = stored_coeffs.is_blocky_i16();
                if self.store_max_edge_active() && !is_i4 && blocky {
                    let d = macroblock_info.intra16_d.unwrap_or(0);
                    if d > self.segments[mb_segment_id].min_disto {
                        self.store_max_delta(mb_segment_id, &stored_coeffs.y2_zigzag);
                    }
                }
            }
            if store_coeffs {
                self.stored_mb_coeffs.push(stored_coeffs);
            }
        } else {
            // Non-trellis path: quantize once, skip-check, record from stored.
            let stored_coeffs = self.quantize_mb_coeffs(
                &macroblock_info,
                &y_block_data.coeffs,
                &u_block_data.coeffs,
                &v_block_data.coeffs,
            );
            let all_zero = stored_coeffs.is_all_zero(is_i4, first_coeff_y1);
            // libwebp records max_edge inside PickBestIntra16, which runs before
            // both the I4 override AND the skip decision — so it must fire here
            // regardless of `is_i4` or `all_zero`. See the trellis branch above.
            // (#38)
            if self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity
                && self.store_max_edge_active()
                && macroblock_info.intra16_blocky
                && macroblock_info.intra16_cand_d > self.segments[mb_segment_id].min_disto
            {
                let y2 = macroblock_info.intra16_y2_zz;
                self.store_max_delta(mb_segment_id, &y2);
            }
            if all_zero {
                row_state.skip_mb += 1;
                mb_info.coeffs_skipped = true;
                // Same EOB-token recording as the trellis path above.
                self.record_from_stored_coeffs(
                    &macroblock_info,
                    mbx as usize,
                    &QuantizedMbCoeffs::ZERO,
                );
            } else {
                // Track edge magnitude for I16 "blocky" MBs (#34).
                // See trellis branch above for the `D > min_disto`
                // rationale (issue #44).
                if self.cost_model != crate::encoder::api::CostModel::StrictLibwebpParity
                    && self.store_max_edge_active()
                    && !is_i4
                    && stored_coeffs.is_blocky_i16()
                {
                    let d = macroblock_info.intra16_d.unwrap_or(0);
                    if d > self.segments[mb_segment_id].min_disto {
                        self.store_max_delta(mb_segment_id, &stored_coeffs.y2_zigzag);
                    }
                }
                self.record_from_stored_coeffs(&macroblock_info, mbx as usize, &stored_coeffs);
            }
            // Only store quantized coefficients when multi-pass needs them
            if store_coeffs {
                self.stored_mb_coeffs.push(stored_coeffs);
            }
        }

        // Freeze the proba-stat histogram at the libwebp fast_probe boundary
        // (method-0 StatLoop subset). `total_mb` is 1-based here (incremented
        // above, before recording), so `== limit` fires right after the
        // `limit`-th MB's residuals were recorded — the same MB count libwebp
        // processes in its stats pass. Later MBs keep accumulating into
        // `proba_stats` (harmless; the snapshot is what emission uses).
        if let Some(limit) = self.fast_probe_stat_limit
            && row_state.total_mb as usize == limit
            && self.fast_probe_snapshot.is_none()
        {
            self.fast_probe_snapshot = Some(self.proba_stats.clone());
            // Freeze the skip count at the same boundary: libwebp's
            // `OneStatPass` accumulates `nb_skip` over exactly these MBs. (#38)
            self.fast_probe_skip_count = Some(row_state.skip_mb);
        }

        // Store macroblock info for header writing
        self.stored_mb_info.push(mb_info);
    }

    /// Merge segments with identical quantizer and filter settings.
    ///
    /// This reduces the number of effective segments when different alpha regions
    /// end up with the same quantization and filter parameters. Reducing segment
    /// count can save bits in the bitstream.
    ///
    /// Ported from libwebp's SimplifySegments.
    #[allow(clippy::needless_range_loop)] // s1 indexes both seg_map and self.segments
    fn simplify_segments(&mut self) {
        // Map from old segment ID to new segment ID
        let mut seg_map = [0u8, 1, 2, 3];
        let num_segments = self.num_segments as usize;
        let mut num_final_segments = 1usize;

        // Check each segment starting from 1 to see if it matches an earlier segment
        for s1 in 1..num_segments {
            let seg1 = &self.segments[s1];
            let mut found = false;

            // Check if we already have a segment with same quant_index and loopfilter_level
            for s2 in 0..num_final_segments {
                let seg2 = &self.segments[s2];
                if seg1.quant_index == seg2.quant_index
                    && seg1.loopfilter_level == seg2.loopfilter_level
                {
                    seg_map[s1] = s2 as u8;
                    found = true;
                    break;
                }
            }

            if !found {
                // This is a new unique segment
                seg_map[s1] = num_final_segments as u8;
                if num_final_segments != s1 {
                    // Move segment data to its new position
                    self.segments[num_final_segments] = self.segments[s1].clone();
                }
                num_final_segments += 1;
            }
        }

        // If we reduced segments, remap the segment map
        if num_final_segments < num_segments {
            for seg_id in &mut self.segment_map {
                *seg_id = seg_map[*seg_id as usize];
            }
            self.num_segments = num_final_segments as u8;

            // Replicate trailing segment infos (cosmetic, required by bitstream syntax)
            for i in num_final_segments..num_segments {
                self.segments[i] = self.segments[num_final_segments - 1].clone();
            }
        }
    }

    /// Analyze image complexity and assign macroblocks to segments.
    ///
    /// This performs a DCT-based analysis pass (ported from libwebp) to:
    /// 1. Compute "alpha" (compressibility) for each macroblock using DCT histogram
    /// 2. Build a histogram of alpha values
    /// 3. Use k-means clustering to assign macroblocks to 4 segments
    /// 4. Configure per-segment quantization based on alpha
    ///
    /// Segments allow different quantization for different image regions:
    /// - Flat areas (high alpha from segment perspective) can use more aggressive quantization
    /// - Textured areas (low alpha) need finer quantization to preserve detail
    ///
    /// Ported from libwebp's VP8EncAnalyze / MBAnalyze.
    fn analyze_and_assign_segments(&mut self, quality: u8) {
        let y_stride = usize::from(self.macroblock_width * 16);
        let uv_stride = usize::from(self.macroblock_width * 8);
        let width = usize::from(self.frame.width);
        let height = usize::from(self.frame.height);

        // Run full DCT-based analysis pass using libwebp-compatible algorithm
        // This tests DC and TM modes for I16 and UV, computes per-MB alpha,
        // and builds the alpha histogram.
        //
        // Gate FastMBAnalyze hint collection: libwebp populates hints whenever
        // method <= 1, but the consumer (`choose_macroblock_info`) discards
        // them when `partition_limit >= 100` because that mode forces I16-only
        // anyway. Skipping the per-MB DC sums when we already know they'll be
        // unused saves ~16 sums per MB on large noisy images that hit the
        // partition_limit retry path.
        // The `partition_limit < 100` clause is a zenwebp-only optimization
        // (plim100 forces I16-only in the tuned default, so hints are dead
        // weight). libwebp's analysis never consults partition_limit, so
        // parity collects hints at every limit. (#38)
        let collect_mode_hints = self.method <= 1
            && (self.partition_limit < 100
                || self.cost_model == super::api::CostModel::StrictLibwebpParity);
        let analysis = analyze_image_with_hint_gate(
            &self.frame.ybuf,
            &self.frame.ubuf,
            &self.frame.vbuf,
            width,
            height,
            y_stride,
            uv_stride,
            self.method,
            self.sns_strength,
            self.cost_model,
            i32::from(quality),
            collect_mode_hints,
        );

        // Auto-detect content type when Preset::Auto is selected.
        // Runs after analyze_image (reuses alpha histogram, nearly free).
        if self.preset == super::api::Preset::Auto {
            let content_type = classify_image_type(
                &self.frame.ybuf,
                width,
                height,
                y_stride,
                &analysis.alpha_histogram,
            );
            let (sns, filter, sharp, segs) = content_type_to_tuning(content_type);
            self.sns_strength = sns;
            self.filter_strength = filter;
            self.filter_sharpness = sharp;
            self.num_segments = segs;
        }

        // Use k-means to assign segments
        // weighted_average is computed from final cluster centers, matching libwebp
        let (centers, alpha_to_segment, mid_alpha) =
            assign_segments_kmeans(&analysis.alpha_histogram, usize::from(self.num_segments));

        // Find min and max of centers for alpha transformation
        // This matches libwebp's SetSegmentAlphas
        let min_center = centers.iter().copied().min().unwrap_or(0) as i32;
        let max_center = centers.iter().copied().max().unwrap_or(255) as i32;
        let range = if max_center == min_center {
            1 // Avoid division by zero
        } else {
            max_center - min_center
        };

        // libwebp's `SetSegmentAlphas` (`analysis_enc.c:92`) only handles the
        // degenerate `min == max` case (`if (max == min) max = min + 1;`) — no
        // additional floor. The previous `MIN_ALPHA_RANGE = 64` floor in
        // zenwebp dampened the SNS modulation by up to 6.4× on flat content
        // (gradients, skies), losing most of the per-segment quantizer spread
        // that makes the larger segments compress better. Removed in #30.
        let effective_range = range; // already >= 1 above

        // Assign segment IDs to macroblocks
        self.segment_map = analysis
            .mb_alphas
            .iter()
            .map(|&alpha| alpha_to_segment[alpha as usize])
            .collect();

        // Store FastMBAnalyze mode hints from the analysis pass. These are only
        // populated at method <= 1 — the encode pass uses them to skip RD mode
        // selection (mirroring libwebp's `RefineUsingDistortion(try_both_modes=0)`).
        self.fast_mb_hints = analysis.mb_mode_hints.unwrap_or_default();
        self.fast_mb_uv_hints = analysis.mb_uv_hints.unwrap_or_default();

        // Smooth segment map (3x3 majority filter) only when the preprocessing
        // smooth_segment_map flag explicitly opts in. libwebp gates this on `config->preprocessing & 1`
        // (`analysis_enc.c:217-218`), default OFF (`config_enc.c:48`); we match.
        // zenwebp previously smoothed unconditionally whenever multi-segment, which
        // could collapse 4 segments into 2 after `simplify_segments` and lose
        // differential-quantization savings (#26).
        if self.num_segments > 1 && self.smooth_segment_map {
            super::cost::smooth_segment_map(
                &mut self.segment_map,
                usize::from(self.macroblock_width),
                usize::from(self.macroblock_height),
            );
        }

        // Configure per-segment quantization using preset's SNS strength
        let sns_strength = self.sns_strength;

        // Compute UV quant deltas from uv_alpha average (from libwebp's VP8SetSegmentParams)
        // uv_alpha is typically ~30 (bad) to ~100 (ok to decimate UV more), centered ~60
        // Constants from libwebp quant_enc.c
        const MID_UV_ALPHA: i32 = 64;
        const MIN_UV_ALPHA: i32 = 30;
        const MAX_UV_ALPHA: i32 = 100;
        const MAX_DQ_UV: i32 = 6;
        const MIN_DQ_UV: i32 = -4;

        // Map uv_alpha to the safe maximal range of MAX/MIN_DQ_UV
        let dq_uv_ac = (analysis.uv_alpha_avg - MID_UV_ALPHA) * (MAX_DQ_UV - MIN_DQ_UV)
            / (MAX_UV_ALPHA - MIN_UV_ALPHA);
        // Rescale by user-defined SNS strength
        let dq_uv_ac = (dq_uv_ac * i32::from(sns_strength) / 100).clamp(MIN_DQ_UV, MAX_DQ_UV);

        // Boost dc-uv-quant based on sns-strength (UV is more reactive to high quants)
        let dq_uv_dc = (-4 * i32::from(sns_strength) / 100).clamp(-15, 15);

        // Write UV quant deltas to the bitstream header so the decoder knows about them.
        // Without this, the encoder quantizes UV with offset step sizes but the decoder
        // dequantizes with base step sizes, causing systematic UV reconstruction errors.
        if dq_uv_dc != 0 {
            self.quantization_indices.uvdc_delta = Some(dq_uv_dc as i8);
        }
        if dq_uv_ac != 0 {
            self.quantization_indices.uvac_delta = Some(dq_uv_ac as i8);
        }

        // libwebp uses segment 0's modulated quantizer + filter as the bitstream
        // base (`enc->base_quant = enc->dqm[0].quant;` at `quant_enc.c:404`), so
        // segment 0's delta is always 0 (saves ~9 bits per non-zero delta in the
        // segment header). zenwebp previously used the unmodulated
        // `quality_to_quant_index(quality)`, making segment 0 always carry a
        // non-zero delta when SNS modulation was active. (#30 C)
        //
        // Pass 1 computes the per-segment (quant, filter) tuples; segment 0's
        // values then become the bitstream base, and pass 2 (the matrix-init
        // loop below) writes deltas relative to that.
        let mut seg_quant_indices = [0u8; 4];
        let mut seg_filters = [0u8; 4];
        for (seg_idx, &center) in centers.iter().enumerate() {
            let center = center as i32;
            let transformed_alpha = (255 * (center - mid_alpha) / effective_range).clamp(-127, 127);
            let beta = (255 * (center - min_center) / effective_range).clamp(0, 255) as u8;
            // Parity needs libwebp's exact libm pow chain — the fast
            // approximation flips the truncated quant index at integer
            // boundaries (synth 33x17 q90: seg1 12 vs 11). (#38)
            let mut seg_quant_index =
                if self.cost_model == super::api::CostModel::StrictLibwebpParity {
                    super::analysis::compute_segment_quant_libm(
                        quality,
                        transformed_alpha,
                        sns_strength,
                    )
                } else {
                    compute_segment_quant(quality, transformed_alpha, sns_strength)
                };
            // target_zensim per-segment correction: apply additive
            // override (clamped) AFTER SNS modulation. `None` (default)
            // leaves the value untouched, so non-target-zensim encodes
            // are bit-identical to before.
            if let Some(deltas) = self.segment_quant_overrides {
                let delta = deltas.get(seg_idx).copied().unwrap_or(0) as i32;
                seg_quant_index = (i32::from(seg_quant_index) + delta).clamp(0, 127) as u8;
            }
            // libwebp's `SetupFilterStrength` reads `filter_hdr.sharpness` at
            // the TOP of the function but only assigns it from the config at
            // the BOTTOM (`quant_enc.c:294`) — so the per-segment strength
            // derivation always sees the PREVIOUS call's sharpness, which at
            // the default single pass is the zero-initialized value. Dumped
            // live (LIBFS, config sharpness=1): `base = qstep` — the identity
            // row 0, not row 1. The header sharpness field and
            // `VP8AdjustFilterStrength` use the real config value. Parity
            // reproduces the stale read (0); the tuned default keeps zen's
            // read-through behavior, deriving strength with the same
            // sharpness the decoder will actually filter with. (#38)
            let strength_sharpness =
                if self.cost_model == super::api::CostModel::StrictLibwebpParity {
                    0
                } else {
                    self.filter_sharpness
                };
            let seg_filter = super::cost::compute_filter_level_with_beta(
                seg_quant_index,
                strength_sharpness,
                self.filter_strength,
                beta,
            );
            seg_quant_indices[seg_idx] = seg_quant_index;
            seg_filters[seg_idx] = seg_filter;
        }

        let base_quant_index_new = seg_quant_indices[0];
        let base_filter_new = seg_filters[0];
        self.quantization_indices.yac_abs = base_quant_index_new;

        // Update the frame-level filter strength to match segment 0's
        // SNS-modulated value. libwebp's `SetupFilterStrength` does this
        // unconditionally (`quant_enc.c:293`):
        //   `enc->filter_hdr.level = enc->dqm[0].fstrength;`
        //
        // Without this, `frame.filter_level` is left at the value
        // `setup_encoding` computed from the *un-modulated* base quant
        // index, which is much lower than segment 0's post-SNS quant.
        // Per-segment `loopfilter_level` deltas are then computed against
        // the wrong base, leaving every segment's absolute filter level
        // ~10–12 below libwebp's. This was previously masked by #34's
        // unfiltered `store_max_delta` bumping the strength back up, but
        // once the proper `D > min_disto` gate (#44) prunes flat-region
        // contributions, the under-filtering became visible as a 5–18
        // zensim regression on smooth gradients at low quality.
        self.frame.filter_level = base_filter_new;

        for seg_idx in 0..centers.len() {
            let seg_quant_index = seg_quant_indices[seg_idx];
            let seg_quant_usize = seg_quant_index as usize;

            // Compute the delta from segment 0's modulated quant.
            let delta = seg_quant_index as i8 - base_quant_index_new as i8;

            // Use the per-segment filter level computed in pass 1; delta is
            // relative to segment 0's filter (the new bitstream base).
            let seg_filter = seg_filters[seg_idx];
            let filter_delta = (seg_filter as i8) - (base_filter_new as i8);

            // Apply UV quant deltas (from libwebp's SetupMatrices)
            // UV DC quant uses dq_uv_dc offset, clamped to [0, 117]
            // UV AC quant uses dq_uv_ac offset, clamped to [0, 127]
            let uv_dc_idx = (seg_quant_usize as i32 + dq_uv_dc).clamp(0, 117) as usize;
            let uv_ac_idx = (seg_quant_usize as i32 + dq_uv_ac).clamp(0, 127) as usize;

            let mut segment = Segment {
                ydc: DC_QUANT[seg_quant_usize],
                yac: AC_QUANT[seg_quant_usize],
                y2dc: DC_QUANT[seg_quant_usize] * 2,
                // Y2 AC uses libwebp's dedicated `kAcTable2` lookup (`quant_enc.c:236`,
                // verified byte-identical to our `VP8_AC_TABLE2`). Previously we
                // synthesized the value as `kAcTable * 155/100` which deviates by up
                // to ~10% at mid-quantizer. Decoder side updated to match (#24).
                y2ac: VP8_AC_TABLE2[seg_quant_usize] as i16,
                uvdc: DC_QUANT[uv_dc_idx],
                uvac: AC_QUANT[uv_ac_idx],
                quantizer_level: delta,
                loopfilter_level: filter_delta,
                quant_index: seg_quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength, self.method, self.cost_model);
            self.segments[seg_idx] = segment;
        }

        // The VP8 segment header always carries 4 quant + 4 filter slots.
        // `SimplifySegments` (below) fills the `[num_final..config]` slots with the
        // last surviving segment (`dqm[num_final-1]`), matching libwebp. But the
        // slots BEYOND the configured count — `[config..4]` — are never touched by
        // either the k-means (0..config) or SimplifySegments (..config); libwebp
        // leaves them at their base/segment-0 values, so `PutSegmentHeader` emits
        // seg0's quant+filter for them (verified: a 2-segment encode has
        // `dqm[2] == dqm[3] == dqm[0]`). zenwebp left them holding segment-1's
        // values, diverging the segment header on every segs<4 encode. Match
        // libwebp under parity by pointing `[config..4]` at segment 0 (delta 0 →
        // absolute = base). Use the pre-simplify count so we don't clobber
        // SimplifySegments' correct `[num_final..config]` replication.
        let config_num_segments = self.num_segments as usize;

        // Simplify segments by merging those with identical quant and filter settings
        // This can reduce the number of effective segments and save bits in the bitstream
        if self.num_segments > 1 {
            self.simplify_segments();
        }

        if self.cost_model == super::api::CostModel::StrictLibwebpParity {
            for i in config_num_segments..self.segments.len() {
                self.segments[i] = self.segments[0].clone();
            }
        }

        // Compute segment tree probabilities from actual distribution
        // This matches libwebp's SetSegmentProbas
        let mut seg_counts = [0u32; 4];
        for &seg_id in &self.segment_map {
            seg_counts[seg_id as usize] += 1;
        }

        // Segment tree uses 3 probabilities for binary splits:
        // prob[0] = P(segment < 2), prob[1] = P(segment == 0 | segment < 2)
        // prob[2] = P(segment == 2 | segment >= 2)
        #[allow(clippy::manual_checked_ops)]
        let get_proba = |a: u32, b: u32| -> u8 {
            let total = a + b;
            if total == 0 {
                255 // default
            } else {
                ((255 * a + total / 2) / total) as u8
            }
        };

        self.segment_tree_probs[0] =
            get_proba(seg_counts[0] + seg_counts[1], seg_counts[2] + seg_counts[3]);
        self.segment_tree_probs[1] = get_proba(seg_counts[0], seg_counts[1]);
        self.segment_tree_probs[2] = get_proba(seg_counts[2], seg_counts[3]);

        // Only enable update_map if probabilities differ from default (255)
        let should_update_map = self.segment_tree_probs[0] != 255
            || self.segment_tree_probs[1] != 255
            || self.segment_tree_probs[2] != 255;

        // libwebp writes `segmentation_enabled = (num_segments > 1)`
        // (`PutSegmentHeader`, after `SimplifySegments` collapses equivalent
        // segments — `quant_enc.c`). When the segments collapse to 1 (uniform
        // quant + filter, e.g. sns=0 where the SNS quantizer spread is 0),
        // libwebp disables segmentation entirely. zenwebp set `segments_enabled`
        // unconditionally, writing a full 4-segment header for uniform segments —
        // which made the whole sns0/segs>1 config diverge (segmentation on where
        // libwebp turned it off). Match libwebp under parity; the tuned default
        // keeps its prior behavior pending a measured adoption.
        self.segments_enabled = if self.cost_model == super::api::CostModel::StrictLibwebpParity {
            self.num_segments > 1
        } else {
            true
        };
        self.segments_update_map = should_update_map && self.segments_enabled;

        // Reset borders for actual encoding pass
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];
    }

    // sets up the encoding of the encoder by setting all the encoder params based on the width and height
    fn setup_encoding(
        &mut self,
        lossy_quality: u8,
        width: u16,
        height: u16,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
    ) {
        // choosing the quantization quality based on the quality passed in
        if lossy_quality > 100 {
            panic!("lossy quality must be between 0 and 100");
        }

        // Use libwebp-style quality curve to match expected behavior at Q75
        // This emulates jpeg-like behavior where Q75 is "good quality".
        // libwebp's base quant TRUNCATES `127*(1-c)` (`VP8SetSegmentParams`);
        // `quality_to_quant_index` rounds, which diverges by +1 at q10/30/50/80
        // (frac >= 0.5). For segs>1 this value is overwritten by the truncating
        // `compute_segment_quant`, so only segs1 is affected — under parity use
        // the truncating form so segs1 is byte-exact away from q75. The tuned
        // default keeps the (rounded) value pending a measured adoption.
        let quant_index: u8 = if self.cost_model == super::api::CostModel::StrictLibwebpParity {
            super::fast_math::quality_to_quant_index_trunc(lossy_quality)
        } else {
            quality_to_quant_index(lossy_quality)
        };
        let quant_index_usize: usize = quant_index as usize;

        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        self.macroblock_width = mb_width;
        self.macroblock_height = mb_height;

        // Compute optimal filter level based on quantization and preset tuning
        // params. Parity derives strength with sharpness 0 — libwebp's
        // `SetupFilterStrength` reads `filter_hdr.sharpness` before assigning
        // it from the config, so its derivation always sees the previous
        // call's value (0 at the default single pass); see the per-segment
        // loop in `encode_image` for the full story. (#38)
        let strength_sharpness = if self.cost_model == super::api::CostModel::StrictLibwebpParity {
            0
        } else {
            self.filter_sharpness
        };
        let filter_level = super::cost::compute_filter_level(
            quant_index,
            strength_sharpness,
            self.filter_strength,
        );

        self.frame = Frame {
            width,
            height,

            ybuf: y_buf,
            ubuf: u_buf,
            vbuf: v_buf,

            // VP8 profile, deduced from config like libwebp (webp_enc.c):
            // normal loop filter -> 0, filter disabled -> 2. (Profile 1 is
            // the simple-filter variant, which we never emit.)
            version: if self.filter_strength > 0 { 0 } else { 2 },

            for_display: true,
            pixel_type: 0,

            filter_type: false,
            filter_level,
            sharpness_level: self.filter_sharpness,
        };

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        self.token_probs = COEFF_PROBS;

        // Enable skip mode for zero macroblocks
        // The probability is P(not skip) - 200 means ~78% expected to have coefficients
        self.macroblock_no_skip_coeff = Some(200);

        let quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };
        self.quantization_indices = quantization_indices;

        // Initialize all 4 segments with base quantization first
        // This provides fallback values before segment analysis. At
        // `num_segments == 1` this init IS the final segment (the
        // segment-analysis reconfiguration never runs), so it must apply the
        // same index clips libwebp's `SetupMatrices` does — in particular the
        // UV DC clip to 117 (`m->uv.q[0] = kDcTable[clip(q + dq_uv_dc, 0,
        // 117)]`, quant_enc.c:233). Only quality 0 reaches indices above 117,
        // which is how the unclipped lookup survived every q1-q100 sweep:
        // at q0/segs1 zen quantized UV DC with step 157 (index 127) while
        // libwebp — and both DECODERS — use 132 (index 117), diverging every
        // chroma DC. (#38)
        for seg_idx in 0..4 {
            let mut segment = Segment {
                ydc: DC_QUANT[quant_index_usize],
                yac: AC_QUANT[quant_index_usize],
                y2dc: DC_QUANT[quant_index_usize] * 2,
                y2ac: VP8_AC_TABLE2[quant_index_usize] as i16,
                uvdc: DC_QUANT[quant_index_usize.min(117)],
                uvac: AC_QUANT[quant_index_usize],
                quantizer_level: 0, // No delta for base segment
                quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength, self.method, self.cost_model);
            self.segments[seg_idx] = segment;
        }

        // Segment-based quantization using DCT histogram analysis (ported from libwebp).
        // This allows different quantization for different image regions:
        // - Flat areas get more aggressive quantization
        // - Textured areas get finer quantization
        //
        // Only enable for images large enough to benefit (overhead vs gain tradeoff).
        // libwebp uses segments for images with method > 0 and multiple segments configured.
        // libwebp gates segmentation on `config->emulate_jpeg_size || num_segments > 1
        // || method <= 1` (`analysis_enc.c:434-436`) — no min-MB threshold. zenwebp
        // previously skipped segmentation entirely below 256 MBs (~256x256 images),
        // losing alpha-driven quantizer differentiation on icons/thumbnails. Removed
        // the `>= 256` gate in #30.
        let use_segments = self.num_segments > 1;

        if use_segments {
            // DCT-based segment analysis and assignment.
            // For Preset::Auto, this also runs content detection and may override
            // sns_strength, filter_strength, filter_sharpness, and num_segments.
            self.analyze_and_assign_segments(lossy_quality);

            // If Auto detection changed filter params, recompute frame filter level
            if self.preset == super::api::Preset::Auto {
                let new_filter = super::cost::compute_filter_level(
                    quant_index,
                    self.filter_sharpness,
                    self.filter_strength,
                );
                self.frame.filter_level = new_filter;
                self.frame.sharpness_level = self.filter_sharpness;
            }
        } else {
            // Single segment: no per-region quantizer differentiation.
            self.segments_enabled = false;
            self.segments_update_map = false;
            self.segment_map = Vec::new();

            // libwebp's analysis gate is `do_segments = ... || method <= 1`
            // (analysis_enc.c:434): FastMBAnalyze — which drives m0/m1 mode
            // selection — runs even with one segment. `analyze_and_assign_
            // segments` (the multi-segment path above) is where hints were
            // populated, so a single-segment m0/m1 encode was left with an
            // empty `fast_mb_hints` and collapsed every MB to I16-DC. Run the
            // hint-collecting analysis here too (segments stay disabled).
            let mut uv_alpha_avg: i32 = 0; // libwebp's no-analysis default (analysis_enc.c:372)
            if self.method <= 1
                && (self.partition_limit < 100
                    || self.cost_model == super::api::CostModel::StrictLibwebpParity)
            {
                let y_stride = usize::from(self.macroblock_width * 16);
                let uv_stride = usize::from(self.macroblock_width * 8);
                let width = usize::from(self.frame.width);
                let height = usize::from(self.frame.height);
                let analysis = analyze_image_with_hint_gate(
                    &self.frame.ybuf,
                    &self.frame.ubuf,
                    &self.frame.vbuf,
                    width,
                    height,
                    y_stride,
                    uv_stride,
                    self.method,
                    self.sns_strength,
                    self.cost_model,
                    i32::from(lossy_quality),
                    true,
                );
                self.fast_mb_hints = analysis.mb_mode_hints.unwrap_or_default();
                self.fast_mb_uv_hints = analysis.mb_uv_hints.unwrap_or_default();
                uv_alpha_avg = analysis.uv_alpha_avg;
            }

            // libwebp applies the uv_alpha-derived UV quant deltas at EVERY
            // segment count — `VP8SetSegmentParams` runs unconditionally, and
            // when its analysis pass doesn't (method >= 2 with one segment)
            // `enc->uv_alpha` keeps the 0 default (`analysis_enc.c:372`),
            // still yielding non-zero deltas whenever sns_strength > 0
            // (traced at 382297 q50 m4 sns80 segs1: lib writes uvac_delta=-4,
            // zen wrote none, flipping 24% of UV picks). zenwebp's tuned
            // single-segment path skips the deltas: MEASURED AND REJECTED
            // for adoption (A/B 2026-07-16, segs1 sns50, 180 cells: +0.35
            // zsim for +2.8% bytes overall, +7.3% bytes at q90 — on/below
            // the tuned RD curve, and at m2+ the delta comes from libwebp's
            // unset uv_alpha=0 default, not content). Parity computes them
            // here. (#38)
            if self.cost_model == super::api::CostModel::StrictLibwebpParity {
                const MID_UV_ALPHA: i32 = 64;
                const MIN_UV_ALPHA: i32 = 30;
                const MAX_UV_ALPHA: i32 = 100;
                const MAX_DQ_UV: i32 = 6;
                const MIN_DQ_UV: i32 = -4;
                let sns = i32::from(self.sns_strength);
                let dq_uv_ac = (uv_alpha_avg - MID_UV_ALPHA) * (MAX_DQ_UV - MIN_DQ_UV)
                    / (MAX_UV_ALPHA - MIN_UV_ALPHA);
                let dq_uv_ac = (dq_uv_ac * sns / 100).clamp(MIN_DQ_UV, MAX_DQ_UV);
                let dq_uv_dc = (-4 * sns / 100).clamp(-15, 15);
                if dq_uv_dc != 0 {
                    self.quantization_indices.uvdc_delta = Some(dq_uv_dc as i8);
                }
                if dq_uv_ac != 0 {
                    self.quantization_indices.uvac_delta = Some(dq_uv_ac as i8);
                }
                let qi = i32::from(quant_index);
                let uv_dc_idx = (qi + dq_uv_dc).clamp(0, 117) as usize;
                let uv_ac_idx = (qi + dq_uv_ac).clamp(0, 127) as usize;
                for seg in self.segments.iter_mut() {
                    seg.uvdc = DC_QUANT[uv_dc_idx];
                    seg.uvac = AC_QUANT[uv_ac_idx];
                }
                let (sns_strength, method, cost_model) =
                    (self.sns_strength, self.method, self.cost_model);
                for seg in self.segments.iter_mut() {
                    seg.init_matrices(sns_strength, method, cost_model);
                }
            }
        }

        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        self.top_border_y = vec![127u8; usize::from(self.macroblock_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(self.macroblock_width) * 8];
        self.top_border_v = vec![127u8; usize::from(self.macroblock_width) * 8];

        // Initialize error diffusion arrays (one entry per macroblock column)
        // [channel][position], channels are U=0, V=1
        self.top_derr = vec![[[0i8; 2]; 2]; usize::from(self.macroblock_width)];
        self.left_derr = [[0; 2]; 2];
    }
}

/// Convert SSE to PSNR in dB. Returns 99.0 for perfect reconstruction (SSE=0).
fn sse_to_psnr(sse: u64, num_pixels: u64) -> f32 {
    if sse == 0 || num_pixels == 0 {
        99.0
    } else {
        let mse = sse as f64 / num_pixels as f64;
        (10.0 * libm::log10(255.0 * 255.0 / mse)) as f32
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossy(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    let width_u16: u16 = width
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;
    let height_u16: u16 = height
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;

    // When the `analyzer` feature is on and the user picked Auto, run
    // the shared zenanalyze scanner up front and bake the resolved
    // tuning into params. This lets the inner encoder skip its homegrown
    // alpha-histogram heuristic. With `analyzer` off, this is a no-op
    // and the inner heuristic runs unchanged.
    let resolved_params =
        resolve_auto_preset_via_analyzer(params, data, width, height, stride, color);
    let params_eff = resolved_params.as_ref().unwrap_or(params);

    // Quality search: if target_size or target_psnr is set, iterate quality to converge
    if params_eff.target_size > 0 {
        Ok(encode_with_quality_search(
            writer, data, width_u16, height_u16, stride, color, params_eff, stop, progress,
        )?)
    } else if params_eff.target_psnr > 0.0 {
        Ok(encode_with_psnr_search(
            writer, data, width_u16, height_u16, stride, color, params_eff, stop, progress,
        )?)
    } else {
        // Single encoding at specified quality, with automatic partition limit retry
        encode_with_partition_retry(
            writer, data, width_u16, height_u16, stride, color, params_eff, stop, progress,
        )
    }
}

/// When the `analyzer` feature is on and `params.preset == Auto`,
/// run zenanalyze on the source RGB(A)8 buffer, derive the bucket and
/// tuning, and return a clone of `params` with `(sns_strength,
/// filter_strength, filter_sharpness, num_segments)` overridden and
/// `preset` set to a concrete (non-Auto) value so the inner encoder's
/// own classifier short-circuits. Returns `None` when no override is
/// needed (analyzer off, preset != Auto, layout unsupported, etc.).
#[cfg(feature = "analyzer")]
fn resolve_auto_preset_via_analyzer(
    params: &super::api::EncoderParams,
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    color: PixelLayout,
) -> Option<super::api::EncoderParams> {
    use crate::encoder::analysis::content_type_to_tuning;
    use crate::encoder::analysis::{
        classifier::classify_image_type_rgb8, classifier::rgba8_to_rgb8,
    };
    if params.preset != super::api::Preset::Auto {
        return None;
    }
    // Only apply when the user did NOT explicitly override these knobs;
    // their config_to_params lowering doesn't preserve "explicit vs
    // preset-default", so this is best-effort: we always set the four
    // tuning fields when running on an Auto preset.
    //
    // The bucket-table path only needs the `ImageContentType`, so use the
    // plain classifier entry (the `_diag` variant's raw zenanalyze signals
    // were only consumed by the removed feature-gated v0.1 MLP picker).
    let bucket = match color {
        PixelLayout::Rgb8 if stride == width as usize => {
            let n = (width as usize) * (height as usize) * 3;
            if data.len() < n {
                return None;
            }
            classify_image_type_rgb8(&data[..n], width, height)
        }
        PixelLayout::Rgba8 if stride == width as usize => {
            let n = (width as usize) * (height as usize) * 4;
            if data.len() < n {
                return None;
            }
            let rgb = rgba8_to_rgb8(&data[..n]);
            classify_image_type_rgb8(&rgb, width, height)
        }
        _ => return None,
    };

    // Bucket-table tuning: map the `ImageContentType` to the
    // (sns, filter, sharpness, segments) tuple. (The former feature-gated
    // v0.1 MLP picker override that sat here was removed — it was a
    // pre-A research spike with no A-era retrain.)
    let (sns, filter, sharp, segs) = content_type_to_tuning(bucket);

    let mut p = params.clone();
    p.sns_strength = sns;
    p.filter_strength = filter;
    p.filter_sharpness = sharp;
    p.num_segments = segs;
    // Map the bucket back to a concrete preset so the inner classifier
    // path short-circuits. Photo bucket -> Preset::Photo, Drawing/Text
    // -> Default (Drawing/Text presets compress worse — see the inner
    // classifier's notes), Icon -> Preset::Icon.
    p.preset = match bucket {
        crate::encoder::analysis::ImageContentType::Photo => super::api::Preset::Photo,
        crate::encoder::analysis::ImageContentType::Drawing
        | crate::encoder::analysis::ImageContentType::Text => super::api::Preset::Default,
        crate::encoder::analysis::ImageContentType::Icon => super::api::Preset::Icon,
    };
    Some(p)
}

#[cfg(not(feature = "analyzer"))]
fn resolve_auto_preset_via_analyzer(
    _params: &super::api::EncoderParams,
    _data: &[u8],
    _width: u32,
    _height: u32,
    _stride: usize,
    _color: PixelLayout,
) -> Option<super::api::EncoderParams> {
    None
}

/// Encode a single frame, automatically retrying with increasing partition_limit
/// if partition 0 overflows and the user didn't set an explicit limit.
#[allow(clippy::too_many_arguments)]
fn encode_with_partition_retry(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    // If the user set an explicit partition_limit, use it as-is (no retry).
    if params.partition_limit.is_some() {
        let mut vp8_encoder = Vp8Encoder::new(writer);
        return vp8_encoder
            .encode_image(data, color, width, height, stride, params, stop, progress);
    }

    // Automatic mode: try encoding, retry with increasing partition_limit on overflow.
    // Escalation steps chosen to quickly find a working limit without too many retries.
    const RETRY_LIMITS: [u8; 4] = [0, 40, 70, 100];

    let mut last_overflow = None;
    for &limit in &RETRY_LIMITS {
        stop.check().map_err(|e| at!(EncodeError::from(e)))?;

        let mut trial_buf = Vec::new();
        let mut trial_params = params.clone();
        trial_params.partition_limit = Some(limit);

        let mut vp8_encoder = Vp8Encoder::new(&mut trial_buf);
        match vp8_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        ) {
            Ok(stats) => {
                writer.extend_from_slice(&trial_buf);
                return Ok(stats);
            }
            Err(e)
                if matches!(e.error(), EncodeError::Partition0Overflow { .. }) && limit < 100 =>
            {
                last_overflow = Some(e);
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    // All retry limits exhausted — return the last overflow error (trace preserved).
    Err(last_overflow.unwrap_or_else(|| {
        at!(EncodeError::Partition0Overflow {
            size: 0,
            max: (1 << 19) - 1,
        })
    }))
}

/// Encode a single lossy frame and return the resulting bytes alongside
/// the encoder's per-MB segment_map and grid dimensions. Used by the
/// `target-zensim` closed-loop iteration to make per-segment correction
/// decisions against the encoder's real k-means assignment.
///
/// Mirrors [`encode_frame_lossy`] (with partition-retry) but routes through
/// a path that retains access to the [`Vp8Encoder`] after `encode_image` so
/// the segment_map can be taken out. The on-wire bytes are byte-identical
/// to `encode_frame_lossy` for the same inputs.
///
/// Caller is responsible for skipping target_size / target_psnr / target_zensim
/// recursion on the params (the iteration loop already does this).
#[cfg(feature = "target-zensim")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossy_with_diagnostics(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<(super::api::EncodeStats, super::api::EncodeDiagnostics)> {
    let width_u16: u16 = width
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;
    let height_u16: u16 = height
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;

    // Pre-resolve Auto preset via zenanalyze when the `analyzer`
    // feature is on. See `resolve_auto_preset_via_analyzer`.
    let resolved_params =
        resolve_auto_preset_via_analyzer(params, data, width, height, stride, color);
    let params = resolved_params.as_ref().unwrap_or(params);

    // Same partition-retry semantics as `encode_with_partition_retry`,
    // but we hold onto the Vp8Encoder so we can take its segment_map out
    // after a successful encode.
    if params.partition_limit.is_some() {
        let mut vp8_encoder = Vp8Encoder::new(writer);
        let stats = vp8_encoder.encode_image(
            data, color, width_u16, height_u16, stride, params, stop, progress,
        )?;
        let diag = vp8_encoder.into_diagnostics();
        return Ok((stats, diag));
    }

    const RETRY_LIMITS: [u8; 4] = [0, 40, 70, 100];
    let mut last_overflow = None;
    for &limit in &RETRY_LIMITS {
        stop.check().map_err(|e| at!(EncodeError::from(e)))?;

        let mut trial_buf = Vec::new();
        let mut trial_params = params.clone();
        trial_params.partition_limit = Some(limit);

        let mut vp8_encoder = Vp8Encoder::new(&mut trial_buf);
        match vp8_encoder.encode_image(
            data,
            color,
            width_u16,
            height_u16,
            stride,
            &trial_params,
            stop,
            progress,
        ) {
            Ok(stats) => {
                let diag = vp8_encoder.into_diagnostics();
                writer.extend_from_slice(&trial_buf);
                return Ok((stats, diag));
            }
            Err(e)
                if matches!(e.error(), EncodeError::Partition0Overflow { .. }) && limit < 100 =>
            {
                last_overflow = Some(e);
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    // All retry limits exhausted — return the last overflow error (trace preserved).
    Err(last_overflow.unwrap_or_else(|| {
        at!(EncodeError::Partition0Overflow {
            size: 0,
            max: (1 << 19) - 1,
        })
    }))
}

/// Encode with quality search to meet target file size.
/// Uses secant method to converge on target size within DQ_LIMIT threshold.
#[allow(clippy::too_many_arguments)]
fn encode_with_quality_search(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    // Initialize quality search state
    // qmin=1, qmax=100 (full range) - libwebp uses config->qmin/qmax
    let mut pass_stats = PassStats::new_for_size(params.target_size, params.lossy_quality, 1, 100);

    // Max iterations (matches libwebp's config->pass, default 6 for target_size search)
    let max_passes = (params.method + 3).max(6) as usize;
    let mut best_output: Option<Vec<u8>> = None;
    let mut best_enc_stats = super::api::EncodeStats::default();
    let mut best_diff = f64::MAX;

    for pass in 0..max_passes {
        stop.check().map_err(|e| at!(EncodeError::from(e)))?;

        // Create temporary buffer for this trial encoding
        let mut trial_buffer = Vec::new();
        let mut trial_encoder = Vp8Encoder::new(&mut trial_buffer);

        // Create params with adjusted quality
        let mut trial_params = params.clone();
        trial_params.lossy_quality = libm::roundf(pass_stats.q).clamp(0.0, 100.0) as u8;

        // Encode to trial buffer
        let enc_stats = trial_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        )?;

        // Update stats with resulting size
        let output_size = trial_buffer.len() as f64;
        pass_stats.value = output_size;

        // Track best result (closest to target)
        let diff = (output_size - pass_stats.target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_enc_stats = enc_stats;
            best_output = Some(trial_buffer);
        }

        // Check convergence
        let is_last = pass + 1 >= max_passes || pass_stats.is_converged();
        if is_last {
            break;
        }

        // Compute next quality to try
        pass_stats.compute_next_q();
    }

    // Write best result to output
    if let Some(output) = best_output {
        writer.extend_from_slice(&output);
    }

    Ok(best_enc_stats)
}

/// Encode with quality search to meet target PSNR.
/// Uses secant method to converge on target PSNR within DQ_LIMIT threshold.
#[allow(clippy::too_many_arguments)]
fn encode_with_psnr_search(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    let mut pass_stats = PassStats::new_for_psnr(params.target_psnr, params.lossy_quality, 1, 100);

    let max_passes = (params.method + 3).max(6) as usize;
    let mut best_output: Option<Vec<u8>> = None;
    let mut best_enc_stats = super::api::EncodeStats::default();
    let mut best_diff = f64::MAX;

    for pass in 0..max_passes {
        stop.check().map_err(|e| at!(EncodeError::from(e)))?;

        let mut trial_buffer = Vec::new();
        let mut trial_encoder = Vp8Encoder::new(&mut trial_buffer);

        let mut trial_params = params.clone();
        trial_params.lossy_quality = libm::roundf(pass_stats.q).clamp(0.0, 100.0) as u8;

        let enc_stats = trial_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        )?;

        // Use "All" PSNR (index 3) as the convergence metric
        let psnr_value = f64::from(enc_stats.psnr[3]);
        pass_stats.value = psnr_value;

        // Track best result (closest PSNR to target)
        let diff = (psnr_value - pass_stats.target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_enc_stats = enc_stats;
            best_output = Some(trial_buffer);
        }

        let is_last = pass + 1 >= max_passes || pass_stats.is_converged();
        if is_last {
            break;
        }

        pass_stats.compute_next_q();
    }

    if let Some(output) = best_output {
        writer.extend_from_slice(&output);
    }

    Ok(best_enc_stats)
}
